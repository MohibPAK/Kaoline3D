"""
Pauling's rules tab for the Ceramic Unit Cell Viewer.

render(B) draws one section at a time:
  0 Overview             how a ceramic structure forms, step by step
  1 Close packing        anion layers, octahedral / tetrahedral holes, filling patterns
  2 Rule 1               radius ratio -> coordination polyhedron (3-D, interactive)
  3 Rule 2               electrostatic valence principle, computed for the selected structure
  4 Rules 3 and 4        corner / edge / face sharing (3-D pair viewer + live analysis)
  5 Rule 5               parsimony, computed for every structure in the app

B is the structure currently chosen in the sidebar (output of crystal.build).
"""
import itertools
import math
import re
from fractions import Fraction

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import crystal as cx

S3 = math.sqrt(3)
HCP_H = 2 * math.sqrt(2 / 3)           # layer separation for touching spheres of radius 1

C_A, C_B, C_C = "#2f6bff", "#ff6a4d", "#9b6bff"     # layer colours
C_OCT, C_TET = "#ffd23f", "#3ee6b0"                  # hole colours
SUPERS = dict(zip("⁰¹²³⁴⁵⁶⁷⁸⁹", range(10)))


def _df(df):
    try:
        st.dataframe(df, hide_index=True, width="stretch")
    except TypeError:                       # older Streamlit
        st.dataframe(df, hide_index=True, use_container_width=True)


def _plot(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:                       # older Streamlit
        st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
#  Generic 3-D helpers
# ===========================================================================
@st.cache_data
def _unit_sphere(n=14):
    th = np.linspace(0, np.pi, n)
    ph = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
    T, P = np.meshgrid(th, ph, indexing="ij")
    V = np.c_[(np.sin(T) * np.cos(P)).ravel(), (np.sin(T) * np.sin(P)).ravel(), np.cos(T).ravel()]
    m = 2 * n
    I, J, K = [], [], []
    for a in range(n - 1):
        for b in range(m):
            p, q = a * m + b, a * m + (b + 1) % m
            r, s = (a + 1) * m + b, (a + 1) * m + (b + 1) % m
            I += [p, q]; J += [r, r]; K += [q, s]
    return V, np.array(I), np.array(J), np.array(K)


def spheres(centers, radii, color, name, opacity=1.0, hover=None, showlegend=True):
    """One Mesh3d trace holding many spheres (same colour)."""
    centers = np.atleast_2d(np.asarray(centers, float))
    radii = np.broadcast_to(np.asarray(radii, float), (len(centers),))
    V, I, J, K = _unit_sphere()
    X, Y, Z, TI, TJ, TK, TXT = [], [], [], [], [], [], []
    for n, (c, r) in enumerate(zip(centers, radii)):
        P = V * r + c
        X += P[:, 0].tolist(); Y += P[:, 1].tolist(); Z += P[:, 2].tolist()
        off = n * len(V)
        TI += (I + off).tolist(); TJ += (J + off).tolist(); TK += (K + off).tolist()
        if hover is not None:
            TXT += [hover[n]] * len(V)
    return go.Mesh3d(x=X, y=Y, z=Z, i=TI, j=TJ, k=TK, color=color, opacity=opacity, name=name,
                     showlegend=showlegend, flatshading=False, hovertext=TXT if hover is not None else None,
                     hoverinfo="text" if hover is not None else "skip",
                     lighting=dict(ambient=0.55, diffuse=0.8, specular=0.35, roughness=0.4))


def lines3d(segs, color, name, width=5, showlegend=False, dash=None):
    xs, ys, zs = [], [], []
    for a, b in segs:
        xs += [a[0], b[0], None]; ys += [a[1], b[1], None]; zs += [a[2], b[2], None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=color, width=width, dash=dash),
                        name=name, showlegend=showlegend, hoverinfo="skip")


def scene_fig(traces, height=520, eye=(1.5, 1.3, 0.9)):
    fig = go.Figure(traces)
    ax = dict(visible=False, showbackground=False)
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0), scene=dict(
        xaxis=ax, yaxis=ax, zaxis=ax, aspectmode="data", camera=dict(eye=dict(x=eye[0], y=eye[1], z=eye[2]))),
        legend=dict(orientation="h", y=0.02, x=0.0, bgcolor="rgba(255,255,255,0.6)"))
    return fig


# ===========================================================================
#  Section 1 - close packing and holes
# ===========================================================================
A1 = np.array([2.0, 0.0])
A2 = np.array([1.0, S3])
OFFSET = {"A": np.zeros(2), "B": np.array([1.0, S3 / 3]), "C": np.array([2.0, 2 * S3 / 3])}
CENTRE = np.array([1.0, S3 / 3])
LAYER_COLOUR = {"A": C_A, "B": C_B, "C": C_C}


def lattice_sites(letter, radius):
    out = []
    for i in range(-6, 7):
        for j in range(-6, 7):
            p = i * A1 + j * A2 + OFFSET[letter]
            if np.linalg.norm(p - CENTRE) <= radius:
                out.append((i, j, p))
    return out


def packing_figure(stacking, n_layers, fill, show_holes, anion_opacity):
    seq = ("AB" * 3 if stacking == "hcp" else "ABC" * 2)[:n_layers]
    traces, stats = [], dict(oct=0, tet=0, oct_f=0, tet_f=0)
    for k, L in enumerate(seq):
        pts = np.array([[p[0], p[1], k * HCP_H] for _, _, p in lattice_sites(L, 3.3)])
        traces.append(spheres(pts, 1.0, LAYER_COLOUR[L], f"Layer {k + 1} ({L})", anion_opacity,
                              [f"Anion, layer {k + 1} ({L})"] * len(pts)))
    oct_pts, oct_fill, tet_up, tet_dn, tet_up_f, tet_dn_f = [], [], [], [], [], []
    for k in range(len(seq) - 1):
        X, Y = seq[k], seq[k + 1]
        Z = next(c for c in "ABC" if c not in (X, Y))
        z0 = k * HCP_H
        for i, j, p in lattice_sites(Z, 2.7):
            filled = fill == "oct_all" or (fill == "oct_23" and (i - j + k) % 3 != 0)
            (oct_fill if filled else oct_pts).append([p[0], p[1], z0 + HCP_H / 2])
        for _, _, p in lattice_sites(Y, 2.7):          # 3 atoms below + 1 above : apex up
            filled = fill in ("tet_half", "tet_all")
            (tet_up_f if filled else tet_up).append([p[0], p[1], z0 + HCP_H / 4])
        for _, _, p in lattice_sites(X, 2.7):          # 3 atoms above + 1 below : apex down
            filled = fill == "tet_all"
            (tet_dn_f if filled else tet_dn).append([p[0], p[1], z0 + 3 * HCP_H / 4])
    r_o, r_t = math.sqrt(2) - 1, math.sqrt(1.5) - 1
    if show_holes or fill != "none":
        for pts, r, col, nm in [(oct_pts, r_o, C_OCT, "Empty octahedral hole"), (tet_up + tet_dn, r_t, C_TET, "Empty tetrahedral hole")]:
            if pts and show_holes:
                traces.append(spheres(pts, r, col, nm, 0.28, [nm] * len(pts)))
    if oct_fill:
        traces.append(spheres(oct_fill, r_o, C_OCT, "Cation in octahedral hole", 1.0, ["Cation, octahedral hole (CN 6)"] * len(oct_fill)))
    tf = tet_up_f + tet_dn_f
    if tf:
        traces.append(spheres(tf, r_t, C_TET, "Cation in tetrahedral hole", 1.0, ["Cation, tetrahedral hole (CN 4)"] * len(tf)))
    stats.update(oct=len(oct_pts) + len(oct_fill), oct_f=len(oct_fill), tet=len(tet_up + tet_dn) + len(tf), tet_f=len(tf))
    return scene_fig(traces, 560), seq, stats


FILL_LABELS = {
    "none": "None (just show the holes)",
    "oct_all": "All octahedral holes filled  →  rock-salt type (AX)",
    "tet_half": "Half of the tetrahedral holes (all apex-up)  →  zinc blende / wurtzite (AX)",
    "tet_all": "All tetrahedral holes filled  →  antifluorite type (A₂X)",
    "oct_23": "Two-thirds of the octahedral holes  →  corundum type (A₂X₃)",
}

FAMILY_TABLE = pd.DataFrame([
    ("Rock salt (NaCl, MgO)", "fcc  (ABC)", "all octahedral", "1 / 1"),
    ("Zinc blende (ZnS, SiC)", "fcc  (ABC)", "tetrahedral", "1 / 2"),
    ("Wurtzite (ZnO, AlN, BeO)", "hcp  (AB)", "tetrahedral", "1 / 2"),
    ("Antifluorite (Li₂O)", "fcc  (ABC)", "tetrahedral", "all"),
    ("NiAs type", "hcp  (AB)", "octahedral", "all"),
    ("Corundum (Al₂O₃)", "hcp  (AB)", "octahedral", "2 / 3"),
    ("Spinel (MgAl₂O₄)", "fcc  (ABC)", "tetrahedral + octahedral", "1/8 of T  +  1/2 of O"),
    ("Olivine (Mg₂SiO₄)", "≈ hcp", "octahedral + tetrahedral", "1/2 of O  +  1/8 of T"),
    ("Perovskite (BaTiO₃)", "fcc of Ba + O (AO₃ layers)", "octahedral", "1/4 of O (all surrounded by O only)"),
], columns=["Structure type", "Anion array", "Holes used", "Fraction filled"])


def section_packing():
    st.markdown("### 1 · Close packing and the holes it creates")
    st.markdown(
        "Anions (O²⁻, Cl⁻, S²⁻ …) are much bigger than cations, so the structure is first an **array of nearly "
        "close-packed anions**. Stack layers **A-B-A-B…** to get hexagonal close packing (hcp), or **A-B-C-A…** for "
        "cubic close packing (fcc). The stacking leaves two kinds of holes, and the small cations sit in them:\n\n"
        "* **octahedral hole**: 6 anions around it (3 below, 3 above), radius **0.414 R**\n"
        "* **tetrahedral hole**: 4 anions around it, radius **0.225 R**\n\n"
        "For N anions there are **N octahedral holes and 2N tetrahedral holes**.")
    c1, c2, c3 = st.columns([1.1, 1, 1])
    stacking = c1.radio("Stacking", ["hcp  (A-B-A-B)", "fcc  (A-B-C-A)"], key="pl_stack")
    stacking = "hcp" if stacking.startswith("hcp") else "fcc"
    n_layers = c2.slider("Number of anion layers", 2, 4, 3, key="pl_layers")
    anion_op = c3.slider("Anion opacity", 0.1, 1.0, 0.30, 0.05, key="pl_op")
    c4, c5 = st.columns([2, 1])
    fill = c4.selectbox("Fill the holes with cations", list(FILL_LABELS), format_func=FILL_LABELS.get, key="pl_fill")
    holes = c5.checkbox("Show empty holes", True, key="pl_holes")
    fig, seq, stats = packing_figure(stacking, n_layers, fill, holes, anion_op)
    _plot(fig)
    msg = f"Layers shown: **{' '.join(seq)}**.  Gold = octahedral holes ({stats['oct']} in this patch), mint = tetrahedral holes ({stats['tet']} in this patch)."
    if fill != "none":
        msg += f"  Filled: {stats['oct_f']} octahedral, {stats['tet_f']} tetrahedral."
    st.caption(msg + "  Cations are drawn at the size of the hole they fill, so their spheres just touch the anions.")
    st.markdown("**Which holes do real ceramics fill?**")
    _df(FAMILY_TABLE)
    st.info("**Why it matters:** the *fraction* of holes that is filled is fixed by the formula (charge balance), "
            "and **which kind** of hole is chosen is decided by the cation size: that is Pauling's rule 1, next.")


# ===========================================================================
#  Section 2 - Rule 1: radius ratio
# ===========================================================================
def poly_unit_vertices(cn):
    if cn == 2:
        v = [(0, 0, 1), (0, 0, -1)]
    elif cn == 3:
        v = [(math.cos(2 * math.pi * k / 3), math.sin(2 * math.pi * k / 3), 0) for k in range(3)]
    elif cn == 4:
        v = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    elif cn == 6:
        v = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    elif cn == 8:
        v = list(itertools.product((-1, 1), repeat=3))
    else:  # 12 : cuboctahedron
        v = [p for p in itertools.product((-1, 0, 1), repeat=3) if sorted(map(abs, p)) == [0, 1, 1]]
    v = np.array(v, float)
    return v / np.linalg.norm(v, axis=1)[:, None]


POLY_NAME = {2: "linear", 3: "triangular", 4: "tetrahedral", 6: "octahedral", 8: "cubic", 12: "cuboctahedral"}
CNS = [2, 3, 4, 6, 8, 12]


def edge_factor(cn):
    """Anion-anion distance / cation-anion distance for the ideal polyhedron."""
    v = poly_unit_vertices(cn)
    d = np.linalg.norm(v[:, None] - v[None], axis=2)
    return float(d[d > 1e-9].min())


def critical_ratio(cn):
    return 2.0 / edge_factor(cn) - 1.0       # anions (radius 1) just touch when edge = 2


def predict_cn(ratio):
    cn = 2
    for c in CNS:
        if ratio >= critical_ratio(c) - 1e-9:
            cn = c
    return cn


# (name, cation, r+, anion, r-, actual CN of the cation)   Shannon radii, Å
COMPOUNDS = [
    ("NaCl", "Na⁺ (VI)", 1.02, "Cl⁻", 1.81, 6),
    ("MgO", "Mg²⁺ (VI)", 0.72, "O²⁻", 1.40, 6),
    ("ZnS (zinc blende)", "Zn²⁺ (IV)", 0.60, "S²⁻", 1.84, 4),
    ("CsCl", "Cs⁺ (VIII)", 1.74, "Cl⁻", 1.81, 8),
    ("CaF₂", "Ca²⁺ (VIII)", 1.12, "F⁻", 1.33, 8),
    ("Al₂O₃", "Al³⁺ (VI)", 0.535, "O²⁻", 1.40, 6),
    ("TiO₂ (rutile)", "Ti⁴⁺ (VI)", 0.605, "O²⁻", 1.40, 6),
    ("ZrO₂ (cubic)", "Zr⁴⁺ (VIII)", 0.84, "O²⁻", 1.40, 8),
    ("SiO₂ (quartz)", "Si⁴⁺ (IV)", 0.26, "O²⁻", 1.40, 4),
    ("BeO", "Be²⁺ (IV)", 0.27, "O²⁻", 1.40, 4),
]


def ratio_table():
    rows = []
    for n, cat, rp, an, rm, cn in COMPOUNDS:
        r = rp / rm
        pr = predict_cn(r)
        rows.append({"Compound": n, "Cation": cat, "r₊ (Å)": rp, "Anion": an, "r₋ (Å)": rm,
                     "r₊ / r₋": round(r, 3), "Predicted CN": pr, "Actual CN": cn,
                     "Rule works?": "✔" if pr == cn else "✘"})
    return pd.DataFrame(rows)


def rule1_figure(ratio, cn):
    v = poly_unit_vertices(cn)
    d = 1.0 + ratio                                   # cation-anion contact (anion radius = 1)
    P = v * d
    a_a = d * edge_factor(cn)
    overlap = a_a < 2.0 - 1e-6
    col = "#e63946" if overlap else C_A
    segs = []
    dist = np.linalg.norm(P[:, None] - P[None], axis=2)
    emin = dist[dist > 1e-9].min()
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            if abs(dist[i, j] - emin) < 1e-6:
                segs.append((P[i], P[j]))
    tr = [spheres(P, 1.0, col, "Anions (radius 1)", 0.45, [f"Anion, distance to cation {d:.2f}"] * len(P)),
          spheres([[0, 0, 0]], ratio, C_OCT, f"Cation (radius {ratio:.3f})", 1.0, [f"Cation, r₊/r₋ = {ratio:.3f}"]),
          lines3d(segs, "#222", "Polyhedron edges", 4)]
    return scene_fig(tr, 520), a_a, overlap


def section_rule1():
    st.markdown("### 2 · Rule 1: the radius ratio sets the coordination polyhedron")
    st.markdown(
        "> *A coordination polyhedron of anions is formed around every cation. The cation–anion distance is the sum of "
        "the radii, and the coordination number is decided by the radius ratio r₊/r₋.*\n\n"
        "**Idea:** a cation wants as many anion neighbours as it can touch. If the anions would have to squeeze "
        "into each other, that CN is impossible. The limit is when the anions **just touch each other**:")
    st.markdown(
        "**Worked example, octahedron (CN 6).**  Anion radius = 1, so the cation–anion distance is d = 1 + r₊. "
        "The octahedron edge is d·√2. The anions touch when edge = 2:  (1 + r₊)·√2 = 2  →  "
        "**r₊ = 2/√2 − 1 = 0.414**. Below that, CN 6 is impossible.")
    crit = pd.DataFrame([{"CN": c, "Polyhedron": POLY_NAME[c], "Edge / bond length": round(edge_factor(c), 4),
                          "Minimum r₊/r₋ = 2/edge − 1": round(critical_ratio(c), 3),
                          "Stable range": f"{critical_ratio(c):.3f} – {critical_ratio(CNS[CNS.index(c) + 1]):.3f}" if c != 12 else "≥ 1.000"}
                         for c in CNS])
    _df(crit)

    st.markdown("**Try it:** drag the ratio, or load a real compound.")
    FREE = "— free slider —"
    names = [FREE] + [c[0] for c in COMPOUNDS]
    st.session_state.setdefault("r1_ratio", 0.414)

    def _load():
        pick = st.session_state["r1_pick"]
        if pick != FREE:
            row = next(c for c in COMPOUNDS if c[0] == pick)
            st.session_state["r1_ratio"] = round(row[2] / row[4], 3)

    c1, c2 = st.columns([1, 2])
    c1.selectbox("Load compound", names, key="r1_pick", on_change=_load)
    c1.selectbox("Polyhedron to draw", ["Predicted by the rule"] + [f"CN {c} – {POLY_NAME[c]}" for c in CNS], key="r1_cn")
    ratio = c2.slider("Radius ratio  r₊ / r₋", 0.05, 1.15, step=0.005, key="r1_ratio")
    pred = predict_cn(ratio)
    sel = st.session_state["r1_cn"]
    cn = pred if sel.startswith("Predicted") else int(sel.split()[1])
    fig, a_a, overlap = rule1_figure(ratio, cn)
    _plot(fig)
    if overlap:
        st.error(f"**CN {cn} is not possible at r₊/r₋ = {ratio:.3f}.** The anions would sit {a_a:.2f} apart "
                 f"(needs ≥ 2.00): they overlap (red). The rule predicts **CN {pred}** here.")
    elif abs(a_a - 2) < 0.02:
        st.success(f"**Perfect fit.** Anions touch each other ({a_a:.2f}) and also touch the cation: the limit of stability for CN {cn}.")
    elif cn == 12:
        st.success(f"**CN 12 works.** The cation touches every anion; the anions are {a_a:.2f} apart.")
    else:
        nxt = critical_ratio(CNS[CNS.index(cn) + 1])
        st.success(f"**CN {cn} works.** The cation touches every anion; the anions are {a_a:.2f} apart (> 2.00). "
                   f"Once r₊/r₋ passes {nxt:.3f}, the next coordination number becomes possible.")
    st.markdown("**Real compounds (Shannon radii):**")
    _df(ratio_table())
    st.warning("**Limits of the rule:** it treats ions as hard spheres. It fails when bonding is strongly covalent "
               "(Si–O, Be–O are partly covalent and prefer tetrahedra) or when polarisation is large. "
               "Treat it as a good first guess, not a law.")


# ===========================================================================
#  Section 3 - Rule 2: electrostatic valence principle
# ===========================================================================
def charge_of(sp):
    if sp.get("charge") is not None:
        return float(sp["charge"])
    qs = []
    for m in re.finditer(r"([⁰¹²³⁴⁵⁶⁷⁸⁹]*)([⁺⁻])", sp["label"]):
        n = int("".join(str(SUPERS[c]) for c in m.group(1))) if m.group(1) else 1
        qs.append(n if m.group(2) == "⁺" else -n)
    return float(np.mean(qs)) if qs else None


def _frac(x):
    f = Fraction(x).limit_denominator(12)
    return f"{f.numerator}/{f.denominator}" if f.denominator != 1 else f"{f.numerator}"


def valence_tables(B):
    sp, cnn = B["species"], B["cn"]
    q = [charge_of(s) for s in sp]
    if any(x is None for x, s in zip(q, sp) if s["role"] in ("cation", "anion")):
        return None
    s_of = {}
    cat_rows = []
    for i, s in enumerate(sp):
        if s["role"] == "cation" and cnn[i]["cn"] > 0:
            s_of[s["label"]] = q[i] / cnn[i]["cn"]
            cat_rows.append({"Cation site": s["label"], "Charge z": round(q[i], 2), "CN": cnn[i]["cn"],
                             "Bond strength s = z / CN": _frac(q[i] / cnn[i]["cn"]) + f"  ({q[i] / cnn[i]['cn']:.3f})"})
    an_rows, bars = [], []
    for j, s in enumerate(sp):
        if s["role"] != "anion":
            continue
        contrib = {lab: n * s_of.get(lab, 0.0) for lab, n in cnn[j]["partners"].items()}
        extra = 1.0 if s["label"].startswith("OH") else 0.0
        need = 2.0 if extra else -q[j]
        tot = sum(contrib.values()) + extra
        bonded = ", ".join(f"{lab} ×{n}" for lab, n in cnn[j]["partners"].items())
        if extra:
            bonded += ", H ×1"
        an_rows.append({"Anion site": s["label"], "Bonded to": bonded, "Σ s received": round(tot, 2),
                        "Anion charge": round(need, 2), "Difference": f"{tot - need:+.2f}",
                        "Rule 2": "✔" if abs(tot - need) <= 0.25 else ("≈  irregular polyhedron" if abs(tot - need) <= 0.5 else "✘  (> 0.5 off)")})
        bars.append((s["label"], contrib, extra, need))
    return pd.DataFrame(cat_rows), pd.DataFrame(an_rows), bars, {s["label"]: s["color"] for s in sp}


def section_rule2(B):
    st.markdown("### 3 · Rule 2: the electrostatic valence principle")
    st.markdown(
        "> *In a stable structure, the total strength of the valence bonds that reach an anion from its neighbouring "
        "cations equals the charge of that anion.*\n\n"
        "A cation of charge **z** with coordination number **CN** spreads its charge equally over its bonds, each "
        "with **bond strength  s = z / CN**. Every anion must collect bonds adding up to its own charge.")
    st.markdown("**Worked example, MgO (rock salt):**  Mg²⁺ has CN 6 → s = 2/6 = ⅓. "
                "Each O²⁻ has 6 Mg neighbours → Σs = 6 × ⅓ = **2** = charge of O²⁻ ✔.  "
                "**Quartz:** Si⁴⁺, CN 4 → s = 1; each O touches 2 Si → Σs = 2 ✔ (so every O must bridge exactly two tetrahedra).")
    st.markdown(f"**Live check for the structure selected in the sidebar:**  *{B['material']}*")
    res = valence_tables(B)
    if res is None or res[1].empty:
        st.info("Charges are not defined for this structure, so rule 2 cannot be evaluated.")
        return
    cat_df, an_df, bars, colours = res
    st.markdown("**Step 1: bond strength of each cation**")
    _df(cat_df)
    st.markdown("**Step 2: what each anion receives**")
    _df(an_df)
    fig = go.Figure()
    labs = [b[0] for b in bars]
    cat_labels = sorted({k for b in bars for k in b[1]})
    for k in cat_labels:
        fig.add_bar(x=labs, y=[b[1].get(k, 0.0) for b in bars], name=k, marker_color=colours.get(k, "#888"))
    if any(b[2] for b in bars):
        fig.add_bar(x=labs, y=[b[2] for b in bars], name="O–H bond (H)", marker_color="#bbbbbb")
    fig.add_scatter(x=labs, y=[b[3] for b in bars], mode="markers", name="Charge needed",
                    marker=dict(symbol="line-ew", size=36, line=dict(width=3, color="#111")))
    fig.update_layout(barmode="stack", height=380, yaxis_title="Σ bond strength reaching the anion",
                      margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25))
    _plot(fig)
    st.caption("Black bars = the anion's charge. A stack that reaches the bar satisfies rule 2. "
               "Differences of up to ±0.25 are normal in real crystals (bond lengths adjust), and up to ±0.5 happens next to "
               "irregular polyhedra (Ca in amphiboles, Li, Ba, K), because the simple rule gives every bond of a cation "
               "the same strength even when some bonds are much longer than others. "
               "Hydroxyl oxygens also receive 1 from their O–H bond. Mixed Si/Al sites use the average charge.")


# ===========================================================================
#  Section 4 - Rules 3 and 4: sharing
# ===========================================================================
def _tetra():
    v = poly_unit_vertices(4)
    return v, [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]


def _octa():
    v = poly_unit_vertices(6)
    return v, [(a, b, c) for a in (0, 1) for b in (2, 3) for c in (4, 5)]


def pair_geometry(kind, mode):
    """Two regular polyhedra (circumradius 1) sharing a corner, an edge or a face."""
    V, faces = _tetra() if kind == "tetrahedron" else _octa()
    dist = np.linalg.norm(V[:, None] - V[None], axis=2)
    a = float(dist[dist > 1e-9].min())
    if mode == "corner":
        g = V[0]
    elif mode == "edge":
        i, j = [(i, j) for i in range(len(V)) for j in range(i + 1, len(V)) if abs(dist[i, j] - a) < 1e-6][0]
        g = (V[i] + V[j]) / 2
    else:
        fidx = faces[0]
        g = V[list(fidx)].mean(axis=0)
    if mode == "face":
        n = g / np.linalg.norm(g)
        V2 = V - 2 * ((V - g) @ n)[:, None] * n
    else:
        V2 = 2 * g - V
    c2 = 2 * g
    shared = [i for i in range(len(V)) if np.min(np.linalg.norm(V2 - V[i], axis=1)) < 1e-6]
    return V, V2, faces, c2, shared, a


def sharing_table():
    rows = []
    for kind in ("tetrahedron", "octahedron"):
        base = None
        for mode in ("corner", "edge", "face"):
            V, V2, f, c2, sh, a = pair_geometry(kind, mode)
            d = float(np.linalg.norm(c2))
            base = base or d
            rows.append({"Polyhedra": "tetrahedra" if kind == "tetrahedron" else "octahedra", "Sharing": mode, "Shared anions": len(sh),
                         "Cation–cation distance (× edge)": round(d / a, 3),
                         "Relative to corner-sharing": round(d / base, 2),
                         "Coulomb repulsion  ∝ 1/d  (corner = 1)": round(base / d, 2)})
    return pd.DataFrame(rows)


def pair_figure(kind, mode):
    V, V2, faces, c2, shared, a = pair_geometry(kind, mode)
    tr = []
    for verts, col, nm in [(V, C_A, "Polyhedron 1"), (V2, C_B, "Polyhedron 2")]:
        tr.append(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                            i=[f[0] for f in faces], j=[f[1] for f in faces], k=[f[2] for f in faces],
                            color=col, opacity=0.28, name=nm, hoverinfo="skip", flatshading=True))
        dd = np.linalg.norm(verts[:, None] - verts[None], axis=2)
        e = dd[dd > 1e-9].min()
        tr.append(lines3d([(verts[i], verts[j]) for i in range(len(verts)) for j in range(i + 1, len(verts))
                           if abs(dd[i, j] - e) < 1e-6], col, nm + " edges", 5))
    own1 = [i for i in range(len(V)) if i not in shared]
    shared_pts = V[shared]
    own2 = [p for p in V2 if np.min(np.linalg.norm(V - p, axis=1)) > 1e-6]
    tr.append(spheres(V[own1], 0.13, C_A, "Anion (polyhedron 1 only)", 1.0, ["Anion"] * len(own1)))
    tr.append(spheres(own2, 0.13, C_B, "Anion (polyhedron 2 only)", 1.0, ["Anion"] * len(own2)))
    tr.append(spheres(shared_pts, 0.17, "#111111", f"Shared anion(s): {len(shared)}", 1.0, ["Shared anion"] * len(shared)))
    tr.append(spheres([[0, 0, 0], c2], 0.16, C_OCT, "Cations", 1.0, ["Cation"] * 2))
    tr.append(lines3d([([0, 0, 0], c2)], "#e63946", "Cation–cation distance", 7, True, "dash"))
    return scene_fig(tr, 520), float(np.linalg.norm(c2)), a, len(shared)


@st.cache_data(show_spinner="Analysing polyhedron sharing …")
def _sharing_cached(key, _B):
    return sharing_analysis(_B)


def sharing_analysis(B):
    """For each cation species: neighbours per cation that share 1 (corner), 2 (edge), 3+ (face) anions."""
    M, sp, cut = B["M"], B["species"], B["cut"]
    cat = [i for i, s in enumerate(sp) if s["role"] == "cation"]
    an = [j for j, s in enumerate(sp) if s["role"] == "anion"]
    A_img = {j: cx.images(sp[j]["orbit"]) @ M for j in an}
    C_img = {k: cx.images(sp[k]["orbit"]) @ M for k in cat}
    agg = {}
    for i in cat:
        orb = sp[i]["orbit"]
        for p in orb:
            pc = p @ M
            shell = []
            for j in an:
                A = A_img[j]
                sel = A[np.linalg.norm(A - pc, axis=1) < cut[i, j] + 1e-6]
                if len(sel):
                    shell.append((j, sel))
            if not shell:
                continue
            for k in cat:
                C = C_img[k]
                dd = np.linalg.norm(C - pc, axis=1)
                for q, dq in zip(C[(dd > 1e-3) & (dd < 6.5)], dd[(dd > 1e-3) & (dd < 6.5)]):
                    n = sum(int((np.linalg.norm(sel - q, axis=1) < cut[k, j] + 1e-6).sum()) for j, sel in shell)
                    if n:
                        kind = {1: "corner", 2: "edge"}.get(n, "face")
                        rec = agg.setdefault((i, k, kind), [0, 0.0])
                        rec[0] += 1; rec[1] += dq
    rows = []
    for i in cat:
        mult = len(sp[i]["orbit"])
        for k in cat:
            c = {kind: agg.get((i, k, kind), [0, 0.0]) for kind in ("corner", "edge", "face")}
            if not any(v[0] for v in c.values()):
                continue
            dist = " / ".join(f"{v[1] / v[0]:.2f}" if v[0] else "–" for v in c.values())
            rows.append({"From cation": sp[i]["label"], "To cation": sp[k]["label"],
                         "Corner-sharing": round(c["corner"][0] / mult, 2),
                         "Edge-sharing": round(c["edge"][0] / mult, 2),
                         "Face-sharing": round(c["face"][0] / mult, 2),
                         "M–M distance (Å)  corner / edge / face": dist})
    return pd.DataFrame(rows)


def section_sharing(B):
    st.markdown("### 4 · Rules 3 and 4: how polyhedra are joined")
    st.markdown(
        "> **Rule 3.** *Sharing of edges, and especially faces, between coordination polyhedra lowers the stability of a "
        "structure.*\n>\n> **Rule 4.** *In a crystal with several cations, those with high charge and small CN tend "
        "**not** to share polyhedron elements with each other.*\n\n"
        "**Why:** every shared anion drags the two cations closer, and two positive charges repel as 1/d. "
        "Compare the three ways two polyhedra can touch:")
    c1, c2 = st.columns([1, 1])
    kind = c1.radio("Polyhedron", ["tetrahedron", "octahedron"], horizontal=True, key="sh_kind")
    mode = c2.radio("Sharing", ["corner", "edge", "face"], horizontal=True, key="sh_mode")
    fig, d, a, n = pair_figure(kind, mode)
    _plot(fig)
    st.caption(f"{kind.capitalize()}s sharing a **{mode}** ({n} shared anion{'s' if n > 1 else ''}): cation–cation distance "
               f"= **{d / a:.3f} × edge length**.")
    _df(sharing_table())
    st.markdown("Face-sharing tetrahedra put the two cations only ⅓ as far apart as corner-sharing ones: "
                "the repulsion is **3 ×** larger. That is why silicates are almost always corner-sharing "
                "(SiO₄ units) and why edge/face sharing appears mainly for low-charge cations (Mg²⁺, Na⁺) or when the cations "
                "shift off-centre (for example the Al pairs in corundum, which share faces along c).")

    st.markdown(f"#### Live analysis: *{B['material']}*")
    sig = (B["struct"], B["material"], B["shape"],
           round(float(sum(np.abs(s["orbit"]).sum() for s in B["species"])), 4))
    df = _sharing_cached(sig, B)
    if df.empty:
        st.info("No cation polyhedra of this structure share anions in the way rule 3 describes "
                "(isolated polyhedra, or no cation sites defined).")
    else:
        st.markdown("Average number of **neighbouring polyhedra per cation**, by how many anions they share "
                    "(1 = corner, 2 = edge, ≥ 3 = face):")
        _df(df)
        sp = B["species"]
        strength = {}
        for i, s_ in enumerate(sp):
            z = charge_of(s_)
            if s_["role"] == "cation" and z and B["cn"][i]["cn"]:
                strength[s_["label"]] = (z / B["cn"][i]["cn"], z, B["cn"][i]["cn"])
        strong = {k: v for k, v in strength.items() if v[0] >= 0.75}
        if not strong:
            st.info("Every cation here has a small bond strength (z/CN < 0.75), so rule 4 puts no strong restriction "
                    "on them: edge sharing is perfectly normal (rock-salt MgO is built from edge-sharing octahedra).")
        else:
            names = ", ".join(f"{k} (z/CN = {v[0]:.2f})" for k, v in strong.items())
            sub = df[df["From cation"].isin(strong) & df["To cation"].isin(strong)]
            bad = float(sub["Edge-sharing"].sum() + sub["Face-sharing"].sum())
            if bad == 0:
                st.success(f"**Rule 4 ✔:** the strongly bonded cations, {names}, share **only corners** "
                           "(or nothing) with one another.")
            else:
                st.warning(f"**Strongly bonded cations** ({names}) share edges/faces with each other "
                           f"({bad:.2f} per cation). Rule 4 is a tendency, so check how the crystal compensates "
                           "(distorted polyhedra, longer M–M distances) in the table.")
    st.caption("Tip: pick quartz, forsterite, kaolinite or spinel in the sidebar and compare: SiO₄ shares only corners, "
               "while the Mg/Al octahedra share edges.")


# ===========================================================================
#  Section 5 - Rule 5: parsimony
# ===========================================================================
@st.cache_data(show_spinner="Comparing all structures …")
def parsimony_table():
    rows = []
    for sname, S in cx.STRUCTURES.items():
        for mname in S["materials"]:
            try:
                Bq = cx.build(sname, mname)
            except Exception:
                continue
            sp = [s for s in Bq["species"] if s["role"] in ("cation", "anion")]
            env = {(s["role"], s["el"], Bq["cn"][i]["cn"]) for i, s in enumerate(Bq["species"]) if s["role"] in ("cation", "anion")}
            poly = {(s["el"], Bq["cn"][i]["cn"]) for i, s in enumerate(Bq["species"]) if s["role"] == "cation"}
            try:
                formula = cx.formula_and_z(Bq)[0]
            except Exception:
                formula = ""
            rows.append({"Material": mname, "Formula": formula,
                         "Elements": len({s["el"] for s in sp}),
                         "Atoms in cell": int(sum(s["mult"] for s in sp)),
                         "Crystallographic sites": len(sp),
                         "Coordination environments": len(env),
                         "Cation polyhedra types": len(poly)})
    return pd.DataFrame(rows)


def section_parsimony():
    st.markdown("### 5 · Rule 5: the principle of parsimony")
    st.markdown(
        "> *The number of essentially different kinds of constituents in a crystal tends to be small.*\n\n"
        "Even a crystal with dozens of atoms in its cell is built from **very few different building blocks**: a "
        "couple of coordination polyhedra repeated by the symmetry. Nature reuses the same SiO₄ tetrahedron and "
        "MgO₆ octahedron over and over. Below, for every structure in this app: how many atoms the cell holds, "
        "how many crystallographically different sites there are, and how many truly different "
        "coordination environments (element + CN) they reduce to.")
    df = parsimony_table()
    _df(df)
    fig = go.Figure()
    fig.add_bar(x=df["Material"], y=df["Atoms in cell"], name="Atoms in the cell", marker_color="#c9d3ea")
    fig.add_bar(x=df["Material"], y=df["Crystallographic sites"], name="Different sites", marker_color=C_A)
    fig.add_bar(x=df["Material"], y=df["Coordination environments"], name="Different environments", marker_color=C_B)
    fig.update_layout(barmode="group", height=420, yaxis_type="log", yaxis_title="count (log scale)",
                      margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1))
    _plot(fig)
    st.caption("Log scale. The pink bars stay low (typically 3–6) while the atom count climbs into the hundreds.")


# ===========================================================================
#  Overview and entry point
# ===========================================================================
def section_overview():
    st.markdown("### How does a ceramic structure form?  A step-by-step story")
    st.markdown(
        "A ceramic is a compound of a metal (or semi-metal) and a non-metal, usually oxygen. Its crystal structure is "
        "not random: a handful of simple rules decide it. Follow the steps below; each has its own page "
        "(use the selector above).")
    steps = pd.DataFrame([
        ("1", "Atoms become ions", "Metals give up electrons, oxygen takes them. Result: small cations (+) and large anions (−).", "Radii from the ionic-radius table", "—"),
        ("2", "Large anions pack", "O²⁻ is ~2× bigger than most cations, so the oxygens form a close-packed array (hcp or fcc).", "Close packing", "Page 1"),
        ("3", "Holes appear", "Stacking leaves octahedral (CN 6) and tetrahedral (CN 4) holes between the anions.", "Close packing", "Page 1"),
        ("4", "Cations pick a hole", "The radius ratio r₊/r₋ decides which polyhedron fits and how many anions touch the cation.", "Rule 1", "Page 2"),
        ("5", "Charges must balance", "Each cation spreads its charge over its bonds (s = z/CN); each anion must receive its own charge.", "Rule 2", "Page 3"),
        ("6", "Polyhedra connect", "Polyhedra join by corners (best), edges or faces (worst): sharing pulls like charges together.", "Rules 3 & 4", "Page 4"),
        ("7", "Nature repeats", "The structure is built from few different polyhedra, repeated by symmetry.", "Rule 5", "Page 5"),
    ], columns=["Step", "What happens", "In words", "Rule", "Where"])
    _df(steps)
    st.markdown("**Pauling's five rules at a glance** (1929, for ionic crystals):")
    _df(pd.DataFrame([
        ("1", "Radius ratio", "Anions surround each cation in a polyhedron; CN from r₊/r₋", "0.225 → tetrahedron, 0.414 → octahedron, 0.732 → cube"),
        ("2", "Electrostatic valence", "Σ bond strengths at an anion = its charge; s = z/CN", "Quartz: s(Si–O) = 1, two Si per O"),
        ("3", "Sharing polyhedra", "Corners are fine; edges and especially faces destabilise", "Face-sharing tetrahedra: repulsion 3× larger"),
        ("4", "Different cations", "High-charge, low-CN cations avoid sharing", "Si⁴⁺ tetrahedra share only corners"),
        ("5", "Parsimony", "Few kinds of polyhedra, repeated", "Dozens of atoms, ~3 environments"),
    ], columns=["Rule", "Name", "Statement", "Example / number"]))
    st.info("Pauling's rules work best for **ionic** compounds (oxides, halides). In silicates the Si–O bond is about "
            "50 % covalent, so rule 1 under-predicts (Si⁴⁺ prefers CN 4, not 3), but rules 2–4 still describe the "
            "structures very well.")


SECTIONS = {
    "Overview": "0 · Overview",
    "pack": "1 · Close packing & holes",
    "r1": "Rule 1 · Radius ratio",
    "r2": "Rule 2 · Valence balance",
    "r34": "Rules 3 & 4 · Sharing",
    "r5": "Rule 5 · Parsimony",
}


def render(B):
    st.subheader("How ceramic structures form, and how Pauling's rules decide them")
    sel = st.radio("Section", list(SECTIONS), format_func=SECTIONS.get, horizontal=True, key="pl_section",
                   label_visibility="collapsed")
    st.divider()
    if sel == "Overview":
        section_overview()
    elif sel == "pack":
        section_packing()
    elif sel == "r1":
        section_rule1()
    elif sel == "r2":
        section_rule2(B)
    elif sel == "r34":
        section_sharing(B)
    else:
        section_parsimony()
