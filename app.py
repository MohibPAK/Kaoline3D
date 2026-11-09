"""
Interactive unit-cell viewer for ceramic crystal structures.

Run with:   streamlit run app.py
Needs crystal.py in the same folder.
"""
import itertools
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import crystal as cx
import pauling

st.set_page_config(page_title="Ceramic Unit Cell Viewer", page_icon="🧊", layout="wide")

CAMERAS = {
    "Free (drag to rotate)": dict(eye=(1.7, 1.5, 1.2), up=(0, 0, 1)),
    "Isometric": dict(eye=(1.7, 1.7, 1.7), up=(0, 0, 1)),
    "Side view (along x)": dict(eye=(2.4, 0.0, 0.0), up=(0, 0, 1)),
    "Side view (along y)": dict(eye=(0.0, 2.4, 0.0), up=(0, 0, 1)),
    "Top view (along z / c)": dict(eye=(0.0, 0.0, 2.4), up=(0, 1, 0)),
}


def show_df(df):
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except Exception:  # older Streamlit
        st.dataframe(df, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------
#  Plot helpers
# --------------------------------------------------------------------------
def spheres_trace(centers, radius, color, name, hovers, opacity, showlegend=True):
    V, F = cx.unit_sphere()
    nv = len(V)
    verts = (V[None] * radius + centers[:, None, :]).reshape(-1, 3)
    faces = (F[None] + (np.arange(len(centers)) * nv)[:, None, None]).reshape(-1, 3)
    hover = np.repeat(np.array(hovers, dtype=object), nv)
    return go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color, opacity=opacity, name=name, showlegend=showlegend,
        hovertext=list(hover), hoverinfo="text", flatshading=False,
        lighting=dict(ambient=0.55, diffuse=0.85, specular=0.35, roughness=0.45, fresnel=0.1),
        lightposition=dict(x=200, y=300, z=400),
    )


def lines_xyz(segments):
    xs, ys, zs = [], [], []
    for p, q in segments:
        for r in (p, q, (None, None, None)):
            xs.append(r[0]); ys.append(r[1]); zs.append(r[2])
    return xs, ys, zs


def cell_corners(M):
    return np.array(list(itertools.product((0, 1), repeat=3)), float) @ M


def cell_edges(M):
    fr = np.array(list(itertools.product((0, 1), repeat=3)), float)
    c = fr @ M
    segs = []
    for i, j in itertools.combinations(range(8), 2):
        if np.sum(np.abs(fr[i] - fr[j]) > 0) == 1:
            segs.append((c[i], c[j]))
    return lines_xyz(segs)


def cell_faces(M):
    c = cell_corners(M)
    quads = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    tri = np.array([t for q in quads for t in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))])
    return go.Mesh3d(x=c[:, 0], y=c[:, 1], z=c[:, 2], i=tri[:, 0], j=tri[:, 1], k=tri[:, 2],
                     color="#7aa6ff", opacity=0.07, hoverinfo="skip", showlegend=False,
                     name="Cell faces")


def prism_edges(M):
    bot, top = cx.hex_prism_vertices(M)
    segs = []
    for k in range(6):
        k2 = (k + 1) % 6
        segs += [(bot[k], bot[k2]), (top[k], top[k2]), (bot[k], top[k])]
    return lines_xyz(segs)


def prism_faces(M):
    bot, top = cx.hex_prism_vertices(M)
    centre_b, centre_t = np.zeros(3), M[2]
    pts = np.vstack([bot, top, centre_b, centre_t])           # 0-5 bottom, 6-11 top, 12/13 centres
    tri = []
    for k in range(6):
        k2 = (k + 1) % 6
        tri += [(12, k, k2), (13, 6 + k, 6 + k2), (k, k2, 6 + k2), (k, 6 + k2, 6 + k)]
    tri = np.array(tri)
    return go.Mesh3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], i=tri[:, 0], j=tri[:, 1], k=tri[:, 2],
                     color="#7aa6ff", opacity=0.07, hoverinfo="skip", showlegend=False, name="Cell faces")


def rotation_frames(eye, n=72):
    r, z = float(np.hypot(eye[0], eye[1])), float(eye[2])
    phi0 = float(np.arctan2(eye[1], eye[0]))
    if r < 0.3:  # looking straight down: rotating about z would do nothing
        r, z, phi0 = 2.4, 1.2, np.pi / 4
    frames = []
    for k in range(n + 1):
        phi = phi0 + 2 * np.pi * k / n
        cam = dict(eye=dict(x=r * np.cos(phi), y=r * np.sin(phi), z=z), up=dict(x=0, y=0, z=1))
        frames.append(go.Frame(name=str(k), layout=dict(scene=dict(camera=cam))))
    return frames


def hover_text(label, wy, p, M, i=None, vacant=False):
    """Hover card for one site; i = species index (adds site geometry + bond directions)."""
    xyz = p @ M
    lines = [f"<b>{label}</b>",
             f"Wyckoff {wy} · {cx.POSITION_KIND[cx.n_contacts(p, B['shape'], M)]} site · counts {cx.share_in(p, B['shape'], M):g} in this cell",
             f"frac (u,v,w) = ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})",
             f"xyz = ({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}) Å",
             "──────────"]
    if vacant:
        lines += cx.vacant_description(B, p)
        lines.append("──────────")
    elif i is not None:
        lines += cx.site_description(B, i, p)
        lines.append("──────────")
    lines += cx.position_direction(p, B)
    if i is not None and not vacant:
        lines += cx.bond_directions(B, i, p)
    return "<br>".join(lines)


# --------------------------------------------------------------------------
#  Sidebar
# --------------------------------------------------------------------------
st.title("🧊 Ceramic Unit Cell Viewer")

tab_viewer, tab_pauling = st.tabs(["🧊 Unit cell viewer", "📘 How structures form · Pauling's rules"])

with tab_viewer:
    st.caption("Drag to rotate, scroll to zoom. Hover any atom to see its position, its crystal direction "
               "[uvw] from the origin, and the directions of its bonds. "
               "Atoms are generated from the real space-group symmetry operations.")

    with st.sidebar:
        st.header("Structure")
        cat_name = st.selectbox("Category", list(cx.CATEGORIES))
        s_name = st.selectbox("Structure type", cx.CATEGORIES[cat_name])
        S = cx.STRUCTURES[s_name]
        m_name = st.selectbox("Material", list(S["materials"]))
        mat = S["materials"][m_name]

        slider_val = None
        if "slider" in mat:
            sl = mat["slider"]
            slider_val = st.slider(sl["label"], float(sl["min"]), float(sl["max"]),
                                   float(sl["default"]), float(sl["step"]))
        is_hex = cx.cell_kind(mat["cell"]) == "hexagonal"
        shape = "cell"
        if is_hex:
            pick = st.radio("Cell shape", ["Hexagonal prism (3 cells, proper hcp cell)",
                                           "Rhombic cell (a = b, γ = 120°)"])
            shape = "hex" if pick.startswith("Hexagonal") else "cell"
        shift = st.checkbox("Shift origin by (½, ½, ½)", False,
                            help="Same crystal, different choice of cell origin.")

    B = cx.build(s_name, m_name, shift=shift, slider=slider_val, shape=shape)
    M = B["M"]
    species = B["species"]
    labels = [s["label"] for s in species]

    with st.sidebar:
        st.header("Display")
        shown = st.multiselect("Show species", labels, default=labels)
        style = st.radio("Atom style", ["Ball-and-stick", "Space-filling (to scale)"])
        if style == "Ball-and-stick":
            ball = st.slider("Ball size (× ionic radius)", 0.10, 0.80, 0.35, 0.05)
        else:
            ball = 1.0
        opacity = st.slider("Atom opacity", 0.15, 1.0, 1.0 if style.startswith("Ball") else 0.85, 0.05)
        show_bonds = st.checkbox("Nearest-neighbour bonds", True)
        show_ghost = st.checkbox("Complete bonds with neighbours outside the cell", False,
                                 help="Adds translucent periodic images so atoms on the cell boundary "
                                      "show their full coordination.")
        show_edges = st.checkbox("Unit-cell edges", True)
        show_faces = st.checkbox("Translucent cell faces", False,
                                 help="Looks nice, but in Plotly the faces sit in front of the atoms and block hover. "
                                      "Turn off if some atoms show no hover card.")
        show_axes = st.checkbox("Axes x, y, z and origin O", True)
        show_labels = st.checkbox("Site labels", False)
        show_vacant = st.checkbox("Vacant interstitial sites", False, disabled=not B["vacant"],
                                  help="Empty tetrahedral / octahedral holes" if B["vacant"]
                                  else "No vacant-site set is defined for this structure")
        show_grid = st.checkbox("Scene axes & grid (Å)", False)

        st.header("Camera")
        cam_name = st.selectbox("View", list(CAMERAS))

        st.header("Coordination polyhedron")
        poly_options = ["None"] + [s["label"] for s, c in zip(species, B["cn"]) if c["cn"] >= 3]
        poly_choice = st.selectbox("Highlight around", poly_options)

        st.header("Crystal direction [uvw]")
        st.caption("Draws a line through the cell origin. Set all three to 0 to hide it.")
        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            du = int(st.number_input("u", -2, 2, 0, 1))
        with dc2:
            dv = int(st.number_input("v", -2, 2, 0, 1))
        with dc3:
            dw = int(st.number_input("w", -2, 2, 0, 1))

    visible = {i for i, s in enumerate(species) if s["label"] in shown}

    # --------------------------------------------------------------------------
    #  Metrics
    # --------------------------------------------------------------------------
    formula, Z = cx.formula_and_z(B)
    zf = B["zfac"]                                   # 3 for the hexagonal prism, else 1
    n_atoms = zf * sum(s["mult"] for s in species)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Lattice", cx.lattice_string(B["cell"]))
    c2.metric("Formula / Z", f"{formula} / {Z * zf}")
    c3.metric("Space group", B["sg"].split(" (")[0])
    c4.metric("Atoms per cell", f"{n_atoms}")
    c5.metric("Density", f"{cx.density(B):.2f} g/cm³")

    # --------------------------------------------------------------------------
    #  Key properties: atoms/cell, CN, cation:anion ratio, lattice parameters, density
    # --------------------------------------------------------------------------
    def crystal_system(cell):
        a, b, c, al, be, ga = cell
        k = cx.cell_kind(cell)
        if k != "other":
            return k
        if al == be == ga == 90:
            return "orthorhombic"
        if al == ga == 90:
            return "monoclinic"
        return "lower symmetry"


    def key_property_tables():
        a, b, c, al, be, ga = B["cell"]
        V = B["volume"] * zf
        cats = [s for s in species if s["role"] == "cation"]
        ans = [s for s in species if s["role"] == "anion"]
        n_cat = sum(s["mult"] for s in cats) * zf
        n_an = sum(s["mult"] for s in ans) * zf
        g = math.gcd(int(n_cat), int(n_an)) or 1
        mass_cell = sum(s["mult"] * s["mass"] for s in species) * zf
        z_cell = Z * zf
        m_formula = mass_cell / z_cell
        rho = cx.density(B)

        def tally(group):
            tot = {}
            for s in group:
                tot[s["el"]] = tot.get(s["el"], 0) + s["mult"] * zf
            return " + ".join(f"{n} {el}" for el, n in tot.items()) or "none"

        cn_by_ion = {}
        for sp_, cn_ in zip(species, B["cn"]):
            cn_by_ion.setdefault(sp_["label"].split(" ")[0], set()).add(cn_["cn"])
        cn_txt = "; ".join(f"{ion}: {', '.join(str(v) for v in sorted(vals))}" for ion, vals in cn_by_ion.items())
        by_species, cat_txt, an_txt = tally(species), tally(cats), tally(ans)

        props = [
            ("Atoms per unit cell", f"{n_atoms}", f"{by_species}  (atoms shared between cells counted by their fraction inside this cell)"),
            ("Coordination number", cn_txt, "number of nearest oppositely charged neighbours; per-site details in the table below"),
            ("Cation : anion ratio", f"{int(n_cat // g)} : {int(n_an // g)}",
             f"cation atoms {n_cat} ({cat_txt}) : anion atoms {n_an} ({an_txt}) per cell"),
            ("Lattice parameters",
             f"a = {a:.4f} Å, b = {b:.4f} Å, c = {c:.4f} Å  |  α = {al:g}°, β = {be:g}°, γ = {ga:g}°",
             f"{crystal_system(B['cell'])} lattice; volume of the drawn cell V = {V:.2f} Å³"
             + (" (hexagonal prism = 3 rhombic cells)" if zf == 3 else "")),
            ("Theoretical density", f"{rho:.3f} g/cm³",
             f"ρ = Z·M / (N_A·V) = {z_cell} × {m_formula:.2f} g/mol ÷ (6.022×10²³ mol⁻¹ × {V:.2f}×10⁻²⁴ cm³)"),
        ]
        cn_rows = []
        for i, (s, c) in enumerate(zip(species, B["cn"])):
            cn_rows.append({"Species": s["label"], "Role": s["role"], "Atoms per cell": s["mult"] * zf,
                            "CN": c["cn"],
                            "Neighbours": ", ".join(f"{k} ×{v}" for k, v in c["partners"].items()),
                            "Bond length (Å)": f"{c['dmin']:.3f}" if abs(c['dmin'] - c['dmean']) < 5e-4
                            else f"{c['dmin']:.3f} – {c['dmean']:.3f} (min – mean)",
                            "Geometry": cx.site_geometry(B, i)[0]})
        return pd.DataFrame(props, columns=["Property", "Value", "Working"]), pd.DataFrame(cn_rows)


    with st.expander("Key properties of this structure", expanded=True):
        prop_df, cn_df = key_property_tables()
        st.table(prop_df.set_index("Property"))
        show_df(cn_df)

    # --------------------------------------------------------------------------
    #  Figure
    # --------------------------------------------------------------------------
    fig = go.Figure()
    if B["shape"] == "hex":
        bbox_pts = list(cx.hex_prism_vertices(M))
    else:
        bbox_pts = [cell_corners(M)]

    if show_faces:
        fig.add_trace(prism_faces(M) if B["shape"] == "hex" else cell_faces(M))

    for i, s in enumerate(species):
        if i not in visible:
            continue
        pos = s["drawn"]
        cart = pos @ M
        bbox_pts.append(cart)
        fig.add_trace(spheres_trace(cart, s["radius"] * ball, s["color"], s["label"],
                                    [hover_text(s["label"], s["wy"], p, M, i) for p in pos], opacity))

    segs, gsegs, gatoms = cx.bonds(B, visible, ghosts=show_ghost)
    if show_bonds and segs:
        bx, by, bz = lines_xyz(segs)
        fig.add_trace(go.Scatter3d(x=bx, y=by, z=bz, mode="lines", name="Bonds",
                                   line=dict(color="#555", width=3), hoverinfo="skip"))

    if show_ghost:
        for i, pos in gatoms.items():
            s = species[i]
            cart = pos @ M
            bbox_pts.append(cart)
            fig.add_trace(spheres_trace(cart, s["radius"] * ball, s["color"], s["label"] + " (image)",
                                        [f"<b>{s['label']}</b><br>periodic image outside the cell"] * len(pos),
                                        0.28, showlegend=False))
        if gsegs:
            gx, gy, gz = lines_xyz(gsegs)
            fig.add_trace(go.Scatter3d(x=gx, y=gy, z=gz, mode="lines", name="Bonds (outside cell)",
                                       line=dict(color="#999", width=2), hoverinfo="skip", showlegend=False))

    if show_edges:
        ex, ey, ez = prism_edges(M) if B["shape"] == "hex" else cell_edges(M)
        fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines", name="Unit cell",
                                   line=dict(color="#111", width=2), hoverinfo="skip"))

    if show_axes:
        for k, (col, txt) in enumerate((("#e63946", "x"), ("#2a9d4a", "y"), ("#1d6fd8", "z"))):
            end = M[k] * 1.25
            fig.add_trace(go.Scatter3d(
                x=[0, end[0]], y=[0, end[1]], z=[0, end[2]], mode="lines+text",
                line=dict(color=col, width=7), text=["", txt], textposition="top center",
                textfont=dict(size=18, color=col), showlegend=False, hoverinfo="skip"))
            bbox_pts.append(end[None, :])
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0], mode="markers+text", text=["O"], textposition="bottom left",
            marker=dict(size=5, color="#111"), textfont=dict(size=18, color="#111"), showlegend=False,
            hovertext=["<b>O</b> – origin (0, 0, 0)"], hoverinfo="text"))

    if show_labels:
        pts, txt = [], []
        for i, s in enumerate(species):
            if i in visible:
                pts.append(s["drawn"] @ M)
                txt += [s["label"].split(" ")[0]] * len(s["drawn"])
        if pts:
            P = np.vstack(pts)
            fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="text", text=txt,
                                       textfont=dict(size=12, color="black"), showlegend=False,
                                       hoverinfo="skip"))

    if show_vacant:
        for v in B["vacant"]:
            P = v["drawn"] @ M
            bbox_pts.append(P)
            fig.add_trace(go.Scatter3d(
                x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers", name=v["label"],
                marker=dict(symbol=v["symbol"], size=7, color="#e67e22", line=dict(width=3, color="#e67e22")),
                hovertext=[hover_text(v["label"], v["wy"], p, M, vacant=True) for p in v["drawn"]],
                hoverinfo="text"))

    if poly_choice != "None":
        i = labels.index(poly_choice)
        c_frac, cc, nb = cx.coordination_shell(B, i)
        color = species[i]["color"]
        if len(nb) >= 4:
            fig.add_trace(go.Mesh3d(x=nb[:, 0], y=nb[:, 1], z=nb[:, 2], alphahull=0, color=color,
                                    opacity=0.30, flatshading=True, hoverinfo="skip", showlegend=True,
                                    name=f"{poly_choice} polyhedron (CN = {len(nb)})"))
        elif len(nb) == 3:
            fig.add_trace(go.Mesh3d(x=nb[:, 0], y=nb[:, 1], z=nb[:, 2], i=[0], j=[1], k=[2], color=color,
                                    opacity=0.30, hoverinfo="skip", showlegend=True,
                                    name=f"{poly_choice} triangle (CN = 3)"))
        segs_p = [(cc, q) for q in nb]
        px, py, pz = lines_xyz(segs_p)
        fig.add_trace(go.Scatter3d(x=px, y=py, z=pz, mode="lines", line=dict(color="#222", width=3),
                                   hoverinfo="skip", showlegend=False))
        bbox_pts.append(nb)
        dists = np.linalg.norm(nb - cc, axis=1)
        st.info(f"**{poly_choice}** at ({c_frac[0]:.3f}, {c_frac[1]:.3f}, {c_frac[2]:.3f}) has "
                f"**{len(nb)}** nearest neighbours, {dists.min():.3f}–{dists.max():.3f} Å away "
                f"({cx.site_geometry(B, i)[0]}).")

    if (du, dv, dw) != (0, 0, 0):
        D = cx.direction_line(B, (du, dv, dw), visible)
        bbox_pts.append(np.array([D["start"], D["end"]]))
        name = cx.fmt_dir(D["vec"], cx.cell_kind(B["cell"]))
        fig.add_trace(go.Scatter3d(
            x=[D["start"][0], D["end"][0]], y=[D["start"][1], D["end"][1]], z=[D["start"][2], D["end"][2]],
            mode="lines", name=f"Direction {name}", line=dict(color="#ff2d55", width=9), hoverinfo="skip"))
        on = ", ".join(f"{n} × {lab}" for lab, n in D["per_species"].items()) or "no visible atoms"
        st.info(f"Direction **{name}**: repeat distance **{D['L']:.3f} Å**. Atoms on the line per repeat "
                f"distance: {on} (total {D['total']}) → linear density **{D['dens']:.2f} atoms/nm** "
                f"(only the species ticked in *Show species* are counted).")

    allp = np.vstack(bbox_pts)
    rmax = max(s["radius"] for s in species) * ball
    span = float((allp.max(0) - allp.min(0)).max())
    pad = rmax + 0.04 * span
    lo, hi = allp.min(0) - pad, allp.max(0) + pad


    def axis_cfg(k, name):
        return dict(title=name, range=[float(lo[k]), float(hi[k])], visible=show_grid, showbackground=False,
                    gridcolor="rgba(128,128,128,0.35)", zeroline=False)


    cam = CAMERAS[cam_name]
    camera = dict(eye=dict(x=cam["eye"][0], y=cam["eye"][1], z=cam["eye"][2]),
                  up=dict(x=cam["up"][0], y=cam["up"][1], z=cam["up"][2]))
    fig.update_layout(
        height=720, margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(xaxis=axis_cfg(0, "x (Å)"), yaxis=axis_cfg(1, "y (Å)"), zaxis=axis_cfg(2, "z (Å)"),
                   aspectmode="data", camera=camera),
        uirevision=f"{s_name}|{m_name}|{cam_name}",  # keep user rotation across reruns
        legend=dict(x=0.01, y=0.88, bgcolor="rgba(255,255,255,0.6)"),
        updatemenus=[dict(
            type="buttons", direction="left", x=0.01, y=0.99, xanchor="left", yanchor="top",
            buttons=[
                dict(label="▶ Auto-rotate", method="animate",
                     args=[None, dict(frame=dict(duration=60, redraw=True), transition=dict(duration=0),
                                      fromcurrent=True, mode="immediate")]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), transition=dict(duration=0),
                                        mode="immediate")]),
            ])],
    )
    fig.frames = rotation_frames(cam["eye"])

    plot_config = {"displaylogo": False,
                   "toImageButtonOptions": {"format": "png", "scale": 3, "filename": f"{formula}_unit_cell"}}
    try:
        st.plotly_chart(fig, width="stretch", config=plot_config)
    except Exception:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True, config=plot_config)

    if B["shape"] == "hex":
        st.caption("Hexagonal prism = 3 rhombic cells (3 × the Z and atom counts shown below). O sits at the centre "
                   "of the bottom face; x runs along the lattice vector a, y along b (120° from a), z along c. "
                   "Atoms on the prism surface are shared with neighbouring prisms (vertical edge = 1/3, side face = 1/2, "
                   "top/bottom = 1/2).")
    elif cx.cell_kind(B["cell"]) == "hexagonal":
        st.caption("Rhombic cell: the x and y axes run along the lattice vectors a and b (120° apart), "
                   "and z along c. O is the origin (0, 0, 0).")

    # --------------------------------------------------------------------------
    #  Tables
    # --------------------------------------------------------------------------
    tab_sites, tab_species, tab_about = st.tabs(["Site table", "Species and coordination", "About this structure"])

    rows = []
    for i, s in enumerate(species):
        if i not in visible:
            continue
        for p in s["drawn"]:
            xyz = p @ M
            rows.append({"Species": s["label"], "Wyckoff": s["wy"], "Site type": cx.POSITION_KIND[cx.n_contacts(p, B['shape'], M)],
                         "u": round(float(p[0]), 4), "v": round(float(p[1]), 4), "w": round(float(p[2]), 4),
                         "x (Å)": round(float(xyz[0]), 3), "y (Å)": round(float(xyz[1]), 3),
                         "z (Å)": round(float(xyz[2]), 3), "Share of cell": cx.share_in(p, B['shape'], M)})
    if show_vacant:
        for v in B["vacant"]:
            for p in v["drawn"]:
                xyz = p @ M
                rows.append({"Species": v["label"], "Wyckoff": v["wy"], "Site type": cx.POSITION_KIND[cx.n_contacts(p, B['shape'], M)],
                             "u": round(float(p[0]), 4), "v": round(float(p[1]), 4), "w": round(float(p[2]), 4),
                             "x (Å)": round(float(xyz[0]), 3), "y (Å)": round(float(xyz[1]), 3),
                             "z (Å)": round(float(xyz[2]), 3), "Share of cell": cx.share_in(p, B['shape'], M)})

    with tab_sites:
        show_df(pd.DataFrame(rows))
        st.caption("(u, v, w) are fractional coordinates. 'Share of cell' = 1/8 for corners, 1/4 for edges, "
                   "1/2 for faces, 1 for interior sites; boundary sites are drawn once for every cell face they touch.")

    with tab_species:
        srows = []
        for i, (s, c) in enumerate(zip(species, B["cn"])):
            srows.append({
                "Species": s["label"], "Element": s["el"], "Role": s["role"], "Wyckoff": s["wy"],
                "Sites per cell": s["mult"] * zf, "Sites drawn": len(s["drawn"]),
                "Radius (Å)": s["radius"], "CN": c["cn"],
                "Shortest bond (Å)": round(c["dmin"], 3), "Mean bond (Å)": round(c["dmean"], 3),
                "Site geometry": cx.site_geometry(B, i)[0],
                "Neighbours": ", ".join(f"{k} ×{v}" for k, v in c["partners"].items()),
            })
        show_df(pd.DataFrame(srows))
        counts = {}
        for s in species:
            counts[s["el"]] = counts.get(s["el"], 0) + s["mult"] * zf
        st.markdown("Atoms per cell: " + ", ".join(f"**{n} {el}**" for el, n in counts.items())
                    + f" → Z = {Z * zf} formula unit(s) of {formula}. "
                      f"Bonded = closer than {cx.BOND_FACTOR:g} × the species' shortest bond.")

    with tab_about:
        st.markdown(S["about"])
        ionic = mat.get("ionic", True)
        st.markdown(
            f"- **Space group:** {B['sg']} ({B['nops']} operations incl. centring)  \n"
            f"- **Pearson symbol:** {B['pearson']}  \n"
            f"- **Lattice:** {cx.lattice_string(B['cell'])}, volume {B['volume']:.1f} Å³  \n"
            f"- **Radii:** {'Shannon ionic radii (CN-appropriate)' if ionic else 'covalent radii'}; "
            f"spheres are scaled by the *ball size* slider  \n"
            f"- **Pauling radius-ratio rule:** 0.225–0.414 → CN 4 · 0.414–0.732 → CN 6 · 0.732–1.0 → CN 8")
        if m_name.startswith("SrFe"):
            st.caption("SrFe₁₂O₁₉ uses the lattice parameters of SrM with the internal coordinates of BaFe₁₂O₁₉ "
                       "(they differ only in the third decimal).")
        st.caption("Lattice parameters and internal coordinates are literature values at room temperature; "
                   "the viewer shows the ideal average structure (no thermal motion or disorder).")



with tab_pauling:
    pauling.render(B)
