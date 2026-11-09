"""Exam-readiness audit for every structure/material in the viewer.
 1. space-group number and Pearson symbol agree with spglib
 2. theoretical density inside the literature range
 3. shortest cation-O bond inside the usual window for that cation
 4. Pauling rule 2 (bond-valence sum at every O) within +/-0.5 (irregular polyhedra deviate) for all oxides
 5. host sublattice / hole labels (verify_structs.py)"""
import re, warnings, numpy as np, spglib
warnings.filterwarnings("ignore")
import crystal as cx, pauling

SYS = {range(1,3):"a", range(3,16):"m", range(16,75):"o", range(75,143):"t", range(143,195):"h", range(195,231):"c"}
DENS = {  # g/cm3 literature range
 "NaCl":(2.16,2.17),"MgO":(3.56,3.60),"CsCl":(3.97,4.00),"ZnS (sphalerite)":(4.0,4.1),"β-SiC":(3.19,3.23),
 "ZnS (wurtzite)":(4.0,4.15),"α-SiC (2H polytype)":(3.19,3.23),"CaF₂":(3.17,3.19),"Li₂O":(2.0,2.03),
 "TiO₂ (rutile)":(4.23,4.27),"α-Al₂O₃":(3.97,4.0),"MgAl₂O₄":(3.55,3.6),"Fe₃O₄ (magnetite)":(5.15,5.21),
 "BaFe₁₂O₁₉ (BaM)":(5.26,5.31),"SrFe₁₂O₁₉ (SrM)":(5.05,5.15),"BaTiO₃ – cubic (paraelectric)":(5.9,6.1),
 "BaTiO₃ – tetragonal (ferroelectric)":(5.95,6.1),"Forsterite Mg₂SiO₄ (peridot / zabarjad)":(3.21,3.23),
 "Zircon (ZrSiO₄)":(4.6,4.72),"Beryl (Be₃Al₂Si₆O₁₈)":(2.6,2.85),"α-Spodumene (LiAlSi₂O₆)":(3.15,3.2),
 "Tremolite (Ca₂Mg₅Si₈O₂₂(OH)₂)":(2.9,3.05),"Kaolinite (Al₂Si₂O₅(OH)₄)":(2.58,2.64),
 "Pyrophyllite-2M (Al₂Si₄O₁₀(OH)₂), approximate 1934 model":(2.65,2.9),"Talc (Mg₃Si₄O₁₀(OH)₂)":(2.7,2.85),
 "Muscovite 2M₁ (KAl₂(AlSi₃)O₁₀(OH)₂)":(2.76,3.0),"α-Quartz (SiO₂)":(2.64,2.66),"β-Quartz (SiO₂, > 573 °C)":(2.5,2.6),
 "α-Cristobalite (SiO₂, low T)":(2.3,2.4),"β-Cristobalite (SiO₂, high T, ideal)":(2.1,2.25),
 "Tridymite, hexagonal (SiO₂, ideal high-T)":(2.1,2.3),"Tridymite, orthorhombic (SiO₂, 220 °C)":(2.1,2.3),
}
WIN = {"Si":(1.53,1.66),"Al":(1.70,2.00),"Mg":(1.90,2.15),"Be":(1.58,1.70),"Zr":(2.05,2.30),"Ca":(2.30,2.55),"Li":(1.90,2.20),
       "K":(2.70,3.00),"Fe":(1.80,2.10),"Ti":(1.85,2.02),"Ba":(2.70,2.95),"Sr":(2.70,2.95)}
tot = dict(n=0, lab=0, den=0, bond=0, r2=0)
fails = []
for sname, S in cx.STRUCTURES.items():
    for mname in S["materials"]:
        B = cx.build(sname, mname); sp = B["species"]; M = B["M"]; tot["n"] += 1
        pos, num = [], []
        for i, s in enumerate(sp):
            for p in s["orbit"]: pos.append(p); num.append(i + 1)
        cell = (M, np.array(pos), np.array(num))
        ds = spglib.get_symmetry_dataset(cell, symprec=2e-3)
        std = spglib.standardize_cell(cell, to_primitive=False, no_idealize=True, symprec=2e-3)
        cnt = len(std[2]) // (3 if ds.international[0] == "R" else 1)
        strict = f"{next(l for r, l in SYS.items() if ds.number in r)}{ds.international[0]}{cnt}"
        ok1 = int(re.search(r"No\. (\d+)", B["sg"]).group(1)) == ds.number and B["pearson"].startswith(strict)
        rho = cx.density(B); lo, hi = DENS.get(mname, (None, None))
        ok2 = lo is None or (lo * 0.99 <= rho <= hi * 1.01)
        ok3 = True; badb = []
        oxide = any(s["el"] == "O" for s in sp if s["role"] == "anion")
        if oxide:
            for i, s in enumerate(sp):
                if s["role"] == "cation" and s["el"] in WIN and B["cn"][i]["cn"]:
                    d = B["cn"][i]["dmin"]
                    if not (WIN[s["el"]][0] <= d <= WIN[s["el"]][1]):
                        ok3 = False; badb.append(f"{s['label']} {d:.3f}")
        ok4 = True; r2bad = []
        if oxide:
            res = pauling.valence_tables(B)
            if res is not None and not res[1].empty:
                for _, r in res[1].iterrows():
                    if abs(float(r["Σ s received"]) - float(r["Anion charge"])) > 0.5:
                        ok4 = False; r2bad.append(r["Anion site"])
        for k, ok in zip(("lab", "den", "bond", "r2"), (ok1, ok2, ok3, ok4)): tot[k] += ok
        if not (ok1 and ok2 and ok3 and ok4):
            fails.append((mname, dict(labels=ok1, density=f"{rho:.3f} vs {lo}-{hi}" if not ok2 else "ok", bonds=badb or "ok", rule2=r2bad or "ok")))
print(f"{tot['n']} materials | labels OK {tot['lab']} | density OK {tot['den']} | bond lengths OK {tot['bond']} | rule-2 OK {tot['r2']}")
for m, d in fails: print(" ⚠", m, d)
