# -*- coding: utf-8 -*-
# Single file: Streamlit + Calculations + Pareto (always draw line) + Rich PDF (EN only)
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import os
import tempfile
import math
from datetime import datetime
from fpdf import FPDF  # pip install fpdf

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="Concrete LCA & Cost – PCC vs GPC", layout="wide")

# ================== TEXT (EN ONLY) ==================
L = {
    "title": "Concrete LCA & Cost – PCC vs GPC",
    "tabs": ["Inputs", "Results", "Pareto", "PDF"],
    "fmt_note": "Numbers are shown with comma decimal format.",
    "chart_note": "Charts are saved with transparent background.",
    "mix_section": "Mix & Strength",
    "region_section": "Region & Distances",
    "cost_section": "Cost Parameters",
    "ef_section": "Emission Factors",
    "curing_section": "Curing",
    "advanced": "Advanced",
    "edit_mixes": "Edit mix contents (kg/m³)",
    "override_distance": "Override city distances by user input",
    "override_ef": "Edit emission factors",
    "override_prices": "Edit unit prices",
    "strength": "Strength class (MPa)",
    "city": "City",
    "cities": ["Istanbul","Izmir","Ankara","Sinop","Adana","Diyarbakir","Erzurum"],
    "fa": "Fly Ash (FA)",
    "naoh": "Sodium Hydroxide (NaOH)",
    "na2sio3": "Sodium Silicate (Na₂SiO₃)",
    "agg_coarse": "Coarse Aggregate",
    "agg_fine": "Fine Aggregate",
    "sand": "Sand",
    "cement": "Portland Cement",
    "water": "Water",
    "sp": "Superplasticizer",
    "materials": "Material",
    "dist_km": "Distance (km)",
    "road": "Road (HGV)",
    "trans_cost": "Transport cost (₺/ton·km)",
    "ef_hgv": "HGV emission factor (kgCO₂e/ton·km)",
    "unit_prices": "Unit prices (₺/kg)",
    "price_ele": "Electricity price (₺/kWh)",
    "ef_grid": "Grid emission factor (kgCO₂e/kWh)",
    "curing_mode": "Curing mode",
    "normal": "Normal",
    "electric": "Electric",
    "elec_note": "For electric curing, temperature/duration can be adjusted.",
    "temp": "Temperature (°C)",
    "time_min": "Duration (minutes)",
    "results": "Results",
    "metric_cost": "Cost (₺/m³)",
    "metric_em": "Emissions (kgCO₂e/m³)",
    "breakdown": "Breakdown (Material / Transport / Curing)",
    "recommendation": "Recommendation",
    "rec_text": "Prefer lower emissions; if equal, choose lower cost.",
    "pcc": "PCC",
    "gpc": "GPC",
    "pareto_title": "Pareto Front – Cost vs Emissions",
    "pdf_title": "PDF Report",
    "pdf_make": "Generate PDF",
    "pdf_ready": "PDF is ready – download below.",
    "pdf_dl": "Download PDF",
    "pdf_font_upl": "Optional TTF font for PDF",
    "mix_used": "Mixes used (kg/m³)",
    "legend_selected": "Selected",
    "legend_front": "Pareto Front",
    "status": "Status",
    "status_default": "Default",
    "status_override": "Override",
    "created_at": "Generated at",
    "page": "Page",
    "electricity_consumption": "Electric curing consumption",
    "cards_title": "Summary",
    "em_compare": "Emissions Comparison (PCC vs GPC)",
    "cost_compare": "Cost Comparison (PCC vs GPC)",
    "material": "Material",
    "transport": "Transport",
    "curing": "Curing",
    "total": "Total",
    "delta": "Δ (GPC–PCC)",
    "delta_pct": "%Δ",
    "winner": "Winner",
    "lower_cost": "Lower cost",
    "lower_em": "Lower emissions",
    # PDF section titles
    "inputs": "Inputs",
    "distances": "Distances",
    "efs": "Emission Factors",
    "prices": "Unit Prices",
    "mixes": "Mix Compositions",
    "curing_details": "Curing Details",
    "em_breakdown": "Emissions Breakdown",
    "cost_breakdown": "Cost Breakdown"
}

# ================== HEADER ==================
st.title(L["title"])
st.caption(L["fmt_note"])

# ================== HELPERS ==================
def fmt(x: float, nd=2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "-"
    s = f"{x:,.{nd}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def transliterate_tr(s: str) -> str:
    rep = {"ş":"s","Ş":"S","ğ":"g","Ğ":"G","ı":"i","İ":"I","ç":"c","Ç":"C","ö":"o","Ö":"O","ü":"u","Ü":"U"}
    return "".join(rep.get(ch, ch) for ch in s)

def safe_pdf_str(s: str) -> str:
    if s is None: return ""
    s = transliterate_tr(str(s))
    for a,b in {"–":"-","—":"-","’":"'", "‘":"'", "“":'"', "”":'"', "…":"...", "•":"-", "→":"->", "°":" deg",
                "₺":" TL", "·":"*", "³":"3", "²":"2",
                "₀":"0","₁":"1","₂":"2","₃":"3","₄":"4","₅":"5","₆":"6","₇":"7","₈":"8","₉":"9"}.items():
        s = s.replace(a,b)
    return "".join(ch if ord(ch) < 256 else "?" for ch in s)

def slugify_filename(s: str) -> str:
    s = transliterate_tr(str(s))
    return "".join(c if c.isalnum() else "_" for c in s).strip("_")

# ================== DEFAULT DATA ==================
EF_DEFAULT = {
    "cement": 0.912, "fa": 0.027, "naoh": 1.955, "na2sio3": 1.382,
    "agg_coarse": 0.0408, "agg_fine": 0.0139, "sand": 0.00493,
    "water": 0.0, "sp": 5.2e-6
}
EF_HGV_DEFAULT = 0.118  # kgCO2e / ton-km

CITIES = L["cities"]
DIST_DEFAULT = {
    "FA":         [65.2, 8.4, 9.7, 158.0, 17.9, 59.2, 54.0],
    "NaOH":       [30.0, 248.0, 27.0, 422.0, 184.0, 518.0, 755.0],
    "Na2SiO3":    [217.0, 25.0, 190.0, 298.0, 20.0, 199.0, 493.0],
    "Aggregates": [63.2, 8.4, 9.7, 158.0, 17.9, 59.2, 54.0],
    "Cement":     [65.2, 8.4, 9.7, 158.0, 17.9, 59.2, 54.0],
    "Sand":       [39.3, 6.6, 12.0, 95.0, 75.2, 29.0, 11.8],
}

PCC_MIX = {
    55: {"cement":480, "agg":920, "sand":640, "water":185, "sp":0.0},
    45: {"cement":465, "agg":944, "sand":762, "water":173, "sp":0.0},
    35: {"cement":433, "agg":920, "sand":780, "water":171, "sp":0.0},
    30: {"cement":399, "agg":948, "sand":797, "water":170, "sp":0.0},
    25: {"cement":382, "agg":931, "sand":886, "water":170, "sp":0.0},
}
GPC_MIX = {
    55: {"fa":500, "naoh":80, "na2sio3":120, "agg_coarse":1321, "agg_fine":320, "sp":6.0},
    45: {"fa":400, "naoh":80, "na2sio3":120, "agg_coarse":1419, "agg_fine":344, "sp":4.0},
    35: {"fa":333, "naoh":80, "na2sio3":120, "agg_coarse":1485, "agg_fine":360, "sp":3.2},
    30: {"fa":286, "naoh":80, "na2sio3":120, "agg_coarse":1533, "agg_fine":371, "sp":0.0},
    25: {"fa":250, "naoh":80, "na2sio3":120, "agg_coarse":1567, "agg_fine":380, "sp":0.0},
}
GPC_CURE_EM_BASE = {55:37.08, 45:34.10, 35:25.65, 30:20.36, 25:17.09}

PRICE_DEFAULT = {
    "cement":2.00,"fa":0.30,"naoh":20.0,"na2sio3":8.0,"agg_coarse":0.20,"agg_fine":0.20,
    "sand":0.15,"water":0.005,"sp":25.0,"electricity":4.5,"transport_tkm":3.5
}

# ================== UI – INPUTS ==================
tabs = st.tabs(L["tabs"])

with tabs[0]:
    st.subheader(L["mix_section"])
    colm1, colm2 = st.columns([1,1])
    with colm1:
        strength = st.selectbox(L["strength"], [25,30,35,45,55], index=0)
    with colm2:
        city = st.selectbox(L["city"], CITIES, index=0)
        city_idx = CITIES.index(city)

    st.markdown("---")
    st.subheader(L["region_section"])
    st.write(L["road"])
    override_dist = st.checkbox(L["override_distance"], value=False)

    def dist_input_row(label_key, default_list):
        lab = L.get(label_key, label_key)
        dflt = default_list[city_idx]
        return st.number_input(f"{lab} – {L['dist_km']}", min_value=0.0, value=float(dflt), step=1.0, key=f"dist_{label_key}")

    FA_km = dist_input_row("fa", DIST_DEFAULT["FA"]) if override_dist else DIST_DEFAULT["FA"][city_idx]
    NaOH_km = dist_input_row("naoh", DIST_DEFAULT["NaOH"]) if override_dist else DIST_DEFAULT["NaOH"][city_idx]
    Na2SiO3_km = dist_input_row("na2sio3", DIST_DEFAULT["Na2SiO3"]) if override_dist else DIST_DEFAULT["Na2SiO3"][city_idx]
    Agg_km = dist_input_row("agg_coarse", DIST_DEFAULT["Aggregates"]) if override_dist else DIST_DEFAULT["Aggregates"][city_idx]
    Cement_km = dist_input_row("cement", DIST_DEFAULT["Cement"]) if override_dist else DIST_DEFAULT["Cement"][city_idx]
    Sand_km = dist_input_row("sand", DIST_DEFAULT["Sand"]) if override_dist else DIST_DEFAULT["Sand"][city_idx]

    st.markdown("---")
    st.subheader(L["ef_section"])
    override_ef = st.checkbox(L["override_ef"], value=False)
    ef = EF_DEFAULT.copy()
    if override_ef:
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            ef["cement"] = st.number_input(L["cement"]+" EF", min_value=0.0, value=float(ef["cement"]), key="ef_cement")
            ef["fa"] = st.number_input(L["fa"]+" EF", min_value=0.0, value=float(ef["fa"]), key="ef_fa")
        with c2:
            ef["naoh"] = st.number_input(L["naoh"]+" EF", min_value=0.0, value=float(ef["naoh"]), key="ef_naoh")
            ef["na2sio3"] = st.number_input(L["na2sio3"]+" EF", min_value=0.0, value=float(ef["na2sio3"]), key="ef_na2sio3")
        with c3:
            ef["agg_coarse"] = st.number_input(L["agg_coarse"]+" EF", min_value=0.0, value=float(ef["agg_coarse"]), key="ef_aggc")
            ef["agg_fine"] = st.number_input(L["agg_fine"]+" EF", min_value=0.0, value=float(ef["agg_fine"]), key="ef_aggf")
        with c4:
            ef["sand"] = st.number_input(L["sand"]+" EF", min_value=0.0, value=float(ef["sand"]), key="ef_sand")
            ef["sp"] = st.number_input(L["sp"]+" EF", min_value=0.0, value=float(ef["sp"]), key="ef_sp")
        EF_HGV = st.number_input(L["ef_hgv"], min_value=0.0, value=float(EF_HGV_DEFAULT), key="ef_hgv")
    else:
        EF_HGV = EF_HGV_DEFAULT

    st.markdown("---")
    st.subheader(L["cost_section"])
    override_prices = st.checkbox(L["override_prices"], value=False)
    price = PRICE_DEFAULT.copy()
    if override_prices:
        p1,p2,p3,p4 = st.columns(4)
        with p1:
            price["cement"] = st.number_input(L["cement"], min_value=0.0, value=float(price["cement"]), key="pr_cement")
            price["fa"] = st.number_input(L["fa"], min_value=0.0, value=float(price["fa"]), key="pr_fa")
        with p2:
            price["naoh"] = st.number_input(L["naoh"], min_value=0.0, value=float(price["naoh"]), key="pr_naoh")
            price["na2sio3"] = st.number_input(L["na2sio3"], min_value=0.0, value=float(price["na2sio3"]), key="pr_na2sio3")
        with p3:
            price["agg_coarse"] = st.number_input(L["agg_coarse"], min_value=0.0, value=float(price["agg_coarse"]), key="pr_aggc")
            price["agg_fine"] = st.number_input(L["agg_fine"], min_value=0.0, value=float(price["agg_fine"]), key="pr_aggf")
        with p4:
            price["sand"] = st.number_input(L["sand"], min_value=0.0, value=float(price["sand"]), key="pr_sand")
            price["water"] = st.number_input(L["water"], min_value=0.0, value=float(price["water"]), key="pr_water")
        pp = st.columns(3)
        with pp[0]:
            price["sp"] = st.number_input(L["sp"], min_value=0.0, value=float(price["sp"]), key="pr_sp")
        with pp[1]:
            price["electricity"] = st.number_input(L["price_ele"], min_value=0.0, value=float(price["electricity"]), key="pr_elec")
        with pp[2]:
            price["transport_tkm"] = st.number_input(L["trans_cost"], min_value=0.0, value=float(price["transport_tkm"]), key="pr_tkm")
    else:
        st.write(f"• {L['unit_prices']}: (use '{L['override_prices']}' to edit)")

    st.markdown("---")
    st.subheader(L["curing_section"])
    ccol1, ccol2, ccol3 = st.columns(3)
    with ccol1:
        curing_mode = st.selectbox(L["curing_mode"], [L["normal"], L["electric"]], index=0, key="cur_mode")

    if curing_mode == L["electric"]:
        with ccol2:
            ef_grid = st.number_input(L["ef_grid"], min_value=0.0, value=0.441)
        with ccol3:
            temp_c = st.slider(L["temp"], min_value=40, max_value=80, value=65, step=1, help=L["elec_note"])
        base_times = {55:1009, 45:928, 35:698, 30:554, 25:465}
        time_min = st.slider(L["time_min"], min_value=60, max_value=1440, value=base_times[strength], step=1, help=L["elec_note"])
    else:
        ef_grid = 0.441
        temp_c = 65
        time_min = {55:1009, 45:928, 35:698, 30:554, 25:465}[strength]

    st.markdown("---")
    st.subheader(L["advanced"])
    edit_mixes = st.checkbox(L["edit_mixes"], value=False)

    def mix_editor(title, dkeys, defaults, strength):
        st.markdown(f"**{title}**")
        cols = st.columns(len(dkeys))
        values = {}
        for i,k in enumerate(dkeys):
            with cols[i]:
                values[k] = st.number_input(
                    k, min_value=0.0, value=float(defaults.get(k, 0.0)),
                    step=1.0, key=f"mix_{title}_{k}_{strength}"
                )
        return values

    if edit_mixes:
        st.caption(L["mix_used"])
        pcc_keys = ["cement","agg","sand","water","sp"]
        pcc_vals = mix_editor("PCC", pcc_keys, PCC_MIX[strength], strength)
        gpc_keys = ["fa","naoh","na2sio3","agg_coarse","agg_fine","sp"]
        gpc_vals = mix_editor("GPC", gpc_keys, GPC_MIX[strength], strength)
    else:
        pcc_vals = PCC_MIX[strength].copy()
        gpc_vals = GPC_MIX[strength].copy()

# ================== CALC HELPERS ==================
def material_emissions_pcc(mix, ef) -> float:
    return (mix["cement"]*ef["cement"] + mix["agg"]*ef["agg_coarse"] + mix["sand"]*ef["sand"] +
            mix["water"]*ef["water"] + mix.get("sp",0.0)*ef["sp"])

def material_emissions_gpc(mix, ef) -> float:
    return (mix["fa"]*ef["fa"] + mix["naoh"]*ef["naoh"] + mix["na2sio3"]*ef["na2sio3"] +
            mix["agg_coarse"]*ef["agg_coarse"] + mix["agg_fine"]*ef["agg_fine"] + mix.get("sp",0.0)*ef["sp"])

def transport_emissions_pcc(mix, cement_km, agg_km, sand_km, ef_hgv) -> float:
    e = 0.0
    e += (mix["cement"]/1000.0) * (cement_km*2.0) * ef_hgv
    e += (mix["agg"]/1000.0)    * (agg_km*2.0)    * ef_hgv
    e += (mix["sand"]/1000.0)   * (sand_km*2.0)   * ef_hgv
    return e

def transport_emissions_gpc(mix, fa_km, naoh_km, na2sio3_km, agg_km, sand_km, ef_hgv) -> float:
    e = 0.0
    e += (mix["fa"]/1000.0)        * (fa_km*2.0)       * ef_hgv
    e += (mix["naoh"]/1000.0)      * (naoh_km*2.0)     * ef_hgv
    e += (mix["na2sio3"]/1000.0)   * (na2sio3_km*2.0)  * ef_hgv
    e += (mix["agg_coarse"]/1000.0)* (agg_km*2.0)      * ef_hgv
    e += (mix["agg_fine"]/1000.0)  * (agg_km*2.0)      * ef_hgv
    return e

def transport_cost_pcc(mix, cement_km, agg_km, sand_km, price_tkm) -> float:
    c = 0.0
    c += (mix["cement"]/1000.0) * (cement_km*2.0) * price_tkm
    c += (mix["agg"]/1000.0)    * (agg_km*2.0)    * price_tkm
    c += (mix["sand"]/1000.0)   * (sand_km*2.0)   * price_tkm
    return c

def transport_cost_gpc(mix, fa_km, naoh_km, na2sio3_km, agg_km, sand_km, price_tkm) -> float:
    c = 0.0
    c += (mix["fa"]/1000.0)        * (fa_km*2.0)       * price_tkm
    c += (mix["naoh"]/1000.0)      * (naoh_km*2.0)     * price_tkm
    c += (mix["na2sio3"]/1000.0)   * (na2sio3_km*2.0)  * price_tkm
    c += (mix["agg_coarse"]/1000.0)* (agg_km*2.0)      * price_tkm
    c += (mix["agg_fine"]/1000.0)  * (agg_km*2.0)      * price_tkm
    return c

def curing_emissions_gpc(mode, strength, temp_c, time_min, ef_grid):
    if mode != L["electric"]:
        return 0.0, 0.0
    base_em = GPC_CURE_EM_BASE[strength]
    base_t = {55:1009, 45:928, 35:698, 30:554, 25:465}[strength]
    temp_scale = max(0.5, (temp_c - 25.0) / (65.0 - 25.0))
    time_scale = max(0.5, time_min / base_t)
    em = base_em * temp_scale * time_scale
    kwh = em / ef_grid if ef_grid > 0 else 0.0
    return em, kwh

def material_cost_pcc(mix, price) -> float:
    return (mix["cement"]*price["cement"] + mix["agg"]*price["agg_coarse"] +
            mix["sand"]*price["sand"] + mix["water"]*price["water"] + mix.get("sp",0.0)*price["sp"])

def material_cost_gpc(mix, price) -> float:
    return (mix["fa"]*price["fa"] + mix["naoh"]*price["naoh"] + mix["na2sio3"]*price["na2sio3"] +
            mix["agg_coarse"]*price["agg_coarse"] + mix["agg_fine"]*price["agg_fine"] + mix.get("sp",0.0)*price["sp"])

def non_dominated(points: np.ndarray) -> np.ndarray:
    N = points.shape[0]
    keep = np.ones(N, dtype=bool)
    for i in range(N):
        if not keep[i]:
            continue
        for j in range(N):
            if i == j:
                continue
            if (points[j,0] <= points[i,0] and points[j,1] <= points[i,1]) and \
               (points[j,0] < points[i,0] or points[j,1] < points[i,1]):
                keep[i] = False
                break
    return keep

# ================== CALCULATE ==================
# PCC
pcc_em_mat = material_emissions_pcc(pcc_vals, ef)
pcc_em_tr  = transport_emissions_pcc(pcc_vals, Cement_km, Agg_km, Sand_km, EF_HGV)
pcc_em_cure = 0.0
pcc_em_tot = pcc_em_mat + pcc_em_tr + pcc_em_cure

pcc_cost_mat = material_cost_pcc(pcc_vals, price)
pcc_cost_tr  = transport_cost_pcc(pcc_vals, Cement_km, Agg_km, Sand_km, price["transport_tkm"])
pcc_cost_cure = 0.0
pcc_cost_tot = pcc_cost_mat + pcc_cost_tr + pcc_cost_cure

# GPC
gpc_em_mat = material_emissions_gpc(gpc_vals, ef)
gpc_em_tr  = transport_emissions_gpc(gpc_vals, FA_km, NaOH_km, Na2SiO3_km, Agg_km, Sand_km, EF_HGV)
gpc_em_cure, gpc_kwh = curing_emissions_gpc(curing_mode, strength, temp_c, time_min, ef_grid)
gpc_em_tot = gpc_em_mat + gpc_em_tr + gpc_em_cure

gpc_cost_mat = material_cost_gpc(gpc_vals, price)
gpc_cost_tr  = transport_cost_gpc(gpc_vals, FA_km, NaOH_km, Na2SiO3_km, Agg_km, Sand_km, price["transport_tkm"])
gpc_cost_cure = gpc_kwh * price["electricity"]
gpc_cost_tot = gpc_cost_mat + gpc_cost_tr + gpc_cost_cure

# ================== RESULTS (UI) ==================
def card_html(title, cost, em, low_cost=False, low_em=False):
    badge_cost = f"<span style='background:#0f5132;color:#d1e7dd;padding:2px 6px;border-radius:8px;margin-left:8px;font-size:12px'>{L['lower_cost']}</span>" if low_cost else ""
    badge_em = f"<span style='background:#084298;color:#cfe2ff;padding:2px 6px;border-radius:8px;margin-left:6px;font-size:12px'>{L['lower_em']}</span>" if low_em else ""
    return f"""
    <div style="border:1px solid rgba(255,255,255,0.1);border-radius:14px;padding:16px 18px;margin-bottom:8px;">
      <div style="font-weight:700;font-size:20px;margin-bottom:8px">{title} {badge_cost} {badge_em}</div>
      <div style="display:flex;gap:40px;align-items:baseline;">
        <div><div style="opacity:.8;font-size:12px">{L['metric_cost']}</div>
             <div style="font-size:28px;font-weight:700">{fmt(cost)}</div></div>
        <div><div style="opacity:.8;font-size:12px">{L['metric_em']}</div>
             <div style="font-size:28px;font-weight:700">{fmt(em)}</div></div>
      </div>
    </div>
    """

with tabs[1]:
    st.markdown("### " + L["results"])
    c1, c2 = st.columns(2)
    low_cost_is_gpc = gpc_cost_tot < pcc_cost_tot
    low_em_is_gpc = gpc_em_tot < pcc_em_tot

    with c1:
        st.markdown(
            card_html(L["pcc"], pcc_cost_tot, pcc_em_tot, low_cost=not low_cost_is_gpc, low_em=not low_em_is_gpc),
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            card_html(L["gpc"], gpc_cost_tot, gpc_em_tot, low_cost=low_cost_is_gpc, low_em=low_em_is_gpc),
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Emissions – stacked
    st.markdown(f"### {L['em_compare']}")
    fig_em, ax_em = plt.subplots()
    labels = [L["pcc"], L["gpc"]]
    mat = [pcc_em_mat, gpc_em_mat]
    trp = [pcc_em_tr,  gpc_em_tr]
    cur = [pcc_em_cure, gpc_em_cure]
    x = np.arange(len(labels)); w = 0.6
    ax_em.bar(x, mat, width=w, label=f"{L['material']} Em.")
    ax_em.bar(x, trp, width=w, bottom=mat, label=f"{L['transport']} Em.")
    ax_em.bar(x, cur, width=w, bottom=np.array(mat)+np.array(trp), label=f"{L['curing']} Em.")
    ax_em.set_xticks(x); ax_em.set_xticklabels(labels)
    ax_em.set_ylabel(L["metric_em"])
    totals_em = [pcc_em_tot, gpc_em_tot]
    for xi, val in zip(x, totals_em):
        ax_em.text(xi, val, fmt(val), ha='center', va='bottom', fontsize=10)
    ax_em.legend()
    st.pyplot(fig_em, clear_figure=True)

    # Emissions table
    row_names = [L["material"], L["transport"], L["curing"], L["total"]]
    em_df = pd.DataFrame({
        "": row_names,
        "PCC": [pcc_em_mat, pcc_em_tr, pcc_em_cure, pcc_em_tot],
        "GPC": [gpc_em_mat, gpc_em_tr, gpc_em_cure, gpc_em_tot],
    })
    em_df[L["delta"]] = em_df["GPC"] - em_df["PCC"]
    em_df[L["delta_pct"]] = np.where(em_df["PCC"]!=0, em_df[L["delta"]]/em_df["PCC"]*100, np.nan)
    em_df[L["winner"]] = np.where(em_df[L["delta"]]<0, L["gpc"], np.where(em_df[L["delta"]]>0, L["pcc"], "—"))

    def style_delta(v):
        if pd.isna(v): return ""
        return "background-color:#e6ffed;color:#0a3622" if v < 0 else ("background-color:#ffebeb;color:#842029" if v > 0 else "background-color:#eee;color:#333")

    st.write(
        em_df.style.format(
            { "PCC": lambda v: fmt(v), "GPC": lambda v: fmt(v),
              L["delta"]: lambda v: fmt(v), L["delta_pct"]: lambda v: f"{fmt(v,2)}%" }
        ).applymap(style_delta, subset=[L["delta"], L["delta_pct"]])
    )

    st.markdown("---")

    # Cost – stacked
    st.markdown(f"### {L['cost_compare']}")
    fig_c, ax_c = plt.subplots()
    matc = [pcc_cost_mat, gpc_cost_mat]
    trpc = [pcc_cost_tr,  gpc_cost_tr]
    curc = [pcc_cost_cure, gpc_cost_cure]
    ax_c.bar(x, matc, width=w, label="Material Cost")
    ax_c.bar(x, trpc, width=w, bottom=matc, label="Transport Cost")
    ax_c.bar(x, curc, width=w, bottom=np.array(matc)+np.array(trpc), label="Curing Cost")
    ax_c.set_xticks(x); ax_c.set_xticklabels(labels)
    ax_c.set_ylabel(L["metric_cost"])
    totals_cost = [pcc_cost_tot, gpc_cost_tot]
    for xi, val in zip(x, totals_cost):
        ax_c.text(xi, val, fmt(val), ha='center', va='bottom', fontsize=10)
    ax_c.legend()
    st.pyplot(fig_c, clear_figure=True)

    # Cost table
    cost_df = pd.DataFrame({
        "": row_names,
        "PCC": [pcc_cost_mat, pcc_cost_tr, pcc_cost_cure, pcc_cost_tot],
        "GPC": [gpc_cost_mat, gpc_cost_tr, gpc_cost_cure, gpc_cost_tot],
    })
    cost_df[L["delta"]] = cost_df["GPC"] - cost_df["PCC"]
    cost_df[L["delta_pct"]] = np.where(cost_df["PCC"]!=0, cost_df[L["delta"]]/cost_df["PCC"]*100, np.nan)
    cost_df[L["winner"]] = np.where(cost_df[L["delta"]]<0, L["gpc"], np.where(cost_df[L["delta"]]>0, L["pcc"], "—"))

    st.write(
        cost_df.style.format(
            { "PCC": lambda v: fmt(v), "GPC": lambda v: fmt(v),
              L["delta"]: lambda v: fmt(v), L["delta_pct"]: lambda v: f"{fmt(v,2)}%" }
        ).applymap(style_delta, subset=[L["delta"], L["delta_pct"]])
    )

    st.markdown("---")
    st.subheader(L["recommendation"])
    st.caption(L["rec_text"])
    if abs(gpc_em_tot - pcc_em_tot) > 1e-6:
        rec = L["gpc"] if gpc_em_tot < pcc_em_tot else L["pcc"]
    else:
        rec = L["gpc"] if gpc_cost_tot < pcc_cost_tot else L["pcc"]
    st.success(f"→ {L['recommendation']}: **{rec}**")

# ================== PARETO (always draw a line) ==================
with tabs[2]:
    st.subheader(L["pareto_title"])

    pts_arr = np.array([
        [pcc_cost_tot, pcc_em_tot],  # PCC
        [gpc_cost_tot, gpc_em_tot],  # GPC
    ])
    keep = non_dominated(pts_arr)
    front = pts_arr[keep]
    if len(front) > 1:
        front = front[np.argsort(front[:,0])]

    fig, ax = plt.subplots()
    ax.scatter([pcc_cost_tot, gpc_cost_tot], [pcc_em_tot, gpc_em_tot], s=70, label=L["legend_selected"])
    ax.annotate(L["pcc"], (pcc_cost_tot, pcc_em_tot), textcoords="offset points", xytext=(8, -10), ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    ax.annotate(L["gpc"], (gpc_cost_tot, gpc_em_tot), textcoords="offset points", xytext=(8, -10), ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

    if len(front) > 1:
        ax.plot(front[:,0], front[:,1], linewidth=2, label=L["legend_front"])
    else:
        ax.plot([pcc_cost_tot, gpc_cost_tot], [pcc_em_tot, gpc_em_tot], linewidth=2, label=L["legend_front"])

    ax.set_xlabel(L["metric_cost"]); ax.set_ylabel(L["metric_em"])
    ax.legend()
    st.pyplot(fig, clear_figure=True)
    st.caption(L["chart_note"])

# ================== PDF ==================
def fig_to_png_bytes(fig) -> bytes:
    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=160, bbox_inches="tight", transparent=True)
    bio.seek(0)
    return bio.read()

def pdf_image_bytes(pdf, img_bytes, x=10, y=None, w=190, h=0, fmt="PNG"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{fmt.lower()}") as tmp:
        tmp.write(img_bytes); tmp.flush(); path = tmp.name
    try:
        pdf.image(path, x=x, y=y, w=w, h=h, type=fmt)
    finally:
        try: os.remove(path)
        except Exception: pass

class PDF(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font(self.font_family, size=9)
        self.cell(0, 10, safe_pdf_str(f"{L['page']} {self.page_no()} / {{nb}}"), 0, 0, "C")

def draw_table(pdf, headers, rows, col_w, x=10, row_h=7):
    def draw_header():
        pdf.set_x(x)
        pdf.set_fill_color(230, 233, 237)
        pdf.set_font(pdf.font_family, size=11)
        for i,h in enumerate(headers):
            pdf.cell(col_w[i], row_h, txt=safe_pdf_str(h), border=1, align='C', fill=True)
        pdf.ln(row_h)
        pdf.set_font(pdf.font_family, size=10)
    draw_header()
    for r in rows:
        if pdf.get_y() > 270:
            pdf.add_page(); draw_header()
        pdf.set_x(x)
        for i,c in enumerate(r):
            txt = c if isinstance(c,str) else f"{c}"
            pdf.cell(col_w[i], row_h, txt=safe_pdf_str(txt), border=1, align='C')
        pdf.ln(row_h)

def section_title(pdf, txt):
    pdf.set_font(pdf.font_family, size=12)
    pdf.cell(0, 8, txt=safe_pdf_str(txt), ln=1)
    pdf.ln(1)

def make_pdf(ttf_file: bytes=None) -> bytes:
    arr = np.array([
        [pcc_cost_tot, pcc_em_tot],
        [gpc_cost_tot, gpc_em_tot],
    ])
    keep = non_dominated(arr)
    fr = arr[keep]
    if len(fr) > 1:
        fr = fr[np.argsort(fr[:,0])]

    fig_p = plt.figure()
    ax = fig_p.add_subplot(111)
    ax.scatter([pcc_cost_tot, gpc_cost_tot],[pcc_em_tot, gpc_em_tot], s=70, label=L["legend_selected"])
    ax.annotate(L["pcc"], (pcc_cost_tot, pcc_em_tot), textcoords="offset points", xytext=(8, -10), ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    ax.annotate(L["gpc"], (gpc_cost_tot, gpc_em_tot), textcoords="offset points", xytext=(8, -10), ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    if len(fr) > 1:
        ax.plot(fr[:,0], fr[:,1], linewidth=2, label=L["legend_front"])
    else:
        ax.plot([pcc_cost_tot, gpc_cost_tot], [pcc_em_tot, gpc_em_tot], linewidth=2, label=L["legend_front"])
    ax.set_xlabel(L["metric_cost"]); ax.set_ylabel(L["metric_em"]); ax.legend()
    pareto_png = fig_to_png_bytes(fig_p); plt.close(fig_p)

    fig_b, axb = plt.subplots()
    labels_b = ["Material","Transport","Curing"]
    pcc_vals_em = [pcc_em_mat, pcc_em_tr, pcc_em_cure]
    gpc_vals_em = [gpc_em_mat, gpc_em_tr, gpc_em_cure]
    xbar = np.arange(len(labels_b)); wbar = 0.35
    axb.bar(xbar - wbar/2, pcc_vals_em, width=wbar, label="PCC")
    axb.bar(xbar + wbar/2, gpc_vals_em, width=wbar, label="GPC")
    axb.set_xticks(xbar); axb.set_xticklabels(labels_b)
    axb.set_ylabel(L["metric_em"]); axb.legend()
    breakdown_png = fig_to_png_bytes(fig_b); plt.close(fig_b)

    pdf = PDF(orientation='P', unit='mm', format='A4')
    pdf.alias_nb_pages()
    pdf.add_page()

    use_unicode = False
    if ttf_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ttf") as tf:
            tf.write(ttf_file); font_path = tf.name
        try:
            pdf.add_font("Custom", "", font_path, uni=True)
            pdf.set_font("Custom", size=14); use_unicode = True
        except Exception:
            pdf.set_font("Helvetica", size=14)
    else:
        pdf.set_font("Helvetica", size=14)

    pdf.cell(0, 10, txt=(L["title"] if use_unicode else safe_pdf_str(L["title"])), ln=1, align="C")
    pdf.set_font(pdf.font_family, size=9)
    pdf.cell(0, 6, txt=safe_pdf_str(f"{L['created_at']}: {datetime.now():%Y-%m-%d %H:%M}"), ln=1)
    pdf.set_font(pdf.font_family, size=11)

    section_title(pdf, f"1) {L['inputs']}")
    pdf.cell(0, 7, txt=safe_pdf_str(f"{L['city']}: {city} | {L['strength']}: {strength} MPa | {L['curing_mode']}: {curing_mode}"), ln=1)

    dist_status = L["status_override"] if override_dist else L["status_default"]
    section_title(pdf, f"1.1) {L['distances']} ({L['status']}: {dist_status})")
    dist_headers = [L["materials"], f"{L['dist_km']}", L["status"]]
    dist_rows = [
        [L["fa"], fmt(FA_km,1), dist_status],
        [L["naoh"], fmt(NaOH_km,1), dist_status],
        [L["na2sio3"], fmt(Na2SiO3_km,1), dist_status],
        [L["agg_coarse"], fmt(Agg_km,1), dist_status],
        [L["cement"], fmt(Cement_km,1), dist_status],
        [L["sand"], fmt(Sand_km,1), dist_status],
    ]
    draw_table(pdf, dist_headers, dist_rows, [70, 50, 50], x=10)

    ef_status = L["status_override"] if override_ef else L["status_default"]
    section_title(pdf, f"1.2) {L['efs']} ({L['status']}: {ef_status})")
    ef_headers = [L["materials"], "EF (kgCO2e/kg)", L["status"]]
    ef_rows = [
        [L["cement"], fmt(ef['cement']), ef_status],
        [L["fa"], fmt(ef['fa']), ef_status],
        [L["naoh"], fmt(ef['naoh']), ef_status],
        [L["na2sio3"], fmt(ef['na2sio3']), ef_status],
        [L["agg_coarse"], fmt(ef['agg_coarse']), ef_status],
        [L["agg_fine"], fmt(ef['agg_fine']), ef_status],
        [L["sand"], fmt(ef['sand']), ef_status],
        [L["sp"], fmt(ef['sp']), ef_status],
        ["HGV", f"{fmt(EF_HGV)} kgCO2e/ton-km", ef_status],
    ]
    draw_table(pdf, ef_headers, ef_rows, [70, 70, 30], x=10)

    price_status = L["status_override"] if override_prices else L["status_default"]
    section_title(pdf, f"1.3) {L['prices']} ({L['status']}: {price_status})")
    pr_headers = [L["materials"], "Price", L["status"]]
    pr_rows = [
        [L["cement"], f"{fmt(price['cement'])} ₺/kg", price_status],
        [L["fa"], f"{fmt(price['fa'])} ₺/kg", price_status],
        [L["naoh"], f"{fmt(price['naoh'])} ₺/kg", price_status],
        [L["na2sio3"], f"{fmt(price['na2sio3'])} ₺/kg", price_status],
        [L["agg_coarse"], f"{fmt(price['agg_coarse'])} ₺/kg", price_status],
        [L["agg_fine"], f"{fmt(price['agg_fine'])} ₺/kg", price_status],
        [L["sand"], f"{fmt(price['sand'])} ₺/kg", price_status],
        [L["water"], f"{fmt(price['water'])} ₺/kg", price_status],
        [L["sp"], f"{fmt(price['sp'])} ₺/kg", price_status],
        ["HGV", f"{fmt(price['transport_tkm'])} ₺/ton-km", price_status],
        [L["price_ele"], f"{fmt(price['electricity'])} ₺/kWh", price_status],
    ]
    draw_table(pdf, pr_headers, pr_rows, [70, 70, 30], x=10)

    section_title(pdf, f"1.4) {L['curing_details']}")
    if curing_mode == L["electric"]:
        cur_rows = [
            [L["curing_mode"], L["electric"]],
            [L["temp"], f"{temp_c} °C"],
            [L["time_min"], f"{time_min} min"],
            [L["ef_grid"], f"{fmt(ef_grid)} kgCO2e/kWh"],
            [L["electricity_consumption"], f"{fmt(gpc_kwh)} kWh/m³"],
        ]
    else:
        cur_rows = [
            [L["curing_mode"], L["normal"]],
            [L["electricity_consumption"], "—"],
        ]
    draw_table(pdf, ["Parameter","Value"], cur_rows, [80, 100], x=10)

    section_title(pdf, f"1.5) {L['mixes']} ({L['status']}: {L['status_override'] if edit_mixes else L['status_default']})")
    pcc_rows = [[k, fmt(v)] for k,v in pcc_vals.items()]
    gpc_rows = [[k, fmt(v)] for k,v in gpc_vals.items()]
    pdf.cell(0, 6, txt=safe_pdf_str(f"{L['pcc']} (kg/m³)"), ln=1)
    draw_table(pdf, ["Component","Value"], pcc_rows, [90, 90], x=10)
    pdf.cell(0, 6, txt=safe_pdf_str(f"{L['gpc']} (kg/m³)"), ln=1)
    draw_table(pdf, ["Component","Value"], gpc_rows, [90, 90], x=10)

    pdf.add_page()
    section_title(pdf, f"2) {L['results']}")
    totals_headers = ["", "PCC", "GPC"]
    totals_rows = [
        [L["metric_cost"], fmt(pcc_cost_tot), fmt(gpc_cost_tot)],
        [L["metric_em"],   fmt(pcc_em_tot),  fmt(gpc_em_tot)],
    ]
    draw_table(pdf, totals_headers, totals_rows, [60, 60, 60], x=10)

    section_title(pdf, f"2.2) {L['em_breakdown']}")
    em_rows = [
        [L["material"], fmt(pcc_em_mat), fmt(gpc_em_mat)],
        [L["transport"], fmt(pcc_em_tr), fmt(gpc_em_tr)],
        [L["curing"], fmt(pcc_em_cure), fmt(gpc_em_cure)],
        [L["total"], fmt(pcc_em_tot), fmt(gpc_em_tot)],
    ]
    draw_table(pdf, ["", "PCC", "GPC"], em_rows, [60, 60, 60], x=10)

    section_title(pdf, f"2.3) {L['cost_breakdown']}")
    cost_rows = [
        [L["material"], fmt(pcc_cost_mat), fmt(gpc_cost_mat)],
        [L["transport"], fmt(pcc_cost_tr), fmt(gpc_cost_tr)],
        [L["curing"], fmt(pcc_cost_cure), fmt(gpc_cost_cure)],
        [L["total"], fmt(pcc_cost_tot), fmt(gpc_cost_tot)],
    ]
    draw_table(pdf, ["", "PCC", "GPC"], cost_rows, [60, 60, 60], x=10)

    pdf.add_page()
    section_title(pdf, f"3) {L['pareto_title']}")
    pdf_image_bytes(pdf, pareto_png, x=10, y=None, w=190)

    out = pdf.output(dest="S")
    return out if isinstance(out, bytes) else out.encode("latin-1", "replace")

with tabs[3]:
    st.subheader(L["pdf_title"])
    font_file = st.file_uploader(L["pdf_font_upl"], type=["ttf"])
    if st.button(L["pdf_make"]):
        ttf_bytes = font_file.read() if font_file is not None else None
        pdf_bytes = make_pdf(ttf_bytes)
        st.success(L["pdf_ready"])
        date_str = datetime.now().strftime("%Y%m%d")
        fname = f"report_{slugify_filename(city)}_{strength}MPa_{date_str}.pdf"
        st.download_button(L["pdf_dl"], data=pdf_bytes, file_name=fname, mime="application/pdf")

# ================== FOOTER ==================
st.caption("© ME499 Track-2 Tool v1.0")
