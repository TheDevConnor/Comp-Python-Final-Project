import os
import json
import sys

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

sys.path.insert(0, os.path.dirname(__file__))

from tools.compute_dosage import compute_dosage
from tools.find_optimal_location import find_optimal_location
from data.hospitals import HOSPITALS, PHARMACIES, ALL_FACILITIES, DRUG_FORMULARY
from rag.knowledge_base import build_or_load_vectorstore
from components.location_search_components import render_location_search, get_custom_facilities
from components.location_search_components import render_location_search, get_custom_facilities


st.set_page_config(
    page_title="SNAP - Medical Distribution Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
  }

  .stApp {
    background: #f4f6f9;
    color: #1a2035;
  }

  /* ── Sidebar ── */
  div[data-testid="stSidebarContent"] {
    background: #1a2035;
    color: #e8ecf4;
    padding-top: 0.5rem;
  }
  div[data-testid="stSidebarContent"] label,
  div[data-testid="stSidebarContent"] p,
  div[data-testid="stSidebarContent"] span {
    color: #b0bcd4 !important;
    font-size: 0.82rem !important;
  }
  div[data-testid="stSidebarContent"] .stMultiSelect span,
  div[data-testid="stSidebarContent"] .stCheckbox span {
    color: #e8ecf4 !important;
  }

  .snap-logo {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #4ade9e;
    letter-spacing: 0.05em;
    padding: 1rem 0 0 0;
    display: block;
  }
  .snap-tagline {
    font-size: 0.73rem;
    color: #5a6a8a;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid #2a3350;
    margin-bottom: 1.2rem;
  }
  .sidebar-section {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #4a5a7a;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 1rem 0 0.5rem 0;
  }

  /* ── Main content ── */
  .page-title {
    font-family: 'Sora', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #1a2035;
    margin-bottom: 0.1rem;
  }
  .page-sub {
    font-size: 0.82rem;
    color: #6b7a9a;
    margin-bottom: 1.5rem;
  }

  /* ── Stat cards ── */
  .stat-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    border: 1px solid #e4e9f2;
  }
  .stat-label {
    font-size: 0.68rem;
    font-weight: 600;
    color: #8a97b0;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.35rem;
  }
  .stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.45rem;
    font-weight: 700;
    color: #1a2035;
    line-height: 1.1;
  }
  .stat-value.accent { color: #0ea875; }
  .stat-value.warn   { color: #f59e0b; }

  /* ── Table section header ── */
  .section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #8a97b0;
    margin: 1.4rem 0 0.6rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #e4e9f2;
  }

  /* ── Drug formulary card ── */
  .drug-card {
    background: #ffffff;
    border: 1px solid #e4e9f2;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.55rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
  }
  .drug-name {
    font-weight: 600;
    font-size: 0.88rem;
    color: #1a2035;
  }
  .drug-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #6b7a9a;
    margin-top: 0.1rem;
  }
  .drug-badge {
    font-size: 0.62rem;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 20px;
    white-space: nowrap;
  }
  .badge-cold { background: #dbeafe; color: #1d4ed8; }
  .badge-warm { background: #f0fdf4; color: #15803d; }

  /* ── Dosage result ── */
  .dose-result-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 1.5rem;
    border: 1px solid #e4e9f2;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
  }
  .dose-big {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    color: #0ea875;
    line-height: 1;
  }
  .dose-big.capped { color: #f59e0b; }
  .dose-detail {
    font-size: 0.82rem;
    color: #6b7a9a;
    margin-top: 0.6rem;
    line-height: 1.55;
    padding: 0.7rem;
    background: #f8fafc;
    border-radius: 8px;
    border-left: 3px solid #0ea875;
  }
  .dose-detail.capped { border-left-color: #f59e0b; }

  /* ── Agent chat ── */
  .chat-user {
    background: #1a2035;
    color: #e8ecf4;
    border-radius: 14px 14px 4px 14px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0 0.5rem 3rem;
    font-size: 0.88rem;
    line-height: 1.55;
  }
  .chat-agent {
    background: #ffffff;
    border: 1px solid #e4e9f2;
    border-radius: 4px 14px 14px 14px;
    padding: 0.85rem 1rem;
    margin: 0.5rem 3rem 0.5rem 0;
    font-size: 0.88rem;
    line-height: 1.6;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
  }
  .tool-call-box {
    background: #f8fafc;
    border: 1px solid #e4e9f2;
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #4a5a7a;
    margin-top: 0.3rem;
  }

  /* ── Pill badge for tool names ── */
  .tool-pill {
    display: inline-block;
    background: #eef2ff;
    color: #4f46e5;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    padding: 2px 8px;
    border-radius: 20px;
    margin-right: 4px;
  }

  /* ── Streamlit overrides ── */
  .stButton > button {
    background: #0ea875 !important;
    color: #ffffff !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.45rem 1.4rem !important;
    transition: background 0.15s !important;
  }
  .stButton > button:hover {
    background: #09c487 !important;
  }

  /* ── Main area inputs: white bg, dark text ── */
  .stTextInput > div > div > input,
  .stNumberInput > div > div > input {
    background: #ffffff !important;
    color: #1a2035 !important;
    border: 1px solid #dde3ee !important;
    border-radius: 8px !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.88rem !important;
  }
  .stTextInput > div > div > input::placeholder,
  .stNumberInput > div > div > input::placeholder {
    color: #9aa5be !important;
  }

  /* ── Selectbox: white bg, dark text ── */
  .stSelectbox > div > div,
  .stSelectbox [data-baseweb="select"] > div {
    background: #ffffff !important;
    color: #1a2035 !important;
    border: 1px solid #dde3ee !important;
    border-radius: 8px !important;
  }
  
  .stSelectbox [data-baseweb="select"] span,
  .stSelectbox [data-baseweb="select"] div {
    color: #1a2035 !important;
  }
  
  /* Dropdown portal list container */
  [data-baseweb="popover"],
  [data-baseweb="popover"] > div,
  [data-baseweb="menu"],
  ul[data-baseweb="menu"] {
    background: #ffffff !important;
    color: #1a2035 !important;
  }
  
  /* Each option item */
  [role="option"],
  li[role="option"] {
    background: #ffffff !important;
    color: #1a2035 !important;
  }
  
  /* Hover state */
  [role="option"]:hover,
  li[role="option"]:hover {
    background: #f0fdf8 !important;
    color: #1a2035 !important;
  }
  
  /* Active / currently-highlighted option (keyboard nav) */
  [aria-selected="true"][role="option"],
  li[aria-selected="true"] {
    background: #e6faf3 !important;
    color: #0ea875 !important;
    font-weight: 600 !important;
  }
  
  /* The text node inside the option */
  [role="option"] span,
  [role="option"] div,
  li[role="option"] span {
    color: inherit !important;
  }
  /* Number input +/- buttons */
  .stNumberInput [data-testid="stNumberInputField"] {
    background: #ffffff !important;
    color: #1a2035 !important;
  }
  .stNumberInput button {
    background: #f4f6f9 !important;
    color: #1a2035 !important;
    border-color: #dde3ee !important;
  }

  /* Slider in main area */
  .stSlider [data-testid="stSlider"] label,
  .stSlider label {
    color: #1a2035 !important;
  }

  /* Main area labels */
  .stSelectbox label,
  .stNumberInput label,
  .stTextInput label,
  .stSlider label {
    color: #1a2035 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
  }

  .stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: transparent;
    border-bottom: 2px solid #e4e9f2;
  }
  .stTabs [data-baseweb="tab"] {
    font-family: 'Sora', sans-serif;
    font-size: 0.85rem;
    font-weight: 500;
    color: #6b7a9a;
    background: transparent;
    border: none;
    padding: 0.5rem 1rem;
  }
  .stTabs [aria-selected="true"] {
    color: #1a2035 !important;
    font-weight: 600 !important;
    border-bottom: 2px solid #0ea875 !important;
  }

  /* ── Sidebar inputs stay dark ── */
  div[data-testid="stSidebarContent"] .stMultiSelect [data-baseweb="tag"] {
    background: #2a3a5a !important;
  }
  div[data-testid="stSidebarContent"] .stTextInput input {
    background: #232f4b !important;
    color: #e8ecf4 !important;
    border-color: #2a3a5a !important;
    border-radius: 8px !important;
  }
  div[data-testid="stSidebarContent"] .stSlider [data-testid="stTickBar"] {
    color: #4a5a7a !important;
  }
  /* Sidebar multiselect input area */
  div[data-testid="stSidebarContent"] .stMultiSelect [data-baseweb="select"] > div {
    background: #232f4b !important;
    border-color: #2a3a5a !important;
    color: #e8ecf4 !important;
  }
  div[data-testid="stSidebarContent"] .stMultiSelect input {
    color: #e8ecf4 !important;
  }
</style>
""", unsafe_allow_html=True)

custom_facilities = get_custom_facilities()
PHARMACY_TYPES = {"walgreens", "cvs", "walmart", "other"}
custom_hospitals  = [f for f in custom_facilities if f["type"] == "hospital"]
custom_pharmacies = [f for f in custom_facilities if f["type"] in PHARMACY_TYPES]

with st.sidebar:
    st.markdown('<span class="snap-logo">SNAP</span>', unsafe_allow_html=True)
    st.markdown('<div class="snap-tagline">Medical Distribution Intelligence</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Service Radius</div>', unsafe_allow_html=True)
    max_radius = st.slider("Max Radius (km)", min_value=10, max_value=200, value=80, step=5, label_visibility="collapsed")
    st.caption(f"Current: **{max_radius} km**")

    st.markdown('<div class="sidebar-section">Hospitals</div>', unsafe_allow_html=True)
    all_hospital_options = [h["name"] for h in HOSPITALS] + [h["name"] for h in custom_hospitals]
    selected_hospitals = st.multiselect(
        "Active Hospitals",
        options=all_hospital_options,
        default=all_hospital_options,
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-section">Pharmacies</div>', unsafe_allow_html=True)
    include_walgreens = st.checkbox("Walgreens (5)", value=True)
    include_cvs       = st.checkbox("CVS (5)", value=True)
    include_walmart   = st.checkbox("Walmart Pharmacy (5)", value=True)

    if custom_pharmacies:
      st.markdown(
        '<div style="font-size:0.68rem;color:#4a5a7a;margin:0.5rem 0 0.25rem;'
        'text-transform:uppercase;letter-spacing:0.08em">Custom Pharmacies</div>',
        unsafe_allow_html=True,
      )
      selected_custom_pharmacies = st.multiselect(
        "Custom Pharmacies",
        options=[p["name"] for p in custom_pharmacies],
        default=[p["name"] for p in custom_pharmacies],
        label_visibility="collapsed",
      )
    else:
      selected_custom_pharmacies = []

    render_location_search()

active_hospitals = ([
    h for h in HOSPITALS
    if h["name"] in selected_hospitals] + [h for h in custom_hospitals if h["name"] in selected_hospitals])
active_pharmacies = (
    [p for p in PHARMACIES 
    if (p["type"] == "walgreens" and include_walgreens)
    or (p["type"] == "cvs"      and include_cvs)
    or (p["type"] == "walmart"  and include_walmart)] + [p for p in custom_pharmacies if p["name"] in selected_custom_pharmacies]
)
custom_facilities = get_custom_facilities()
active_facilities = active_hospitals + active_pharmacies

@st.cache_data(show_spinner=False)
def compute_location(static_names: tuple, radius: float, custom_json: str):
    import json as _json
    from data.hospitals import ALL_FACILITIES as _ALL
 
    static = [f for f in _ALL if f["name"] in static_names]
 
    custom = _json.loads(custom_json) if custom_json else []
 
    facilities = static + custom
    if not facilities:
        return None
    return find_optimal_location(facilities, max_radius_km=radius)

import json as _json
 
loc_result = compute_location(
    static_names=tuple(
        f["name"] for f in active_facilities if not f.get("custom")
    ),
    radius=max_radius,
    custom_json=_json.dumps(
        [f for f in active_facilities if f.get("custom")],
        sort_keys=True,
    ),
)
tab_dash, tab_dosage, tab_agent = st.tabs(["Distribution Map", "Dosage Calculator", "AI Agent"])

with tab_dash:
    st.markdown('<div class="page-title">Distribution Center Optimizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Demand-weighted centroid placement across the South Florida facility network</div>', unsafe_allow_html=True)

    if not active_facilities:
        st.warning("No facilities selected. Choose at least one hospital or pharmacy from the sidebar.")
    elif loc_result is None:
        st.error("Could not compute location — no facilities found.")
    else:
        outside_count = len(loc_result.get("hospitals_outside_radius", []))
        total_doses   = sum(f["daily_doses_needed"] for f in active_facilities)

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Optimal Center</div>
                <div class="stat-value accent" style="font-size:1rem">{loc_result['optimal_lat']}°N<br>{abs(loc_result['optimal_lon'])}°W</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Avg Weighted Dist.</div>
                <div class="stat-value accent">{loc_result['weighted_avg_distance_km']} <span style="font-size:1rem;color:#8a97b0">km</span></div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Daily Doses</div>
                <div class="stat-value">{total_doses:,}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Active Facilities</div>
                <div class="stat-value">{len(active_hospitals)} <span style="font-size:0.9rem;color:#8a97b0">hosp</span> · {len(active_pharmacies)} <span style="font-size:0.9rem;color:#8a97b0">rx</span></div>
            </div>""", unsafe_allow_html=True)
        with c5:
            warn_color = "warn" if outside_count > 0 else "accent"
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Coverage ({max_radius} km)</div>
                <div class="stat-value {warn_color}">{"⚠ " + str(outside_count) + " outside" if outside_count else "✓ All covered"}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        map_col, table_col = st.columns([3, 2], gap="medium")

        with map_col:
            st.markdown('<div class="section-label">Facility Network Map</div>', unsafe_allow_html=True)

            CENTER_LAT = loc_result["optimal_lat"]
            CENTER_LON = loc_result["optimal_lon"]

            m = folium.Map(
                location=[CENTER_LAT, CENTER_LON],
                zoom_start=10,
                tiles="CartoDB positron",
            )

            folium.Marker(
                [CENTER_LAT, CENTER_LON],
                tooltip="📍 Optimal Distribution Center",
                icon=folium.Icon(color="green", icon="plus-sign"),
            ).add_to(m)

            type_colors = {"hospital": "blue", "walgreens": "purple", "cvs": "orange", "walmart": "cadetblue"}
            outside_set = set(loc_result.get("hospitals_outside_radius", []))

            for f in active_facilities:
                color = "red" if f["name"] in outside_set else type_colors.get(f.get("type", "hospital"), "gray")
                folium.Marker(
                    [f["lat"], f["lon"]],
                    tooltip=f"{f['name']} — {f['daily_doses_needed']} doses/day",
                    icon=folium.Icon(color=color, icon="info-sign"),
                ).add_to(m)

            st_folium(m, width="100%", height=420, returned_objects=[])

            # Legend
            st.markdown("""
            <div style="display:flex;gap:1rem;flex-wrap:wrap;font-size:0.73rem;color:#6b7a9a;margin-top:0.4rem">
              <span>🟢 Distribution Center</span>
              <span>🔵 Hospital</span>
              <span>🟣 Walgreens</span>
              <span>🟠 CVS</span>
              <span>🔷 Walmart Pharmacy</span>
              <span>🔴 Outside Radius</span>
            </div>
            """, unsafe_allow_html=True)

        with table_col:
            st.markdown('<div class="section-label">Facility Distance Breakdown</div>', unsafe_allow_html=True)

            rows = []
            for hd in loc_result["hospital_distances"]:
                f_data  = next((f for f in active_facilities if f["name"] == hd["name"]), None)
                outside = hd["name"] in outside_set
                ftype   = f_data.get("type", "hospital") if f_data else "unknown"
                emoji   = {"hospital": "🏥", "walgreens": "🟣", "cvs": "🟠", "walmart": "🔷"}.get(ftype, "📍")
                rows.append({
                    "Facility":       emoji + " " + hd["name"],
                    "Distance (km)":  hd["distance_km"],
                    "Daily Doses":    hd["daily_doses_needed"],
                    "Status":         "⚠ Outside" if outside else "✓ OK",
                })

            df = pd.DataFrame(rows)
            st.dataframe(
                df,
                width='content',
                hide_index=True,
                height=420,
                column_config={
                    "Distance (km)": st.column_config.NumberColumn(format="%.1f km"),
                    "Daily Doses":   st.column_config.NumberColumn(format="%d"),
                },
            )

with tab_dosage:
    st.markdown('<div class="page-title">Dosage Calculator</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Weight-based safe dosing with clinical maximum cap</div>', unsafe_allow_html=True)

    calc_col, result_col = st.columns([1, 1], gap="large")

    with calc_col:
        st.markdown('<div class="section-label">Patient & Drug Parameters</div>', unsafe_allow_html=True)

        drug_choice  = st.selectbox("Select Drug", options=[d["name"] for d in DRUG_FORMULARY], index=0)
        selected_drug = next(d for d in DRUG_FORMULARY if d["name"] == drug_choice)

        weight = st.number_input("Patient Weight (kg)", min_value=0.5, max_value=300.0, value=70.0, step=0.5)

        mg_per_kg = st.number_input(
            "mg/kg Dose Factor",
            min_value=0.001, max_value=100.0,
            value=float(selected_drug["mg_per_kg"]),
            step=0.1,
            help=f"Default for {drug_choice}: {selected_drug['mg_per_kg']} mg/kg",
        )

        max_dose = st.number_input(
            "Maximum Single Dose (mg)",
            min_value=0.1, max_value=5000.0,
            value=float(selected_drug["max_dose_mg"]),
            step=1.0,
            help=f"Clinical max for {drug_choice}: {selected_drug['max_dose_mg']} mg",
        )

        calculate_btn = st.button("Calculate Safe Dose")

        st.markdown('<div class="section-label" style="margin-top:1.8rem">Drug Formulary</div>', unsafe_allow_html=True)
        for drug in DRUG_FORMULARY:
            cold    = drug["cold_chain"]
            badge   = f'<span class="drug-badge badge-cold">❄ Cold Chain</span>' if cold else f'<span class="drug-badge badge-warm">✓ Ambient</span>'
            active  = "border: 2px solid #0ea875;" if drug["name"] == drug_choice else ""
            st.markdown(f"""
            <div class="drug-card" style="{active}">
              <div style="flex:1">
                <div class="drug-name">{drug['name']}</div>
                <div class="drug-meta">{drug['mg_per_kg']} mg/kg · max {drug['max_dose_mg']} mg · {drug['category']}</div>
              </div>
              {badge}
            </div>
            """, unsafe_allow_html=True)

    with result_col:
        st.markdown('<div class="section-label">Result</div>', unsafe_allow_html=True)

        if calculate_btn:
            try:
                dose_result = compute_dosage(weight, mg_per_kg, max_dose)
                capped      = "exceeded" in dose_result["detail"].lower()
                cap_class   = "capped" if capped else ""

                st.markdown(f"""
                <div class="dose-result-card">
                  <div style="font-size:0.72rem;font-weight:600;color:#8a97b0;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.4rem">Safe Dose</div>
                  <div class="dose-big {cap_class}">{dose_result['result']} <span style="font-size:1.4rem">mg</span></div>
                  <div class="dose-detail {cap_class}">{'⚠️' if capped else '✅'} {dose_result['detail']}</div>
                  {"<div style='margin-top:0.8rem;padding:0.6rem;background:#eff6ff;border-radius:8px;font-size:0.8rem;color:#1d4ed8'>❄️ Cold-chain drug — requires refrigerated storage and transport (2–8°C).</div>" if selected_drug.get('cold_chain') else ""}
                </div>
                """, unsafe_allow_html=True)

            except ValueError as e:
                st.error(f"Input error: {e}")
        else:
            st.markdown("""
            <div style="margin-top:3rem;text-align:center;color:#8a97b0;font-size:0.88rem;line-height:1.7">
              Select a drug and enter patient weight,<br>then click <strong>Calculate Safe Dose</strong>.
            </div>
            """, unsafe_allow_html=True)

with tab_agent:
    st.markdown('<div class="page-title">SNAP AI Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">LangChain · llama3.1 (Ollama) · RAG · Tool-Calling</div>', unsafe_allow_html=True)

    if "messages"    not in st.session_state: st.session_state.messages    = []
    if "agent_steps" not in st.session_state: st.session_state.agent_steps = {}

    st.markdown('<div class="section-label">Suggested Queries</div>', unsafe_allow_html=True)
    suggestions = [
        "Where should we place the distribution center for all South Florida hospitals?",
        "Calculate the morphine dose for a 55 kg patient.",
        "What are the cold chain requirements for insulin distribution?",
    ]
    sc1, sc2, sc3 = st.columns(3)
    for col, sug in zip([sc1, sc2, sc3], suggestions):
        with col:
            if st.button(sug, key=f"sug_{sug[:20]}"):
                st.session_state.pending_input = sug

    st.markdown("")

    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-agent">{msg["content"]}</div>', unsafe_allow_html=True)
            steps = st.session_state.agent_steps.get(i, [])
            if steps:
                with st.expander(f"🔧 {len(steps)} tool call(s)"):
                    for step in steps:
                        st.markdown(
                            f'<div class="tool-call-box">'
                            f'▶ <strong>{step["tool"]}</strong><br>'
                            f'Input: {json.dumps(step["tool_input"], indent=2)}<br>'
                            f'Result: {str(step["result"])[:300]}{"..." if len(str(step["result"])) > 300 else ""}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    user_input = st.chat_input("Ask about distribution logistics, dosages, or hospital coverage…")

    if "pending_input" in st.session_state:
        user_input = st.session_state.pop("pending_input")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("Agent thinking…"):
            try:
                from agent.snap_agent import run_agent
                history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
                response = run_agent(user_input, chat_history=history, extra_facilities=get_custom_facilities())
                answer   = response["output"]
                steps    = response["intermediate_steps"]
                msg_idx  = len(st.session_state.messages)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.agent_steps[msg_idx] = steps
            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"Agent error: {e}"})

        st.rerun()