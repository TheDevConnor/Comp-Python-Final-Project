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
from data.hospitals import HOSPITALS, DRUG_FORMULARY
from rag.knowledge_base import build_or_load_vectorstore

st.set_page_config(
    page_title="SNAP – Medical Distribution Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }
  .stApp {
    background: #0a0f1e;
    color: #e0e6f0;
  }
  .main-header {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #00d4aa;
    letter-spacing: -0.02em;
    margin-bottom: 0;
  }
  .sub-header {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    color: #6b7a99;
    margin-top: 0;
    margin-bottom: 1.5rem;
  }
  .metric-card {
    background: #111827;
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
  }
  .metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #00d4aa;
  }
  .metric-label {
    font-size: 0.75rem;
    color: #6b7a99;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .tool-badge {
    display: inline-block;
    background: #0d2137;
    border: 1px solid #00d4aa44;
    border-radius: 4px;
    padding: 2px 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #00d4aa;
    margin-right: 6px;
  }
  .agent-response {
    background: #0d1929;
    border-left: 3px solid #00d4aa;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.92rem;
    line-height: 1.6;
  }
  .user-msg {
    background: #111827;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.92rem;
  }
  .step-box {
    background: #07111f;
    border: 1px solid #1e2d4a;
    border-radius: 6px;
    padding: 0.6rem 0.9rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #4a8fbb;
    margin-top: 0.3rem;
  }
  .section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #6b7a99;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border-bottom: 1px solid #1e2d4a;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
  }
  .stTextInput > div > div > input {
    background: #111827 !important;
    color: #e0e6f0 !important;
    border: 1px solid #1e2d4a !important;
    border-radius: 8px !important;
  }
  .stSelectbox > div > div {
    background: #111827 !important;
    color: #e0e6f0 !important;
  }
  .stSlider > div {
    color: #e0e6f0 !important;
  }
  div[data-testid="stSidebarContent"] {
    background: #070d1a;
    border-right: 1px solid #1e2d4a;
  }
  .stButton > button {
    background: #00d4aa !important;
    color: #0a0f1e !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.4rem 1.2rem !important;
  }
  .stButton > button:hover {
    background: #00efc0 !important;
  }
  .warning-box {
    background: #1a1000;
    border: 1px solid #f59e0b44;
    border-radius: 6px;
    padding: 0.6rem 0.9rem;
    color: #f59e0b;
    font-size: 0.85rem;
  }
  .ok-box {
    background: #001a10;
    border: 1px solid #00d4aa44;
    border-radius: 6px;
    padding: 0.6rem 0.9rem;
    color: #00d4aa;
    font-size: 0.85rem;
  }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="main-header">🏥 SNAP</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Medical Distribution Intelligence</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Configuration</div>', unsafe_allow_html=True)
    api_key = st.text_input("Google Gemini API Key", type="password", placeholder="AIza...")

    st.markdown('<div class="section-title">Distribution Settings</div>', unsafe_allow_html=True)
    max_radius = st.slider("Max Service Radius (km)", min_value=10, max_value=200, value=80, step=5)

    selected_hospitals = st.multiselect(
        "Active Hospitals",
        options=[h["name"] for h in HOSPITALS],
        default=[h["name"] for h in HOSPITALS],
    )

    st.markdown('<div class="section-title">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.78rem; color:#6b7a99; line-height:1.6">
    SNAP uses a LangChain-powered Gemini agent with two Non-LLM tools:<br><br>
    <span class="tool-badge">compute_dosage</span> Weight-based safe dosing<br><br>
    <span class="tool-badge">find_optimal_location</span> Weighted centroid placement<br><br>
    Plus a ChromaDB RAG pipeline for medical logistics domain knowledge.
    </div>
    """, unsafe_allow_html=True)

active_hospitals = [h for h in HOSPITALS if h["name"] in selected_hospitals]

@st.cache_data(show_spinner=False)
def compute_location(hospital_names: tuple, radius: float):
    hospitals = [h for h in HOSPITALS if h["name"] in hospital_names]
    if not hospitals:
        return None
    return find_optimal_location(hospitals, max_radius_km=radius)


loc_result = compute_location(tuple(selected_hospitals), max_radius)

tab_dash, tab_dosage, tab_agent = st.tabs(["📍 Distribution Map", "💊 Dosage Calculator", "🤖 AI Agent"])

with tab_dash:
    st.markdown('<div class="main-header">Distribution Center Optimizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Demand-weighted centroid placement across South Florida hospital network</div>', unsafe_allow_html=True)

    if not active_hospitals:
        st.warning("Select at least one hospital in the sidebar.")
    elif loc_result:

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Optimal Center</div>
                <div class="metric-value" style="font-size:1rem">{loc_result['optimal_lat']:.4f}°N</div>
                <div class="metric-value" style="font-size:1rem">{abs(loc_result['optimal_lon']):.4f}°W</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Avg Weighted Distance</div>
                <div class="metric-value">{loc_result['weighted_avg_distance_km']:.1f} km</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            total_doses = sum(h["daily_doses_needed"] for h in active_hospitals)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Daily Doses Required</div>
                <div class="metric-value">{total_doses:,}</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            coverage_color = "#00d4aa" if loc_result["coverage_ok"] else "#f59e0b"
            coverage_text = "✓ FULL" if loc_result["coverage_ok"] else f"⚠ {len(loc_result['hospitals_outside_radius'])} OUTSIDE"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Coverage ({max_radius} km)</div>
                <div class="metric-value" style="color:{coverage_color}">{coverage_text}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        col_map, col_table = st.columns([3, 2])

        with col_map:
            st.markdown('<div class="section-title">Hospital Network Map</div>', unsafe_allow_html=True)

            m = folium.Map(
                location=[loc_result["optimal_lat"], loc_result["optimal_lon"]],
                zoom_start=9,
                tiles="CartoDB dark_matter",
            )

            folium.Marker(
                location=[loc_result["optimal_lat"], loc_result["optimal_lon"]],
                popup=folium.Popup(
                    f"<b>Recommended Distribution Center</b><br>"
                    f"Lat: {loc_result['optimal_lat']:.4f}, Lon: {loc_result['optimal_lon']:.4f}<br>"
                    f"Weighted avg distance: {loc_result['weighted_avg_distance_km']:.1f} km",
                    max_width=280,
                ),
                icon=folium.Icon(color="green", icon="industry", prefix="fa"),
                tooltip="Distribution Center",
            ).add_to(m)

            folium.Circle(
                location=[loc_result["optimal_lat"], loc_result["optimal_lon"]],
                radius=max_radius * 1000,
                color="#00d4aa",
                fill=True,
                fill_opacity=0.06,
                weight=1.5,
                tooltip=f"{max_radius} km service radius",
            ).add_to(m)

            for hd in loc_result["hospital_distances"]:
                h_data = next((h for h in active_hospitals if h["name"] == hd["name"]), None)
                if not h_data:
                    continue

                outside = hd["name"] in loc_result.get("hospitals_outside_radius", [])
                icon_color = "red" if outside else "blue"

                folium.Marker(
                    location=[hd["lat"], hd["lon"]],
                    popup=folium.Popup(
                        f"<b>{hd['name']}</b><br>"
                        f"Distance to center: {hd['distance_km']:.1f} km<br>"
                        f"Daily doses: {hd['daily_doses_needed']}<br>"
                        f"Patients: {h_data.get('patients', 'N/A')}<br>"
                        f"Specialty: {h_data.get('specialty', 'N/A')}",
                        max_width=280,
                    ),
                    icon=folium.Icon(color=icon_color, icon="plus", prefix="fa"),
                    tooltip=hd["name"],
                ).add_to(m)

                folium.PolyLine(
                    locations=[[hd["lat"], hd["lon"]], [loc_result["optimal_lat"], loc_result["optimal_lon"]]],
                    color="#00d4aa" if not outside else "#f59e0b",
                    weight=1.5,
                    opacity=0.5,
                    dash_array="6",
                ).add_to(m)

            st_folium(m, width=None, height=450)

        with col_table:
            st.markdown('<div class="section-title">Hospital Distance Breakdown</div>', unsafe_allow_html=True)

            rows = []
            for hd in loc_result["hospital_distances"]:
                outside = hd["name"] in loc_result.get("hospitals_outside_radius", [])
                rows.append({
                    "Hospital": hd["name"].replace(" Hospital", "").replace(" Medical Center", ""),
                    "Distance (km)": hd["distance_km"],
                    "Daily Doses": hd["daily_doses_needed"],
                    "Status": "⚠ Outside" if outside else "✓ OK",
                })

            df = pd.DataFrame(rows)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Distance (km)": st.column_config.NumberColumn(format="%.1f km"),
                    "Daily Doses": st.column_config.NumberColumn(format="%d"),
                },
            )

            st.markdown("---")
            st.markdown('<div class="section-title">Drug Formulary</div>', unsafe_allow_html=True)
            for drug in DRUG_FORMULARY:
                cold = "❄️" if drug["cold_chain"] else "🌡️"
                st.markdown(
                    f"**{drug['name']}** {cold}  \n"
                    f"`{drug['mg_per_kg']} mg/kg` · max `{drug['max_dose_mg']} mg`  \n"
                    f"<span style='color:#6b7a99;font-size:0.78rem'>{drug['category']}</span>",
                    unsafe_allow_html=True,
                )

with tab_dosage:
    st.markdown('<div class="main-header">Dosage Calculator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Weight-based safe dosing with clinical maximum cap</div>', unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown('<div class="section-title">Patient & Drug Parameters</div>', unsafe_allow_html=True)

        drug_choice = st.selectbox(
            "Select Drug",
            options=[d["name"] for d in DRUG_FORMULARY],
            index=0,
        )
        selected_drug = next(d for d in DRUG_FORMULARY if d["name"] == drug_choice)

        weight = st.number_input(
            "Patient Weight (kg)",
            min_value=0.5,
            max_value=300.0,
            value=70.0,
            step=0.5,
        )

        mg_per_kg = st.number_input(
            "mg/kg Dose Factor",
            min_value=0.001,
            max_value=100.0,
            value=float(selected_drug["mg_per_kg"]),
            step=0.1,
            help=f"Default for {drug_choice}: {selected_drug['mg_per_kg']} mg/kg",
        )

        max_dose = st.number_input(
            "Maximum Single Dose (mg)",
            min_value=0.1,
            max_value=5000.0,
            value=float(selected_drug["max_dose_mg"]),
            step=1.0,
            help=f"Clinical max for {drug_choice}: {selected_drug['max_dose_mg']} mg",
        )

        calculate_btn = st.button("Calculate Safe Dose")

    with col_result:
        st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)

        if calculate_btn:
            try:
                dose_result = compute_dosage(weight, mg_per_kg, max_dose)
                capped = "capped" in dose_result["detail"].lower() and "exceeded" in dose_result["detail"]

                box_class = "warning-box" if capped else "ok-box"
                icon = "⚠️" if capped else "✅"

                st.markdown(f"""
                <div class="metric-card" style="margin-bottom:1rem">
                    <div class="metric-label">Safe Dose</div>
                    <div class="metric-value">{dose_result['result']} {dose_result['unit']}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="{box_class}">
                {icon} {dose_result['detail']}
                </div>
                """, unsafe_allow_html=True)

                if selected_drug.get("cold_chain"):
                    st.markdown("""
                    <div class="warning-box" style="margin-top:0.5rem">
                    ❄️ Cold-chain drug — requires refrigerated storage and transport (2–8°C).
                    </div>
                    """, unsafe_allow_html=True)

            except ValueError as e:
                st.error(f"Input error: {e}")
        else:
            st.markdown("""
            <div style="color:#6b7a99; font-size:0.88rem; margin-top:2rem; text-align:center">
                Enter patient and drug parameters, then click <strong>Calculate Safe Dose</strong>.
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Batch Dosage Reference — All Drugs</div>', unsafe_allow_html=True)

    ref_weight = st.slider("Reference Patient Weight (kg)", 10, 120, 70, 5)
    batch_rows = []
    for drug in DRUG_FORMULARY:
        result = compute_dosage(ref_weight, drug["mg_per_kg"], drug["max_dose_mg"])
        calc = ref_weight * drug["mg_per_kg"]
        batch_rows.append({
            "Drug": drug["name"],
            "Category": drug["category"],
            "Cold Chain": "❄️" if drug["cold_chain"] else "",
            "Calculated (mg)": round(calc, 2),
            "Safe Dose (mg)": result["result"],
            "Capped?": "⚠ Yes" if calc > drug["max_dose_mg"] else "✓ No",
        })

    st.dataframe(pd.DataFrame(batch_rows), use_container_width=True, hide_index=True)

with tab_agent:
    st.markdown('<div class="main-header">SNAP AI Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">LangChain · Gemini 1.5 Flash · RAG · Tool-Calling</div>', unsafe_allow_html=True)

    if not api_key:
        st.info("Enter your Google Gemini API key in the sidebar to activate the AI agent.")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "agent_steps" not in st.session_state:
            st.session_state.agent_steps = {}

        st.markdown('<div class="section-title">Suggested Queries</div>', unsafe_allow_html=True)
        sc1, sc2, sc3 = st.columns(3)
        suggestions = [
            "Where should we place the distribution center for all South Florida hospitals?",
            "Calculate the morphine dose for a 55 kg patient.",
            "What are the cold chain requirements for insulin distribution?",
        ]
        for col, suggestion in zip([sc1, sc2, sc3], suggestions):
            with col:
                if st.button(suggestion, key=f"sug_{suggestion[:20]}"):
                    st.session_state.pending_input = suggestion

        st.markdown("---")

        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                st.markdown(f'<div class="user-msg">👤 {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="agent-response">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
                # Show tool steps if available
                steps = st.session_state.agent_steps.get(i, [])
                if steps:
                    with st.expander(f"🔧 {len(steps)} tool call(s)"):
                        for step in steps:
                            st.markdown(
                                f'<div class="step-box">▶ <b>{step["tool"]}</b><br>'
                                f'Input: {json.dumps(step["tool_input"], indent=2)}<br>'
                                f'Result: {str(step["result"])[:300]}{"..." if len(str(step["result"])) > 300 else ""}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

        user_input = st.chat_input("Ask about distribution logistics, dosages, or hospital coverage...")
        if "pending_input" in st.session_state:
            user_input = st.session_state.pop("pending_input")

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})

            with st.spinner("Agent thinking..."):
                try:
                    from agent.snap_agent import run_agent

                    # Build chat history for context
                    history = []
                    for msg in st.session_state.messages[:-1]:
                        history.append({"role": msg["role"], "content": msg["content"]})

                    response = run_agent(
                        user_input,
                        chat_history=history,
                        api_key=api_key,
                    )

                    answer = response["output"]
                    steps = response["intermediate_steps"]

                    msg_idx = len(st.session_state.messages)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.agent_steps[msg_idx] = steps

                except Exception as e:
                    err_msg = f"Agent error: {e}"
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})

            st.rerun()
