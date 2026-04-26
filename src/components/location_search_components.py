import requests
import streamlit as st

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_HEADERS = {"User-Agent": "SNAP-Medical-Distribution/1.0"}

FACILITY_TYPES = ["hospital", "walgreens", "cvs", "walmart", "other"]
TYPE_EMOJIS    = {"hospital": "🏥", "walgreens": "🟣", "cvs": "🟠", "walmart": "🔷", "other": "⭐"}

def geocode_address(query: str) -> list[dict]:
    """Return up to 5 candidate results from Nominatim for the given query."""
    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={
                "q": query,
                "format": "json",
                "addressdetails": 1,
                "limit": 5,
                "countrycodes": "us",
            },
            headers=NOMINATIM_HEADERS,
            timeout=8,
        )
        resp.raise_for_status()
        results = resp.json()
        return [
            {
                "display_name": r["display_name"],
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
            }
            for r in results
        ]
    except requests.exceptions.Timeout:
        st.error("Geocoding timed out — check your internet connection.")
        return []
    except requests.exceptions.RequestException as e:
        st.error(f"Geocoding error: {e}")
        return []

def _init_state():
    if "custom_facilities" not in st.session_state:
        st.session_state["custom_facilities"] = []
    if "_loc_candidates" not in st.session_state:
        st.session_state["_loc_candidates"] = []
    if "_loc_selected_idx" not in st.session_state:
        st.session_state["_loc_selected_idx"] = 0


def get_custom_facilities() -> list[dict]:
    """Call this from app.py to get the current custom facility list."""
    _init_state()
    return st.session_state["custom_facilities"]


def remove_custom_facility(name: str):
    st.session_state["custom_facilities"] = [
        f for f in st.session_state["custom_facilities"] if f["name"] != name
    ]

def render_location_search():
    """
    Renders the full Add Custom Location UI.
    Call this inside a st.sidebar block or any tab.
    """
    _init_state()

    st.markdown(
        '<div class="sidebar-section" style="margin-top:1.2rem">Custom Locations</div>',
        unsafe_allow_html=True,
    )

    query = st.text_input(
        "Search address or place name",
        placeholder="e.g. 123 Main St, Miami, FL",
        key="_loc_query",
        label_visibility="collapsed",
    )

    col_search, col_clear = st.columns([3, 1])
    with col_search:
        search_clicked = st.button("🔍 Search", key="_loc_search_btn", use_container_width=True)
    with col_clear:
        if st.button("✕", key="_loc_clear_btn", use_container_width=True):
            st.session_state["_loc_candidates"] = []
            st.session_state["_loc_selected_idx"] = 0
            st.rerun()

    if search_clicked and query.strip():
        with st.spinner("Looking up location…"):
            candidates = geocode_address(query.strip())
        if candidates:
            st.session_state["_loc_candidates"] = candidates
            st.session_state["_loc_selected_idx"] = 0
        else:
            st.warning("No results found. Try a more specific address.")

    candidates = st.session_state["_loc_candidates"]

    if candidates:
        labels = [c["display_name"][:72] + ("…" if len(c["display_name"]) > 72 else "")
                  for c in candidates]
        chosen_idx = st.selectbox(
            "Select the correct location",
            options=range(len(candidates)),
            format_func=lambda i: labels[i],
            key="_loc_picker",
            label_visibility="collapsed",
        )
        chosen = candidates[chosen_idx]

        st.caption(f"📍 {chosen['lat']:.5f}, {chosen['lon']:.5f}")

        st.markdown(
            '<div style="font-size:0.72rem;font-weight:600;color:#8a97b0;'
            'text-transform:uppercase;letter-spacing:0.08em;margin:0.6rem 0 0.3rem">Facility details</div>',
            unsafe_allow_html=True,
        )

        custom_name = st.text_input(
            "Facility name",
            value=chosen["display_name"].split(",")[0],
            key="_loc_name",
            label_visibility="collapsed",
            placeholder="Facility name",
        )
        facility_type = st.selectbox(
            "Facility type",
            options=FACILITY_TYPES,
            key="_loc_type",
            label_visibility="collapsed",
        )
        daily_doses = st.number_input(
            "Daily doses needed",
            min_value=1,
            max_value=10_000,
            value=300,
            step=10,
            key="_loc_doses",
            label_visibility="collapsed",
        )

        if st.button("➕ Add to network", key="_loc_add_btn", use_container_width=True):
            existing_names = {f["name"] for f in st.session_state["custom_facilities"]}
            if custom_name.strip() in existing_names:
                st.warning(f'"{custom_name}" is already in your custom locations.')
            elif not custom_name.strip():
                st.warning("Please enter a facility name.")
            else:
                new_facility = {
                    "name": custom_name.strip(),
                    "lat": chosen["lat"],
                    "lon": chosen["lon"],
                    "daily_doses_needed": daily_doses,
                    "patients": None,
                    "specialty": "Custom",
                    "address": chosen["display_name"],
                    "type": facility_type,
                    "custom": True,
                }
                st.session_state["custom_facilities"].append(new_facility)
                st.session_state["_loc_candidates"] = []
                st.success(f"Added {TYPE_EMOJIS.get(facility_type, '📍')} {custom_name}")
                st.rerun()

    custom = st.session_state["custom_facilities"]
    if custom:
        st.markdown(
            f'<div style="font-size:0.68rem;color:#4a5a7a;margin:0.8rem 0 0.3rem;'
            f'text-transform:uppercase;letter-spacing:0.08em">{len(custom)} custom location(s)</div>',
            unsafe_allow_html=True,
        )
        for fac in list(custom):
            emoji = TYPE_EMOJIS.get(fac["type"], "⭐")
            col_name, col_del = st.columns([5, 1])
            with col_name:
                st.markdown(
                    f'<div style="font-size:0.78rem;color:#e8ecf4;padding:2px 0">'
                    f'{emoji} {fac["name"]}<br>'
                    f'<span style="font-size:0.65rem;color:#5a6a8a">'
                    f'{fac["daily_doses_needed"]} doses/day</span></div>',
                    unsafe_allow_html=True,
                )
            with col_del:
                if st.button("✕", key=f"_del_{fac['name']}", help=f"Remove {fac['name']}"):
                    remove_custom_facility(fac["name"])
                    st.rerun()