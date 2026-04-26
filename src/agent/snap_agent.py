import json
import os
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent      
from langchain_core.tools import StructuredTool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field

from tools.compute_dosage import compute_dosage
from tools.find_optimal_location import find_optimal_location
from rag.knowledge_base import query_knowledge
from data.hospitals import HOSPITALS, ALL_FACILITIES

class DosageInput(BaseModel):
    weight_kg: float = Field(..., description="Patient weight in kilograms. Must be > 0.")
    drug_mg_per_kg: float = Field(..., description="Dose in mg per kg body weight. Must be > 0.")
    max_dose_mg: float = Field(..., description="Maximum allowable single dose in mg. Must be > 0.")


class LocationInput(BaseModel):
    hospital_names: Optional[list[str]] = Field(
        default=None,
        description=(
            "Optional list of facility names to include. "
            "If omitted, ALL facilities in the system are used."
        ),
    )
    max_radius_km: Optional[float] = Field(
        default=None,
        description="Optional maximum service radius in km. Flags any facility outside this range.",
    )


class RAGInput(BaseModel):
    question: str = Field(..., description="Medical logistics question to look up in the knowledge base.")

def _dosage_tool(weight_kg: float, drug_mg_per_kg: float, max_dose_mg: float) -> str:
    try:
        result = compute_dosage(weight_kg, drug_mg_per_kg, max_dose_mg)
        return json.dumps(result)
    except ValueError as e:
        return f"Error: {e}"

def _make_location_tool(extra_facilities: list[dict] | None = None):
    """
    Returns a closure that includes any user-added custom facilities
    alongside the static ALL_FACILITIES list.
    """
    def _location_tool(
        hospital_names: Optional[list[str]] = None,
        max_radius_km: Optional[float] = None,
    ) -> str:
        try:
            base = ALL_FACILITIES + (extra_facilities or [])
            facilities = base
            if hospital_names:
                facilities = [f for f in base if f["name"] in hospital_names]
                if not facilities:
                    return "Error: None of the specified facility names were found in the database."
            result = find_optimal_location(facilities, max_radius_km=max_radius_km)
            return json.dumps(result)
        except ValueError as e:
            return f"Error: {e}"

    return _location_tool

def _rag_tool(question: str) -> str:
    passages = query_knowledge(question, n_results=3)
    if not passages:
        return "No relevant information found in the knowledge base."
    return "\n\n---\n\n".join(passages)

def _build_system_prompt(extra_facilities: list[dict] | None = None) -> str:
    custom_section = ""
    if extra_facilities:
        names = ", ".join(f["name"] for f in extra_facilities)
        custom_section = (
            f"\n\nThe user has also added {len(extra_facilities)} custom location(s) "
            f"to the network: {names}. These are included in find_optimal_location automatically."
        )

    return (
        "You are SNAP (Systems for Niche Agentic Programming), an AI logistics agent "
        "specialized in medical supply chain optimization for South Florida hospitals and retail pharmacies.\n\n"
        "You have access to three tools:\n"
        "1. **compute_dosage** — Calculates safe medication doses based on patient weight.\n"
        "2. **find_optimal_location** — Finds the optimal GPS location for a medical distribution center.\n"
        "3. **lookup_medical_logistics** — Retrieves domain knowledge about regulations, cold chain, and guidelines.\n\n"
        "The facility network includes 8 hospitals (Jackson Memorial, Baptist Health, Broward Health, etc.) "
        "and 15 retail pharmacies: 5 Walgreens, 5 CVS, and 5 Walmart Pharmacy locations across Miami-Dade, "
        "Broward, and Palm Beach counties."
        + custom_section
        + "\n\nRULES:\n"
        "- Always use tools for calculations. Never guess dosages or distances.\n"
        "- For general medical logistics questions, use lookup_medical_logistics first.\n"
        "- Be precise and professional. You serve healthcare workers and operations teams.\n"
        "- When reporting dosage results, always include the detail field explaining whether the dose was capped.\n"
        "- When reporting location results, always report the weighted average distance and flag any "
        "facilities outside radius.\n"
        "- If the user's question involves both dosage and location, call both tools.\n"
        "- Format numerical results clearly with units.\n"
        "- When asked about pharmacies specifically, note that Walmart locations have the highest "
        "daily dose volume among retail pharmacies.\n"
    )

def build_agent(api_key: Optional[str] = None, extra_facilities: list[dict] | None = None):
    """
    Build a LangGraph ReAct agent.

    Parameters
    ----------
    api_key          : Google Gemini API key (falls back to GOOGLE_API_KEY env var).
    extra_facilities : Custom facilities added by the user via the location search UI.
                       Passed in from app.py as get_custom_facilities().
    """
    key = api_key or os.environ.get("GOOGLE_API_KEY", "")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=key,
        temperature=0.2,
    )

    tools = [
        StructuredTool.from_function(
            func=_dosage_tool,
            name="compute_dosage",
            description=(
                "Calculates the safe medication dose for a patient given their weight (kg), "
                "the drug's mg/kg dosing factor, and the maximum allowable single dose (mg). "
                "Use this whenever a user asks about dosage, medication amounts, or safe drug quantities."
            ),
            args_schema=DosageInput,
        ),
        StructuredTool.from_function(
            func=_make_location_tool(extra_facilities),
            name="find_optimal_location",
            description=(
                "Finds the optimal GPS coordinates for a medical distribution center given the "
                "facility network. Uses demand-weighted centroid algorithm. Optionally accepts a "
                "list of facility names and a max service radius in km. "
                "Use this whenever a user asks where to place a distribution center or about "
                "coverage, logistics routing, or hospital proximity."
            ),
            args_schema=LocationInput,
        ),
        StructuredTool.from_function(
            func=_rag_tool,
            name="lookup_medical_logistics",
            description=(
                "Retrieves relevant passages from the medical logistics knowledge base. "
                "Use this to answer questions about cold chain, compliance, dosage guidelines, "
                "Florida hospital geography, emergency preparedness, and inventory management. "
                "Always call this BEFORE answering general medical logistics questions."
            ),
            args_schema=RAGInput,
        ),
    ]

    system_prompt = _build_system_prompt(extra_facilities)

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
    )

    return agent

def run_agent(
    user_input: str,
    chat_history: list | None = None,
    api_key: str | None = None,
    extra_facilities: list[dict] | None = None,
) -> dict:
    """
    Run the SNAP agent for one user turn.

    Parameters
    ----------
    user_input       : The latest message from the user.
    chat_history     : Previous turns as [{"role": "user"|"assistant", "content": "..."}].
    api_key          : Gemini API key.
    extra_facilities : Custom facilities from get_custom_facilities() in the Streamlit session.

    Returns
    -------
    {"output": str, "intermediate_steps": list[dict]}
    """
    agent = build_agent(api_key=api_key, extra_facilities=extra_facilities or [])

    messages = []
    for msg in (chat_history or []):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=user_input))

    result = agent.invoke({"messages": messages})

    all_messages = result.get("messages", [])
    output = ""
    intermediate_steps = []

    for msg in all_messages:
        msg_type = type(msg).__name__

        if msg_type == "AIMessage" and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                intermediate_steps.append({
                    "tool": tc.get("name", ""),
                    "tool_input": tc.get("args", {}),
                    "result": "",
                })

        elif msg_type == "ToolMessage":
            for step in reversed(intermediate_steps):
                if step["result"] == "":
                    step["result"] = msg.content
                    break

        elif msg_type == "AIMessage" and not getattr(msg, "tool_calls", None):
            if msg.content:
                output = msg.content

    return {
        "output": output,
        "intermediate_steps": intermediate_steps,
    }