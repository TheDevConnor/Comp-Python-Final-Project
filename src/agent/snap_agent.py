import json
import os
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
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
            "Optional list of hospital names to include. "
            "If omitted, ALL hospitals in the system are used."
        ),
    )
    max_radius_km: Optional[float] = Field(
        default=None,
        description="Optional maximum service radius in km. Flags any hospital outside this range.",
    )


class RAGInput(BaseModel):
    question: str = Field(..., description="Medical logistics question to look up in the knowledge base.")


def _dosage_tool(weight_kg: float, drug_mg_per_kg: float, max_dose_mg: float) -> str:
    try:
        result = compute_dosage(weight_kg, drug_mg_per_kg, max_dose_mg)
        return json.dumps(result)
    except ValueError as e:
        return f"Error: {e}"


def _location_tool(
    hospital_names: Optional[list[str]] = None,
    max_radius_km: Optional[float] = None,
) -> str:
    try:
        facilities = ALL_FACILITIES
        if hospital_names:
            facilities = [f for f in ALL_FACILITIES if f["name"] in hospital_names]
            if not facilities:
                return "Error: None of the specified facility names were found in the database."
        result = find_optimal_location(facilities, max_radius_km=max_radius_km)
        return json.dumps(result)
    except ValueError as e:
        return f"Error: {e}"


def _rag_tool(question: str) -> str:
    passages = query_knowledge(question, n_results=3)
    if not passages:
        return "No relevant information found in the knowledge base."
    return "\n\n---\n\n".join(passages)


LANGCHAIN_TOOLS = [
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
        func=_location_tool,
        name="find_optimal_location",
        description=(
            "Finds the optimal GPS coordinates for a medical distribution center given the "
            "hospital network. Uses demand-weighted centroid algorithm. Optionally accepts a "
            "list of hospital names and a max service radius in km. "
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

SYSTEM_PROMPT = """You are SNAP (Systems for Niche Agentic Programming), an AI logistics agent 
specialized in medical supply chain optimization for South Florida hospitals and retail pharmacies.

You have access to three tools:
1. **compute_dosage** — Calculates safe medication doses based on patient weight.
2. **find_optimal_location** — Finds the optimal GPS location for a medical distribution center.
3. **lookup_medical_logistics** — Retrieves domain knowledge about regulations, cold chain, and guidelines.

The facility network includes 8 hospitals (Jackson Memorial, Baptist Health, Broward Health, etc.)
and 15 retail pharmacies: 5 Walgreens, 5 CVS, and 5 Walmart Pharmacy locations across Miami-Dade,
Broward, and Palm Beach counties.

RULES:
- Always use tools for calculations. Never guess dosages or distances.
- For general medical logistics questions, use lookup_medical_logistics first.
- Be precise and professional. You serve healthcare workers and operations teams.
- When reporting dosage results, always include the detail field explaining whether the dose was capped.
- When reporting location results, always report the weighted average distance and flag any facilities outside radius.
- If the user's question involves both dosage and location (e.g., "how much stock do we need?"), call both tools.
- Format numerical results clearly with units.
- When asked about pharmacies specifically, note that Walmart locations have the highest daily dose volume among retail pharmacies.
"""


def build_agent(api_key: Optional[str] = None):
    key = api_key or os.environ.get("GOOGLE_API_KEY", "")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=key,
        temperature=0.2,
    )

    agent = create_agent(
        model=llm,
        tools=LANGCHAIN_TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )

    return agent


def run_agent(user_input: str, chat_history: list = None, api_key: str = None) -> dict:
    agent = build_agent(api_key=api_key)

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

        if msg_type == "AIMessage" and hasattr(msg, "tool_calls") and msg.tool_calls:
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

        elif msg_type == "AIMessage" and (not hasattr(msg, "tool_calls") or not msg.tool_calls):
            if msg.content:
                output = msg.content

    return {
        "output": output,
        "intermediate_steps": intermediate_steps,
    }
