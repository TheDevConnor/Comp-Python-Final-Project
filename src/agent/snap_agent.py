import json
import os
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import StructuredTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from tools.compute_dosage import compute_dosage
from tools.find_optimal_location import find_optimal_location
from rag.knowledge_base import query_knowledge
from data.hospitals import HOSPITALS

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
        hospitals = HOSPITALS
        if hospital_names:
            hospitals = [h for h in HOSPITALS if h["name"] in hospital_names]
            if not hospitals:
                return "Error: None of the specified hospital names were found in the database."
        result = find_optimal_location(hospitals, max_radius_km=max_radius_km)
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
specialized in medical supply chain optimization for South Florida hospitals.

You have access to three tools:
1. **compute_dosage** — Calculates safe medication doses based on patient weight.
2. **find_optimal_location** — Finds the optimal GPS location for a medical distribution center.
3. **lookup_medical_logistics** — Retrieves domain knowledge about regulations, cold chain, and guidelines.

RULES:
- Always use tools for calculations. Never guess dosages or distances.
- For general medical logistics questions, use lookup_medical_logistics first.
- Be precise and professional. You serve healthcare workers and operations teams.
- When reporting dosage results, always include the detail field explaining whether the dose was capped.
- When reporting location results, always report the weighted average distance and flag any hospitals outside radius.
- If the user's question involves both dosage and location (e.g., "how much stock do we need?"), call both tools.
- Format numerical results clearly with units.
"""

def build_agent(api_key: Optional[str] = None) -> AgentExecutor:
    """
    Build and return the SNAP LangChain agent executor.

    Args:
        api_key: Google Gemini API key. If None, reads GOOGLE_API_KEY from env.

    Returns:
        An AgentExecutor ready to invoke with {"input": "...", "chat_history": [...]}.
    """
    key = api_key or os.environ.get("GOOGLE_API_KEY", "")

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=key,
        temperature=0.2,
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, LANGCHAIN_TOOLS, prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=LANGCHAIN_TOOLS,
        verbose=True,
        max_iterations=6,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )

    return executor

def run_agent(user_input: str, chat_history: list = None, api_key: str = None) -> dict:
    """
    Run the agent for a single user message.

    Returns:
        dict with keys:
            "output"              : str  - Final agent response.
            "intermediate_steps"  : list - Tool calls made (name + result).
    """
    executor = build_agent(api_key=api_key)
    response = executor.invoke({
        "input": user_input,
        "chat_history": chat_history or [],
    })

    steps = []
    for action, observation in response.get("intermediate_steps", []):
        steps.append({
            "tool": action.tool,
            "tool_input": action.tool_input,
            "result": observation,
        })

    return {
        "output": response["output"],
        "intermediate_steps": steps,
    }
