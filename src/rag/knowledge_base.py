import os
import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "medical_logistics"

DOCUMENTS = [
    {
        "id": "dist_001",
        "text": (
            "A medical distribution center should be located to minimize the demand-weighted "
            "average travel distance to all hospitals it serves. This ensures that facilities "
            "with higher patient volumes receive priority in proximity, reducing delivery latency "
            "for high-consumption medications."
        ),
        "metadata": {"category": "logistics", "topic": "center_placement"},
    },
    {
        "id": "dist_002",
        "text": (
            "Cold-chain medications such as insulin, vaccines, and biologics must be maintained "
            "between 2°C and 8°C throughout the supply chain. Distribution centers storing "
            "cold-chain items require refrigerated warehousing, temperature-monitored vehicles, "
            "and must prioritize shorter delivery routes to minimize temperature excursion risk."
        ),
        "metadata": {"category": "logistics", "topic": "cold_chain"},
    },
    {
        "id": "dist_003",
        "text": (
            "Par level management ensures that each hospital's on-hand drug inventory stays "
            "between a minimum and maximum threshold. The distribution center replenishes stock "
            "when levels drop below the minimum par. Daily doses needed per hospital drives the "
            "reorder point calculation."
        ),
        "metadata": {"category": "logistics", "topic": "inventory"},
    },
    {
        "id": "dist_004",
        "text": (
            "The Joint Commission requires that medication storage areas in hospitals maintain "
            "proper temperature, light, and humidity conditions. Distribution centers are "
            "responsible for validating proper packaging before dispatch to ensure regulatory "
            "compliance upon receipt."
        ),
        "metadata": {"category": "compliance", "topic": "joint_commission"},
    },
    {
        "id": "dose_001",
        "text": (
            "Pediatric dosing is almost always weight-based (mg/kg). The prescribing clinician "
            "must confirm the patient's current weight in kilograms and apply the appropriate "
            "mg/kg factor for the specific drug. The calculated dose must never exceed the "
            "established maximum single dose to prevent toxicity."
        ),
        "metadata": {"category": "dosage", "topic": "pediatric"},
    },
    {
        "id": "dose_002",
        "text": (
            "Amoxicillin is typically dosed at 25 mg/kg for pediatric patients, with a maximum "
            "single dose of 500 mg. It is used for bacterial infections including otitis media, "
            "strep throat, and community-acquired pneumonia. Administered orally every 8–12 hours."
        ),
        "metadata": {"category": "dosage", "topic": "amoxicillin"},
    },
    {
        "id": "dose_003",
        "text": (
            "Morphine sulfate for acute pain management is dosed at 0.1 mg/kg IV/IM, with a "
            "maximum single dose of 15 mg. Patients must be monitored for respiratory depression. "
            "Naloxone should be immediately available as a reversal agent during administration."
        ),
        "metadata": {"category": "dosage", "topic": "morphine"},
    },
    {
        "id": "dose_004",
        "text": (
            "Ibuprofen is dosed at 10 mg/kg per dose in pediatric patients, maximum 400 mg. "
            "It should be administered with food or milk to reduce gastrointestinal irritation. "
            "Contraindicated in patients with renal impairment, active GI bleed, or known NSAID allergy."
        ),
        "metadata": {"category": "dosage", "topic": "ibuprofen"},
    },
    {
        "id": "dose_005",
        "text": (
            "Vancomycin for serious gram-positive infections is dosed at 15 mg/kg IV every 6–12 hours "
            "in adults, with a maximum single dose of 2000 mg. Trough levels must be monitored to "
            "maintain therapeutic concentrations and avoid nephrotoxicity."
        ),
        "metadata": {"category": "dosage", "topic": "vancomycin"},
    },
    {
        "id": "dose_006",
        "text": (
            "Epinephrine for anaphylaxis is dosed at 0.01 mg/kg IM into the anterolateral thigh, "
            "with a maximum single dose of 1 mg (1:1000 concentration). It is the first-line "
            "treatment for anaphylactic shock and should be administered immediately upon diagnosis."
        ),
        "metadata": {"category": "dosage", "topic": "epinephrine"},
    },
    {
        "id": "fl_001",
        "text": (
            "South Florida's hospital network is geographically spread across Miami-Dade, Broward, "
            "and Palm Beach counties. A centrally located distribution hub near the Broward–Miami-Dade "
            "county line (approximately 26.0°N, 80.18°W) minimizes average delivery distance across "
            "the tri-county area while staying accessible via I-95 and Florida's Turnpike."
        ),
        "metadata": {"category": "logistics", "topic": "florida_geography"},
    },
    {
        "id": "fl_002",
        "text": (
            "Florida's hurricane season (June–November) poses a significant risk to medical supply "
            "chains. Distribution centers in South Florida should maintain a 72-hour emergency "
            "stockpile for critical medications. Backup generator power and flood-resistant "
            "facilities are required under Florida Department of Health emergency preparedness standards."
        ),
        "metadata": {"category": "compliance", "topic": "emergency_preparedness"},
    },
]


def build_or_load_vectorstore():
    """
    Initialize ChromaDB and return the collection.
    If the collection already exists, it reloads it.
    If not, it creates it and populates it with DOCUMENTS.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = embedding_functions.DefaultEmbeddingFunction()

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        collection = client.get_collection(
            name=COLLECTION_NAME, embedding_function=ef
        )
        return collection

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[d["id"] for d in DOCUMENTS],
        documents=[d["text"] for d in DOCUMENTS],
        metadatas=[d["metadata"] for d in DOCUMENTS],
    )
    return collection


def query_knowledge(question: str, n_results: int = 3) -> list[str]:
    collection = build_or_load_vectorstore()
    results = collection.query(query_texts=[question], n_results=n_results)
    return results["documents"][0] if results["documents"] else []
