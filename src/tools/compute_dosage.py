def compute_dosage(weight_kg: float, drug_mg_per_kg: float, max_dose_mg: float) -> dict:
    """
    Compute the safe dose of a drug for a patient given their weight.

    The calculated dose is weight_kg * drug_mg_per_kg, capped to
    max_dose_mg so the result never exceeds the clinically approved
    maximum regardless of patient size.

    Args:
        weight_kg (float): Patient weight in kilograms. Must be > 0.
        drug_mg_per_kg (float): Prescribed dose in mg per kg of body
            weight. Must be > 0.
        max_dose_mg (float): Maximum allowable single dose in mg.
            Must be > 0.

    Returns:
        dict: {
            "result"  : float  - safe dose in mg (capped at max_dose_mg),
            "unit"    : str    - always "mg",
            "detail"  : str    - human-readable explanation including
                                 whether the dose was capped.
        }

    Raises:
        ValueError: if any argument is non-positive or not a real number.
    """

    for name, value in [
        ("weight_kg", weight_kg),
        ("drug_mg_per_kg", drug_mg_per_kg),
        ("max_dose_mg", max_dose_mg),
    ]:
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"'{name}' must be a numeric value, got {type(value).__name__}."
            )
        if value <= 0:
            raise ValueError(
                f"'{name}' must be greater than 0, got {value}."
            )

    calculated_dose = weight_kg * drug_mg_per_kg
    safe_dose = min(calculated_dose, max_dose_mg)
    capped = calculated_dose > max_dose_mg

    if capped:
        detail = (
            f"Calculated dose ({calculated_dose:.2f} mg) exceeded the "
            f"maximum allowable dose. Dose capped at {max_dose_mg:.2f} mg."
        )
    else:
        detail = (
            f"Calculated dose: {weight_kg} kg x {drug_mg_per_kg} mg/kg "
            f"= {calculated_dose:.2f} mg. Within the {max_dose_mg:.2f} mg limit."
        )

    return {
        "result": round(safe_dose, 4),
        "unit": "mg",
        "detail": detail,
    }
