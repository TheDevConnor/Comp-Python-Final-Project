import math
from typing import Optional

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0  # Earth's mean radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def _weighted_centroid(hospitals: list[dict]) -> tuple[float, float]:
    total_weight = sum(h["daily_doses_needed"] for h in hospitals)
    if total_weight == 0:
        raise ValueError("Total daily_doses_needed across all hospitals must be > 0.")

    lat_c = sum(h["lat"] * h["daily_doses_needed"] for h in hospitals) / total_weight
    lon_c = sum(h["lon"] * h["daily_doses_needed"] for h in hospitals) / total_weight
    return round(lat_c, 6), round(lon_c, 6)

def find_optimal_location(hospitals: list[dict], max_radius_km: Optional[float] = None) -> dict:
    """
      1. Computes a demand-weighted geographic centroid of all hospitals.
      2. Calculates the haversine distance from that centroid to every hospital.
      3. Computes a weighted average distance (demand-weighted) as the
         primary performance metric (lower = better coverage).
      4. Optionally checks whether every hospital lies within max_radius_km.
    """

    if not hospitals:
        raise ValueError("'hospitals' list must not be empty.")

    required_keys = {"name", "lat", "lon", "daily_doses_needed"}
    for i, h in enumerate(hospitals):
        missing = required_keys - h.keys()
        if missing:
            raise ValueError(f"Hospital at index {i} is missing keys: {missing}.")
        if not (-90 <= h["lat"] <= 90):
            raise ValueError(f"Hospital '{h['name']}': lat {h['lat']} is out of range [-90, 90].")
        if not (-180 <= h["lon"] <= 180):
            raise ValueError(f"Hospital '{h['name']}': lon {h['lon']} is out of range [-180, 180].")
        if not isinstance(h["daily_doses_needed"], (int, float)) or h["daily_doses_needed"] <= 0:
            raise ValueError(
                f"Hospital '{h['name']}': daily_doses_needed must be a positive number, "
                f"got {h['daily_doses_needed']}."
            )

    if max_radius_km is not None:
        if not isinstance(max_radius_km, (int, float)) or max_radius_km <= 0:
            raise ValueError(f"'max_radius_km' must be a positive number, got {max_radius_km}.")

    opt_lat, opt_lon = _weighted_centroid(hospitals)

    total_doses = sum(h["daily_doses_needed"] for h in hospitals)
    hospital_distances = []

    for h in hospitals:
        dist_km = _haversine_km(opt_lat, opt_lon, h["lat"], h["lon"])
        hospital_distances.append({
            "name": h["name"],
            "distance_km": round(dist_km, 2),
            "daily_doses_needed": h["daily_doses_needed"],
            "lat": h["lat"],
            "lon": h["lon"],
        })

    hospital_distances.sort(key=lambda x: x["distance_km"])

    weighted_avg = sum(
        d["distance_km"] * d["daily_doses_needed"] for d in hospital_distances
    ) / total_doses

    nearest = hospital_distances[0]
    farthest = hospital_distances[-1]

    outside_radius = []
    if max_radius_km is not None:
        outside_radius = [
            d["name"] for d in hospital_distances if d["distance_km"] > max_radius_km
        ]

    coverage_ok = len(outside_radius) == 0

    detail_lines = [
        f"Recommended distribution center: ({opt_lat}, {opt_lon}).",
        f"Covers {len(hospitals)} hospital(s) with a total of {int(total_doses)} daily doses.",
        f"Demand-weighted average distance: {weighted_avg:.2f} km.",
        f"Nearest  hospital: {nearest['name']} ({nearest['distance_km']:.2f} km).",
        f"Farthest hospital: {farthest['name']} ({farthest['distance_km']:.2f} km).",
    ]
    if max_radius_km is not None:
        if coverage_ok:
            detail_lines.append(
                f"All hospitals are within the {max_radius_km:.1f} km service radius. ✓"
            )
        else:
            detail_lines.append(
                f"WARNING: {len(outside_radius)} hospital(s) exceed the "
                f"{max_radius_km:.1f} km radius: {', '.join(outside_radius)}."
            )

    return {
        "optimal_lat": opt_lat,
        "optimal_lon": opt_lon,
        "weighted_avg_distance_km": round(weighted_avg, 2),
        "max_distance_km": farthest["distance_km"],
        "nearest_hospital": nearest["name"],
        "farthest_hospital": farthest["name"],
        "coverage_ok": coverage_ok,
        "hospitals_outside_radius": outside_radius,
        "hospital_distances": hospital_distances,
        "detail": " ".join(detail_lines),
    }
