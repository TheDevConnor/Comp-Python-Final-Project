import sys
import os
import time
import concurrent.futures
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.compute_dosage import compute_dosage
from tools.find_optimal_location import find_optimal_location

@pytest.fixture
def sample_hospitals():
    return [
        {"name": "Hospital A", "lat": 25.79,  "lon": -80.21, "daily_doses_needed": 1200},
        {"name": "Hospital B", "lat": 26.12,  "lon": -80.14, "daily_doses_needed": 870},
        {"name": "Hospital C", "lat": 26.28,  "lon": -80.21, "daily_doses_needed": 680},
    ]


@pytest.fixture
def single_hospital():
    return [{"name": "Solo Hospital", "lat": 26.0, "lon": -80.2, "daily_doses_needed": 500}]

class TestComputeDosage:

    def test_normal_dose_not_capped(self):
        result = compute_dosage(weight_kg=50, drug_mg_per_kg=5, max_dose_mg=400)
        assert result["result"] == 250.0
        assert result["unit"] == "mg"
        assert "capped" not in result["detail"].lower() or "exceeded" not in result["detail"].lower()

    def test_dose_capped_at_max(self):
        result = compute_dosage(weight_kg=100, drug_mg_per_kg=10, max_dose_mg=500)
        assert result["result"] == 500.0
        assert "Capped" in result["detail"] or "capped" in result["detail"]

    def test_exact_at_max_not_capped(self):
        result = compute_dosage(weight_kg=50, drug_mg_per_kg=10, max_dose_mg=500)
        assert result["result"] == 500.0
        assert "exceeded" not in result["detail"].lower()

    def test_very_small_patient(self):
        result = compute_dosage(weight_kg=3, drug_mg_per_kg=25, max_dose_mg=500)
        assert result["result"] == 75.0

    def test_precision_rounding(self):
        result = compute_dosage(weight_kg=7.3, drug_mg_per_kg=0.1, max_dose_mg=15)
        assert isinstance(result["result"], float)
        assert len(str(result["result"]).split(".")[-1]) <= 4

    def test_raises_on_zero_weight(self):
        with pytest.raises(ValueError, match="weight_kg"):
            compute_dosage(weight_kg=0, drug_mg_per_kg=5, max_dose_mg=500)

    def test_raises_on_negative_weight(self):
        with pytest.raises(ValueError):
            compute_dosage(weight_kg=-10, drug_mg_per_kg=5, max_dose_mg=500)

    def test_raises_on_zero_max_dose(self):
        with pytest.raises(ValueError, match="max_dose_mg"):
            compute_dosage(weight_kg=70, drug_mg_per_kg=5, max_dose_mg=0)

    def test_raises_on_string_input(self):
        with pytest.raises(ValueError):
            compute_dosage(weight_kg="seventy", drug_mg_per_kg=5, max_dose_mg=500)

    def test_morphine_real_world(self):
        result = compute_dosage(weight_kg=70, drug_mg_per_kg=0.1, max_dose_mg=15)
        assert result["result"] == 7.0

    def test_morphine_large_patient_capped(self):
        result = compute_dosage(weight_kg=200, drug_mg_per_kg=0.1, max_dose_mg=15)
        assert result["result"] == 15.0

    def test_epinephrine_anaphylaxis(self):
        """Epinephrine: 0.01 mg/kg, max 1 mg — 60 kg patient."""
        result = compute_dosage(weight_kg=60, drug_mg_per_kg=0.01, max_dose_mg=1.0)
        assert result["result"] == 0.6

class TestFindOptimalLocation:

    def test_returns_required_keys(self, sample_hospitals):
        result = find_optimal_location(sample_hospitals)
        required = {
            "optimal_lat", "optimal_lon", "weighted_avg_distance_km",
            "max_distance_km", "nearest_hospital", "farthest_hospital",
            "coverage_ok", "hospitals_outside_radius", "hospital_distances", "detail",
        }
        assert required.issubset(result.keys())

    def test_centroid_within_bounding_box(self, sample_hospitals):
        result = find_optimal_location(sample_hospitals)
        lats = [h["lat"] for h in sample_hospitals]
        lons = [h["lon"] for h in sample_hospitals]
        assert min(lats) <= result["optimal_lat"] <= max(lats)
        assert min(lons) <= result["optimal_lon"] <= max(lons)

    def test_single_hospital_zero_distance(self, single_hospital):
        result = find_optimal_location(single_hospital)
        assert result["weighted_avg_distance_km"] == 0.0
        assert result["optimal_lat"] == single_hospital[0]["lat"]
        assert result["optimal_lon"] == single_hospital[0]["lon"]

    def test_all_within_generous_radius(self, sample_hospitals):
        result = find_optimal_location(sample_hospitals, max_radius_km=500)
        assert result["coverage_ok"] is True
        assert result["hospitals_outside_radius"] == []

    def test_tight_radius_triggers_flag(self, sample_hospitals):
        result = find_optimal_location(sample_hospitals, max_radius_km=1)
        assert result["coverage_ok"] is False
        assert len(result["hospitals_outside_radius"]) > 0

    def test_hospital_distances_sorted(self, sample_hospitals):
        result = find_optimal_location(sample_hospitals)
        dists = [d["distance_km"] for d in result["hospital_distances"]]
        assert dists == sorted(dists)

    def test_weighted_avg_le_max_distance(self, sample_hospitals):
        result = find_optimal_location(sample_hospitals)
        assert result["weighted_avg_distance_km"] <= result["max_distance_km"]

    def test_raises_on_empty_list(self):
        with pytest.raises(ValueError, match="empty"):
            find_optimal_location([])

    def test_raises_on_missing_key(self):
        with pytest.raises(ValueError, match="missing"):
            find_optimal_location([{"name": "X", "lat": 25.0, "lon": -80.0}])  # no daily_doses_needed

    def test_raises_on_invalid_lat(self):
        with pytest.raises(ValueError, match="lat"):
            find_optimal_location([{"name": "X", "lat": 999, "lon": -80.0, "daily_doses_needed": 100}])

    def test_raises_on_zero_doses(self):
        with pytest.raises(ValueError):
            find_optimal_location([{"name": "X", "lat": 25.0, "lon": -80.0, "daily_doses_needed": 0}])

    def test_raises_on_invalid_radius(self, sample_hospitals):
        with pytest.raises(ValueError, match="max_radius_km"):
            find_optimal_location(sample_hospitals, max_radius_km=-10)

    def test_high_demand_hospital_pulls_centroid(self):
        h_low  = {"name": "Low",  "lat": 25.0, "lon": -80.0, "daily_doses_needed": 100}
        h_high = {"name": "High", "lat": 27.0, "lon": -82.0, "daily_doses_needed": 1000}
        result = find_optimal_location([h_low, h_high])
        # Centroid should be much closer to h_high
        assert abs(result["optimal_lat"] - h_high["lat"]) < abs(result["optimal_lat"] - h_low["lat"])

class TestIntegration:

    def test_location_then_dosage_workflow(self, sample_hospitals):
        loc = find_optimal_location(sample_hospitals, max_radius_km=100)
        assert loc["optimal_lat"] is not None

        # Amoxicillin: 25 mg/kg, max 500 mg
        for h in sample_hospitals:
            for weight in [20, 50, 80]:
                dose = compute_dosage(weight, 25, 500)
                assert dose["result"] > 0
                assert dose["result"] <= 500

    def test_all_formulary_drugs_for_reference_patient(self):
        from data.hospitals import DRUG_FORMULARY
        for drug in DRUG_FORMULARY:
            result = compute_dosage(70, drug["mg_per_kg"], drug["max_dose_mg"])
            assert 0 < result["result"] <= drug["max_dose_mg"]

class TestLoad:
    def test_dosage_100_concurrent_requests(self):
        import random

        def task(i):
            weight = 10 + (i % 90)          # 10–99 kg
            mg_per_kg = 0.01 + (i % 30)     # 0.01–30
            max_dose = 50 + (i % 1000)      # 50–1049 mg
            return compute_dosage(weight, mg_per_kg, max_dose)

        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
            futures = [ex.submit(task, i) for i in range(100)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        elapsed = time.time() - start

        assert len(results) == 100
        assert all(r["result"] > 0 for r in results)
        assert elapsed < 5.0, f"Load test too slow: {elapsed:.2f}s (limit 5s)"

    def test_location_50_concurrent_requests(self, sample_hospitals):
        def task(_):
            return find_optimal_location(sample_hospitals)

        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            futures = [ex.submit(task, i) for i in range(50)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        elapsed = time.time() - start

        assert len(results) == 50
        lats = {r["optimal_lat"] for r in results}
        lons = {r["optimal_lon"] for r in results}
        assert len(lats) == 1, "Non-deterministic centroid across concurrent calls"
        assert len(lons) == 1
        assert elapsed < 5.0, f"Load test too slow: {elapsed:.2f}s (limit 5s)"
