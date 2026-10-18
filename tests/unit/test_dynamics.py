"""Unit tests for the pure model functions (paper Eqs. 4–11)."""

import math

import numpy as np
import pytest

from uavta.dynamics.energy import (
    consumption_rate,
    is_energy_feasible,
    residual_energy,
    time_to_depletion,
)
from uavta.dynamics.kinematics import advance_toward, clip_to_airspace, distance, travel_time
from uavta.dynamics.sensing import sensing_capability
from uavta.dynamics.survival import cumulative_survival, segment_survival
from uavta.dynamics.value import decay_factor, expected_collected_value


class TestKinematics:
    def test_distance_3d(self):
        assert distance([0, 0, 0], [3, 4, 12]) == pytest.approx(13.0)

    def test_travel_time(self):
        assert travel_time(1500.0, 15.0) == pytest.approx(100.0)

    @pytest.mark.parametrize("speed", [0.0, -1.0])
    def test_travel_time_rejects_nonpositive_speed(self, speed):
        with pytest.raises(ValueError):
            travel_time(10.0, speed)

    def test_advance_partial(self):
        pos, flown = advance_toward([0, 0, 0], [300, 400, 0], speed=10.0, dt=10.0)
        np.testing.assert_allclose(pos, [60, 80, 0])
        assert flown == pytest.approx(100.0)

    def test_advance_stops_at_destination(self):
        pos, flown = advance_toward([0, 0, 0], [30, 40, 0], speed=10.0, dt=100.0)
        np.testing.assert_allclose(pos, [30, 40, 0])
        assert flown == pytest.approx(50.0)

    def test_advance_does_not_mutate_input(self):
        start = np.array([1.0, 2.0, 3.0])
        advance_toward(start, [10, 10, 10], speed=1.0, dt=1.0)
        np.testing.assert_array_equal(start, [1.0, 2.0, 3.0])

    def test_partial_moves_compose(self):
        a, d1 = advance_toward([0, 0, 0], [1000, 0, 0], speed=15.0, dt=20.0)
        b, d2 = advance_toward(a, [1000, 0, 0], speed=15.0, dt=30.0)
        c, d3 = advance_toward([0, 0, 0], [1000, 0, 0], speed=15.0, dt=50.0)
        np.testing.assert_allclose(b, c)
        assert d1 + d2 == pytest.approx(d3)

    def test_clip(self):
        np.testing.assert_array_equal(
            clip_to_airspace([-5, 9000, 100], [8000, 8000, 200]), [0, 8000, 100]
        )


class TestEnergy:
    def test_type_a_rate(self):
        # Type A: 1800 s endurance (Table 3) -> half the battery after 900 s.
        rate = consumption_rate(1800.0)
        assert residual_energy(1.0, rate, 900.0) == pytest.approx(0.5)

    def test_feasibility_is_strict(self):
        rate = consumption_rate(1800.0)
        assert not is_energy_feasible(1.0, rate, 1800.0)  # exactly zero is not > 0 (Eq. 11)
        assert is_energy_feasible(1.0, rate, 1799.0)

    def test_reserve(self):
        rate = consumption_rate(1000.0)
        assert is_energy_feasible(0.5, rate, 300.0, reserve=0.1)
        assert not is_energy_feasible(0.5, rate, 400.0, reserve=0.1)

    def test_time_to_depletion(self):
        assert time_to_depletion(0.25, consumption_rate(2400.0)) == pytest.approx(600.0)

    def test_negative_duration_raises(self):
        with pytest.raises(ValueError):
            residual_energy(1.0, 0.001, -1.0)


class TestSensing:
    def test_clear_weather_single_sensor_is_perfect(self):
        assert sensing_capability([1, 0, 0], [1.0, 1.0, 1.0]) == pytest.approx(1.0)

    def test_no_working_sensor(self):
        assert sensing_capability([0, 0, 0], [0.9, 0.9, 0.9]) == pytest.approx(0.0)

    def test_paper_event7_weather_type_a(self):
        # Type A carries VL+IR; omega = (0.94, 0.58, 0.54) after Table 5 event 7.
        expected = 1 - (1 - 0.94) * (1 - 0.58)
        assert sensing_capability([1, 1, 0], [0.94, 0.58, 0.54]) == pytest.approx(expected)

    def test_paper_event7_weather_type_b(self):
        expected = 1 - (1 - 0.94) * (1 - 0.58) * (1 - 0.54)
        assert sensing_capability([1, 1, 1], [0.94, 0.58, 0.54]) == pytest.approx(expected)

    def test_sensor_damage_lowers_eta(self):
        omega = [0.75, 0.76, 0.91]
        assert sensing_capability([0, 1, 1], omega) < sensing_capability([1, 1, 1], omega)

    @pytest.mark.parametrize("omega", [[0.0, 1.0, 1.0], [1.1, 1.0, 1.0]])
    def test_rejects_weather_out_of_range(self, omega):
        with pytest.raises(ValueError):
            sensing_capability([1, 1, 1], omega)


class TestSurvival:
    def test_segment_matches_eq6(self):
        lam, xi, d = 0.0002, 0.9, 5000.0
        theta = 1 - math.exp(-lam * xi * d)
        assert segment_survival(d, xi, lam) == pytest.approx(1 - theta)

    def test_cumulative_recursion_eq7(self):
        lam, xi = 0.0002, 1.0
        expected = (1 - (1 - math.exp(-lam * 1000))) * (1 - (1 - math.exp(-lam * 2500)))
        assert cumulative_survival([1000, 2500], xi, lam) == pytest.approx(expected)

    def test_split_segments_compose(self):
        # Applying survival to partial segments equals the full-segment factor.
        lam, xi = 0.0002, 0.9
        assert cumulative_survival([400, 600], xi, lam) == pytest.approx(
            segment_survival(1000, xi, lam)
        )

    def test_initial_survival_carries_over(self):
        assert cumulative_survival([0.0], 1.0, 0.0002, initial=0.8) == pytest.approx(0.8)


class TestValue:
    def test_decay(self):
        assert decay_factor(600.0, 100.0, 0.001) == pytest.approx(math.exp(-0.5))

    def test_decay_rejects_arrival_before_reference(self):
        with pytest.raises(ValueError):
            decay_factor(50.0, 100.0, 0.001)

    def test_expected_value(self):
        assert expected_collected_value(80.0, 0.5, 0.9, 0.5) == pytest.approx(18.0)

    def test_expected_value_rejects_bad_probability(self):
        with pytest.raises(ValueError):
            expected_collected_value(80.0, 1.2, 0.9, 0.5)
