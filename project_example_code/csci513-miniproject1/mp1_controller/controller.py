"""
Ultra‑Simple Adaptive Cruise Controller (minimal knobs)
------------------------------------------------------
Design goals:
- Keep it tiny and readable for a class project.
- Still reaches target speed band (±2 m/s) in ~6 s when NOT following.
- Sanity‑check FOLLOWING detection and basic safety overrides.

What’s simplified vs previous:
- No derivative term in following (P only).
- No cruise guard, no start‑up floors.
- TTC thresholds and near‑field constants are in‑code literals.
- One helper for desired gap, one for TTC.

Swap‑in ready: class name/signature preserved.
"""

from typing import Tuple, Optional
from mp1_simulator.simulator import Observation, Mode


class Controller:
    def __init__(self, distance_threshold: float):
        # Required state
        self.distance_threshold = distance_threshold
        self.prev_distance: Optional[float] = None
        self.last_mode: Optional[Mode] = None

        # Minimal timing/limits
        self.dt = 0.1
        self.a_limit = 10.0

        # Cruise (critically‑damped on speed with a single knob)
        self.wn = 3.0
        self.prev_e = 0.0
        self.cruise_time = 0.0

        # 6 s reach requirement when NOT following
        self.deadline_s = 6.0
        self.reach_tol = 2.0  # ±2 m/s band
        self.min_push = 6.5   # minimal push toward band

        # Following: time‑headway with minimal params (P only)
        self.headway = 1.2
        self.min_gap = 5.0
        self.Kp_gap = 0.5

    # --- helpers ---
    def _desired_gap(self, v_ego: float) -> float:
        return max(self.distance_threshold, self.min_gap, self.headway * max(v_ego, 0.0))

    def _steps_to_collision(self, d_now: Optional[float], d_prev: Optional[float]) -> float:
        if d_now is None or d_prev is None:
            return float("inf")
        dd = d_now - d_prev
        return float("inf") if dd >= 0 else d_now / max(-dd, 1e-6)

    # --- main ---
    def run_step(self, obs: Observation) -> Tuple[float, Mode]:
        v = obs.ego_velocity
        v_set = obs.desired_speed
        d = obs.distance_to_lead

        steps_ttc = self._steps_to_collision(d, self.prev_distance) if d is not None else float("inf")
        gap_des = self._desired_gap(v) if d is not None else None

        # FOLLOWING decision (minimal yet robust):
        # follow if within distance threshold/headway gap, or absolutely near, or short TTC
        following = False
        if d is not None:
            if gap_des is not None:
                following = (d <= max(self.distance_threshold, gap_des))
            following = following or (d < 16.0)            # absolute near gate
            following = following or (steps_ttc < 35.0)    # ~3.5 s at dt=0.1

        mode = Mode.FOLLOWING if following else Mode.CRUISING

        # Cruise timer for 6 s deadline
        if mode == Mode.CRUISING:
            self.cruise_time = 0.0 if self.last_mode != Mode.CRUISING else self.cruise_time + self.dt
        else:
            self.cruise_time = 0.0

        # --- base control laws ---
        if mode == Mode.CRUISING:
            # Critically‑damped acceleration toward set speed
            e = v_set - v
            edot = (e - self.prev_e) / max(self.dt, 1e-6)
            a_cmd = (self.wn ** 2) * e - (2.0 * self.wn) * edot

            # 6 s reach: add a minimal push until inside band
            t_left = max(0.1, self.deadline_s - self.cruise_time)
            if abs(e) > self.reach_tol:
                a_req = max((abs(e) - self.reach_tol) / t_left, self.min_push)
                a_cmd = (max(a_cmd, 0.0) + a_req) if e > 0 else (min(a_cmd, 0.0) - a_req)

            # Gentle asymmetry to avoid deep undershoot after small overshoot
            a_cmd = max(a_cmd, 0.0) if v < v_set else max(a_cmd, -1.8)
            a = max(min(a_cmd, self.a_limit), -self.a_limit)
            self.prev_e = e

        else:  # FOLLOWING
            gap = self._desired_gap(v)
            gap_err = (d or gap) - gap
            a_cmd = self.Kp_gap * gap_err

            # Don’t accelerate above set speed while still too close
            if (v > v_set) and (gap_err > 0):
                a_cmd = min(a_cmd, 0.0)

            # Simple proximity clamps
            if d is not None:
                if d < 0.9 * gap:
                    a_cmd = min(a_cmd, -3.0)
                if d < 0.7 * gap:
                    a_cmd = min(a_cmd, -5.0)

            a = max(min(a_cmd, self.a_limit), -self.a_limit)
            self.prev_e = 0.0

        # --- safety overrides (compact, literal constants) ---
        if d is not None:
            # 1) TTC hard brake
            if steps_ttc < 40.0:   # ~4.0 s at dt=0.1
                a = -10.0

            # 2) Stopping‑distance check (single assumed decel, small buffer)
            s_stop = (v ** 2) / max(2.0 * 8.0, 1e-6)  # assume 8 m/s² comfort decel
            safe_margin = max(d - (self.min_gap + 4.0), 0.0)  # 4 m buffer
            if s_stop > safe_margin:
                a = -10.0

            # 3) Near‑field
            if d < 5.0:
                a = -10.0

        # final clamp & bookkeeping
        a = max(min(a, self.a_limit), -self.a_limit)
        self.prev_distance = d
        self.last_mode = mode
        return a, mode
