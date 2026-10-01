"""Adaptive BER working-point search with censoring.

All estimators of a configuration are measured on the same slots, so one SNR
evaluation yields a BER for every estimator. The search

1. starts at a predicted SNR;
2. brackets: while some estimator has every measured BER above the target it
   steps up, while some estimator has every BER at or below the target it steps
   down (``coarse_db``, 1.5x when the BER is far away);
3. refines: halves the widest bracket that is wider than ``fine_db``;
4. stops when every estimator is bracketed to ``fine_db``, censored, or the
   point budget is spent.

Per estimator the result is one of

``ok``              BER crosses the target between two evaluated SNRs
                    ``wp_lower_db`` < WP <= ``wp_upper_db``; ``wp_db`` is the
                    log-linear interpolation inside that bracket.
``right_censored``  BER above the target at every SNR up to ``wp_lower_db``:
                    WP > ``wp_lower_db`` (``reason``: ``hi_limit``, ``floor``,
                    ``excess_loss`` or ``budget``).
``left_censored``   BER at or below the target already at ``wp_upper_db``:
                    WP <= ``wp_upper_db`` (``reason``: ``lo_limit`` or
                    ``budget``).

``(wp_lower_db, wp_upper_db)`` with ``None`` as an open end is the interval
label of an interval-censored regression (e.g. XGBoost survival:aft).

The WP is where the BER *stays* below the target: the bracket is taken above
the highest SNR whose BER exceeds the target. A lower SNR with BER below the
target (noise) sets ``nonmonotone``.
"""

from __future__ import annotations

import math
from typing import Callable

FLOOR_BER = 0.05        # tail rule: only below this BER ...
FLOOR_RATIO_4DB = 1.5   # ... the BER must drop by at least this factor per 4 dB
FLAT_SPAN_DB = 6.0      # flat rule: over the top three points spanning this ...
FLAT_RATIO = 0.7        # ... the BER keeps at least this fraction, while the
                        # reference estimator is already below the target
FLAT_MAX_BER = 0.25     # flat rule only below this BER (not the ~0.5 guessing plateau)


def _ber_for_log(stat: dict) -> float:
    """BER clipped to half an error, so a zero-error point has a finite log."""
    return max(float(stat["ber"]), 0.5 / max(int(stat.get("num_bits", 0)), 1))


def classify(curve: list[tuple[float, dict]], target: float) -> dict:
    """Position of the target in one estimator's curve (sorted by SNR)."""
    snrs = [s for s, _ in curve]
    bers = [float(st["ber"]) for _, st in curve]
    above = [s for s, b in zip(snrs, bers) if b > target]
    if not above:
        return {"kind": "all_below"}
    a = max(above)
    ia = snrs.index(a)
    nonmono = any(b <= target for b in bers[:ia])
    if ia == len(snrs) - 1:
        return {"kind": "all_above", "nonmonotone": nonmono}
    return {"kind": "bracket", "a": a, "b": snrs[ia + 1], "nonmonotone": nonmono}


def is_floor(curve: list[tuple[float, dict]], target: float, ref_below: bool = False) -> bool:
    """Error floor above the target.

    Tail rule: the top two points are above the target, below ``FLOOR_BER``, and
    the BER falls by less than ``FLOOR_RATIO_4DB`` per 4 dB.
    Flat rule (needs ``ref_below``: the reference estimator is at or below the
    target at the top SNR, so this is not the low-SNR plateau): over the top
    three points spanning ``FLAT_SPAN_DB`` the BER stays below ``FLAT_MAX_BER``
    and keeps ``FLAT_RATIO`` of its value, or rises.
    """
    if len(curve) < 2:
        return False
    (s1, st1), (s2, st2) = curve[-2], curve[-1]
    b1, b2 = float(st1["ber"]), float(st2["ber"])
    if b1 > target and b2 > target and b2 < FLOOR_BER and s2 > s1:
        drop_per_4db = (b1 / b2) ** (4.0 / (s2 - s1)) if b2 > 0 else math.inf
        if drop_per_4db < FLOOR_RATIO_4DB:
            return True
    if ref_below and len(curve) >= 3:
        (s0, st0), (s3, st3) = curve[-3], curve[-1]
        b0, b3 = float(st0["ber"]), float(st3["ber"])
        if (s3 - s0 >= FLAT_SPAN_DB and b0 > target and b3 > target
                and b3 < FLAT_MAX_BER and b3 >= FLAT_RATIO * b0):
            return True
    return False


def interpolate_wp(a: float, st_a: dict, b: float, st_b: dict, target: float) -> float:
    la, lb = math.log10(_ber_for_log(st_a)), math.log10(_ber_for_log(st_b))
    lt = math.log10(target)
    if la == lb:
        return b
    t = min(max((lt - la) / (lb - la), 0.0), 1.0)
    return a + t * (b - a)


def search_working_point(
    measure: Callable[[float], dict[str, dict]],
    names: list[str],
    *,
    lo: float,
    hi: float,
    start: float,
    target: float = 1e-2,
    coarse_db: float = 4.0,
    fine_db: float = 2.0,
    max_points: int = 10,
    grid_db: float = 0.5,
    reference: str | None = "perfect",
    max_excess_db: float = 15.0,
) -> dict:
    """Run the search; ``measure(snr)`` returns ``{name: ErrorStats.as_dict()}``.

    ``reference`` (perfect CSI) anchors two stopping rules for the other
    estimators: the flat-floor rule of :func:`is_floor`, and no climbing beyond
    ``max_excess_db`` above the reference's working-point bracket.
    """
    points: dict[float, dict[str, dict]] = {}
    if reference not in names:
        reference = None

    def quantize(s: float) -> float:
        return float(min(max(round(s / grid_db) * grid_db, lo), hi))

    def evaluate(s: float) -> bool:
        s = quantize(s)
        if s in points:
            return False
        points[s] = measure(s)
        return True

    def curves() -> dict[str, list[tuple[float, dict]]]:
        return {n: [(s, points[s][n]) for s in sorted(points)] for n in names}

    def ref_state(cv, cls, smax):
        """(reference below target at smax, cap SNR for climbing)."""
        if reference is None:
            return False, math.inf
        below = float(points[smax][reference]["ber"]) <= target
        c = cls[reference]
        if c["kind"] == "bracket":
            return below, c["b"] + max_excess_db
        if c["kind"] == "all_below":
            return below, cv[reference][0][0] + max_excess_db
        return below, math.inf

    def stop_climbing(n, cv, cls, smax):
        below, cap = ref_state(cv, cls, smax)
        if n == reference:
            return is_floor(cv[n], target)
        return is_floor(cv[n], target, ref_below=below) or smax >= cap

    evaluate(start)
    while len(points) < max_points:
        cv = curves()
        cls = {n: classify(cv[n], target) for n in names}
        smax, smin = max(points), min(points)
        up = [n for n in names
              if cls[n]["kind"] == "all_above" and smax < hi and not stop_climbing(n, cv, cls, smax)]
        down = [n for n in names if cls[n]["kind"] == "all_below" and smin > lo]
        if up:
            closest = min(float(cv[n][-1][1]["ber"]) for n in up)
            if evaluate(smax + coarse_db * (1.5 if closest > 0.2 else 1.0)):
                continue
        if down:
            closest = max(float(cv[n][0][1]["ber"]) for n in down)
            if evaluate(smin - coarse_db * (1.5 if closest == 0.0 else 1.0)):
                continue
        gaps = sorted(
            {(c["a"], c["b"]) for c in cls.values()
             if c["kind"] == "bracket" and c["b"] - c["a"] > fine_db + 1e-9},
            key=lambda ab: ab[0] - ab[1],
        )
        if not any(evaluate(0.5 * (a + b)) for a, b in gaps):
            break

    cv = curves()
    cls_all = {n: classify(cv[n], target) for n in names}
    smax, smin = max(points), min(points)
    ref_below, cap = ref_state(cv, cls_all, smax)
    results = {}
    for n in names:
        curve = cv[n]
        c = classify(curve, target)
        rec = {
            "snr_db": [s for s, _ in curve],
            "ber": [st["ber"] for _, st in curve],
            "bler": [st.get("bler") for _, st in curve],
            "bit_errors": [st.get("bit_errors") for _, st in curve],
            "num_bits": [st.get("num_bits") for _, st in curve],
            "block_errors": [st.get("block_errors") for _, st in curve],
            "num_blocks": [st.get("num_blocks") for _, st in curve],
            "nonmonotone": bool(c.get("nonmonotone", False)),
        }
        if c["kind"] == "bracket":
            st = dict(curve)
            rec.update(status="ok", reason=None,
                       wp_db=interpolate_wp(c["a"], st[c["a"]], c["b"], st[c["b"]], target),
                       wp_lower_db=c["a"], wp_upper_db=c["b"])
        elif c["kind"] == "all_above":
            if smax >= hi:
                reason = "hi_limit"
            elif is_floor(curve, target, ref_below=ref_below and n != reference):
                reason = "floor"
            elif n != reference and smax >= cap:
                reason = "excess_loss"
            else:
                reason = "budget"
            rec.update(status="right_censored", reason=reason, wp_db=None,
                       wp_lower_db=curve[-1][0], wp_upper_db=None)
        else:
            reason = "lo_limit" if smin <= lo else "budget"
            rec.update(status="left_censored", reason=reason, wp_db=None,
                       wp_lower_db=None, wp_upper_db=curve[0][0])
        results[n] = rec
    return {"num_points": len(points), "snr_db": sorted(points), "estimators": results}
