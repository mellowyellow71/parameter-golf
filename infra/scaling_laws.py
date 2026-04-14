#!/usr/bin/env python3
"""
Scaling Law Regression for Parameter Golf.

Predicts final BPB from early-stage measurements using calibrated linear regression.
Provides go/no-go decision functions for the funnel pipeline.

Based on meta-analysis findings (abay.tech/posts/pgolf-meta, 975 runs):
  - Step 1000 BPB correlates r=0.86 with final BPB
  - Seed variance: 0.5 mBPB median
  - Second half of training has near-zero predictive value

Usage:
    from scaling_laws import ScalingPredictor
    sp = ScalingPredictor.load()
    decision = sp.decide_qualify(step_1000_bpb=1.12)
    # decision.action in ("promote", "kill", "marginal")
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


STATE_PATH = Path("infra/scaling_laws_state.json")

# --- Prior from meta-analysis (used before calibration data is available) ---
# r=0.86 implies slope ~0.86 if both distributions have similar variance.
# The intercept captures the improvement from training beyond step 1000.
META_PRIOR_SLOPE = 0.90
META_PRIOR_INTERCEPT = 0.05  # ~50 mBPB improvement from remaining training
META_PRIOR_RESIDUAL_STD = 0.008  # 8 mBPB


@dataclass
class CalibrationPoint:
    """A paired observation of (step_1000_bpb, final_bpb)."""
    experiment: str
    seed: int
    step_1000_bpb: float
    final_bpb: float


@dataclass
class RegressionFit:
    """Linear regression: final_bpb = intercept + slope * step_1000_bpb."""
    intercept: float
    slope: float
    r_squared: float
    residual_std: float
    n_points: int
    calibrated: bool  # True if fit from real data, False if using prior


@dataclass
class QualifyDecision:
    """Go/no-go decision from qualify stage."""
    action: str  # "promote", "kill", "marginal"
    predicted_final_bpb: float
    prediction_low: float   # 95% CI lower bound
    prediction_high: float  # 95% CI upper bound
    reason: str


class ScalingPredictor:
    """Predicts final BPB from step-1000 measurements."""

    def __init__(self):
        self.calibration_points: list[CalibrationPoint] = []
        self.fit: RegressionFit = RegressionFit(
            intercept=META_PRIOR_INTERCEPT,
            slope=META_PRIOR_SLOPE,
            r_squared=0.74,  # r=0.86 squared
            residual_std=META_PRIOR_RESIDUAL_STD,
            n_points=0,
            calibrated=False,
        )
        self.best_known_final_bpb: float = 1.0810  # current SOTA

    def add_calibration_point(self, experiment: str, seed: int,
                              step_1000_bpb: float, final_bpb: float) -> None:
        """Add a paired observation and refit."""
        self.calibration_points.append(CalibrationPoint(
            experiment=experiment, seed=seed,
            step_1000_bpb=step_1000_bpb, final_bpb=final_bpb,
        ))
        if final_bpb < self.best_known_final_bpb:
            self.best_known_final_bpb = final_bpb
        self._refit()

    def _refit(self) -> None:
        """Fit linear regression on calibration data."""
        n = len(self.calibration_points)
        if n < 3:
            # Not enough data for reliable fit, keep prior
            self.fit = RegressionFit(
                intercept=META_PRIOR_INTERCEPT,
                slope=META_PRIOR_SLOPE,
                r_squared=0.74,
                residual_std=META_PRIOR_RESIDUAL_STD,
                n_points=n,
                calibrated=False,
            )
            return

        xs = [p.step_1000_bpb for p in self.calibration_points]
        ys = [p.final_bpb for p in self.calibration_points]

        x_mean = sum(xs) / n
        y_mean = sum(ys) / n

        ss_xx = sum((x - x_mean) ** 2 for x in xs)
        ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        ss_yy = sum((y - y_mean) ** 2 for y in ys)

        if ss_xx < 1e-12:
            return  # All x values identical, can't fit

        slope = ss_xy / ss_xx
        intercept = y_mean - slope * x_mean

        # R-squared
        ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
        r_squared = 1.0 - ss_res / ss_yy if ss_yy > 1e-12 else 0.0

        # Residual standard deviation
        residual_std = math.sqrt(ss_res / (n - 2)) if n > 2 else META_PRIOR_RESIDUAL_STD

        self.fit = RegressionFit(
            intercept=intercept,
            slope=slope,
            r_squared=r_squared,
            residual_std=residual_std,
            n_points=n,
            calibrated=True,
        )

    def predict(self, step_1000_bpb: float) -> tuple[float, float, float]:
        """Predict final BPB with 95% prediction interval.

        Returns (predicted, lower_95, upper_95).
        """
        predicted = self.fit.intercept + self.fit.slope * step_1000_bpb
        margin = 1.96 * self.fit.residual_std
        return predicted, predicted - margin, predicted + margin

    def decide_qualify(self, step_1000_bpb: float,
                       novel_technique: bool = False) -> QualifyDecision:
        """Make promote/kill/marginal decision for a qualify-stage result.

        Args:
            step_1000_bpb: Measured val_bpb at step ~1000
            novel_technique: If True, widen the marginal band (don't kill promising novel work)
        """
        predicted, low, high = self.predict(step_1000_bpb)
        best = self.best_known_final_bpb

        # Thresholds (relative to best known final BPB)
        promote_threshold = best * 1.02   # Within 2% of best
        kill_threshold = best * 1.05      # 5% worse than best
        immediate_promote = best * 0.98   # 2% better than best

        if novel_technique:
            kill_threshold = best * 1.08  # More lenient for novel approaches

        if predicted <= immediate_promote:
            return QualifyDecision(
                action="promote",
                predicted_final_bpb=predicted,
                prediction_low=low,
                prediction_high=high,
                reason=f"Predicted {predicted:.4f} beats best {best:.4f} by "
                       f"{(best - predicted)*1000:.1f} mBPB. IMMEDIATE PROMOTE.",
            )
        elif predicted <= promote_threshold:
            return QualifyDecision(
                action="promote",
                predicted_final_bpb=predicted,
                prediction_low=low,
                prediction_high=high,
                reason=f"Predicted {predicted:.4f} within 2% of best {best:.4f}. Promote to full.",
            )
        elif predicted > kill_threshold:
            return QualifyDecision(
                action="kill",
                predicted_final_bpb=predicted,
                prediction_low=low,
                prediction_high=high,
                reason=f"Predicted {predicted:.4f} is {(predicted - best)*1000:.1f} mBPB worse "
                       f"than best {best:.4f}. Kill.",
            )
        else:
            return QualifyDecision(
                action="marginal",
                predicted_final_bpb=predicted,
                prediction_low=low,
                prediction_high=high,
                reason=f"Predicted {predicted:.4f} is between promote ({promote_threshold:.4f}) "
                       f"and kill ({kill_threshold:.4f}). Marginal — promote only if novel.",
            )

    def decide_smoke(self, train_loss_200: float) -> tuple[str, str]:
        """Make pass/fail decision for smoke test.

        Returns (action, reason) where action is "pass", "fail", or "marginal".
        """
        if train_loss_200 > 2.60:
            return "fail", f"train_loss_200={train_loss_200:.4f} > 2.60 threshold. Fundamental architecture problem."
        elif train_loss_200 > 2.55:
            return "marginal", f"train_loss_200={train_loss_200:.4f} in marginal range [2.55, 2.60]. Promote only if novel."
        else:
            return "pass", f"train_loss_200={train_loss_200:.4f} < 2.55. Pass."

    def cost_per_mbpb(self, experiments: list[dict]) -> float:
        """Calculate marginal cost per mBPB improvement across experiments.

        Each experiment dict should have 'cost' and 'delta_bpb' keys.
        """
        total_cost = sum(e.get("cost", 0) for e in experiments)
        total_delta = sum(e.get("delta_bpb", 0) for e in experiments)
        if total_delta == 0:
            return float("inf")
        return total_cost / (abs(total_delta) * 1000)

    def summary(self) -> str:
        """Human-readable summary of the predictor state."""
        lines = [
            "=== Scaling Law Predictor ===",
            f"Calibration points: {self.fit.n_points}",
            f"Calibrated: {self.fit.calibrated}",
            f"Regression: final_bpb = {self.fit.intercept:.4f} + {self.fit.slope:.4f} * step_1000_bpb",
            f"R²: {self.fit.r_squared:.3f}",
            f"Residual std: {self.fit.residual_std*1000:.1f} mBPB",
            f"Best known final BPB: {self.best_known_final_bpb:.4f}",
        ]
        if self.calibration_points:
            lines.append("\nCalibration data:")
            for p in sorted(self.calibration_points, key=lambda x: x.final_bpb):
                lines.append(f"  {p.experiment} seed={p.seed}: "
                             f"step1000={p.step_1000_bpb:.4f} -> final={p.final_bpb:.4f}")
        return "\n".join(lines)

    def save(self, path: Path = STATE_PATH) -> None:
        """Persist predictor state to JSON."""
        data = {
            "best_known_final_bpb": self.best_known_final_bpb,
            "fit": {
                "intercept": self.fit.intercept,
                "slope": self.fit.slope,
                "r_squared": self.fit.r_squared,
                "residual_std": self.fit.residual_std,
                "n_points": self.fit.n_points,
                "calibrated": self.fit.calibrated,
            },
            "calibration_points": [
                {
                    "experiment": p.experiment,
                    "seed": p.seed,
                    "step_1000_bpb": p.step_1000_bpb,
                    "final_bpb": p.final_bpb,
                }
                for p in self.calibration_points
            ],
        }
        path.write_text(json.dumps(data, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path = STATE_PATH) -> ScalingPredictor:
        """Load predictor from persisted state, or return fresh instance with priors."""
        sp = cls()
        if not path.exists():
            return sp
        data = json.loads(path.read_text())
        sp.best_known_final_bpb = data.get("best_known_final_bpb", 1.0810)
        for pt in data.get("calibration_points", []):
            sp.calibration_points.append(CalibrationPoint(**pt))
        if sp.calibration_points:
            sp._refit()
        else:
            fit_data = data.get("fit", {})
            sp.fit = RegressionFit(
                intercept=fit_data.get("intercept", META_PRIOR_INTERCEPT),
                slope=fit_data.get("slope", META_PRIOR_SLOPE),
                r_squared=fit_data.get("r_squared", 0.74),
                residual_std=fit_data.get("residual_std", META_PRIOR_RESIDUAL_STD),
                n_points=0,
                calibrated=False,
            )
        return sp


# --- CLI for quick checks ---
if __name__ == "__main__":
    import sys
    sp = ScalingPredictor.load()

    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        bpb = float(sys.argv[2])
        pred, lo, hi = sp.predict(bpb)
        decision = sp.decide_qualify(bpb)
        print(f"Step 1000 BPB: {bpb:.4f}")
        print(f"Predicted final: {pred:.4f} [{lo:.4f}, {hi:.4f}]")
        print(f"Decision: {decision.action} — {decision.reason}")
    elif len(sys.argv) > 1 and sys.argv[1] == "add":
        # scaling_laws.py add <experiment> <seed> <step1000_bpb> <final_bpb>
        sp.add_calibration_point(sys.argv[2], int(sys.argv[3]),
                                 float(sys.argv[4]), float(sys.argv[5]))
        sp.save()
        print(f"Added calibration point. Now {sp.fit.n_points} points.")
        print(sp.summary())
    else:
        print(sp.summary())
