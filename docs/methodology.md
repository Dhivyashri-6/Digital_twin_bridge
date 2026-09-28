# Methodology

## BOCPD — when did behavior change?

Incoming feature → predictive comparison → Bayesian update → run-length distribution → change probability → threshold decision.

The **run length** is the number of observations since the most recent statistical change. A constant hazard `H(r)=h` expresses the prior probability of a change at each observation. For observation `x_t`, growth probabilities are proportional to `p(x_t | r)p(r)(1-h)`, while reset probability is proportional to their sum times `h`. The implementation uses a conjugate Student-t predictive distribution and normalizes every update. Hazard and threshold are configurable and are not claimed to be optimal. A detected statistical change can reflect environment or operation, not necessarily damage.

## XGBoost — what condition?

Features → sequentially boosted decision trees → corrected residual errors → probability output → condition. Each new tree focuses on errors left by earlier trees, while learning rate, depth, row subsampling, and feature subsampling control complexity. Recording-grouped splitting prevents overlapping windows from the same source entering both train and test. Training stops when reliable labels or valid groups are absent.

## SHAP — why this prediction?

Prediction → SHAP values → per-feature contributions → important features → possible sensor aggregation. Positive values push toward the explained target class and negative values push away. SHAP describes model attribution, not physical causation.

## Monitoring twin

Physical bridge observations → analytics → virtual representation → visual health state. This project is a visual/monitoring Digital Twin, not a calibrated physics-based twin. If verified coordinates are unavailable, it explicitly uses a schematic sensor layout.

## Prototype risk

`risk = (w_b BOCPD + w_m abnormal probability + w_a normalized attribution) / sum(weights)`. Thresholds map it to LOW, MEDIUM, HIGH, or CRITICAL. This transparent score is decision support only, not safety certification.
