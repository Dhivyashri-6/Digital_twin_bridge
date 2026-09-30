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

## v2 study pipeline (journal revision)

Run everything with `python scripts/run_all_experiments.py`; each stage writes `results/runs/<run_id>/`
(config snapshot, package versions, seeds, wall time, peak memory, SHA-256 of every output).

1. **Data** (`scripts/process_z24.py`, `src/z24_catalog.py`). All 306 PDT recordings (17 scenarios x AVT/FVT x
   9 setups) at 100 Hz (verified from all 891 `.aaa` headers). Recording ID = `S<scenario>_<test>_setup<NN>`.
   `.aaa` and `.zip` copies are read, verified sample-by-sample against the `.mat` data and counted once.
   Preprocessing: linear detrend + 0.5-40 Hz Butterworth band-pass (order 4, zero phase); no normalisation
   (per-record z-score only as an ablation). 10 s windows, 50% overlap; 20 time/spectral features per window.
2. **Modal features** (`src/modal_analysis.py`). FDD of the multichannel CSD (Welch, 4096-sample Hann segments,
   0.0244 Hz resolution); peaks of the first singular value with >= 0.3 decades prominence in 1.5-15 Hz.
   Baseline modes from the averaged spectrum of the scenario-01 AVT records; each record's peaks are assigned to
   baseline modes by Hungarian matching of MAC (on the five reference sensors present in every setup, MAC >= 0.8)
   minus a relative-frequency penalty. Damping by EFDD (MAC >= 0.8 bell, log decrement on the correlation
   function); estimates are kept only if 0 < zeta < 10%, R^2 >= 0.95. FVT records are shaker-dominated, so modal
   features are used for AVT only.
3. **Change detection** (`src/bocpd.py`, `scripts/run_change_detection.py`). Within-record univariate BOCPD on
   window RMS (as before), and a multivariate (diagonal Normal-Gamma) BOCPD on the AVT recordings ordered by
   scenario, using f1-f4 standardised with the 9 scenario-01 records; detections are MAP run-length resets.
   Baseline: two-sided tabular CUSUM with re-baselining. Metrics: precision, recall, F1 and delay against the
   16 scenario boundaries (tolerance 4 records). The prior precision kappa0 of the segment mean was set to 0.01
   (vague) after the default kappa0 = 1 was found to suppress new segments; all values are reported.
4. **Classification** (`src/evaluation.py`, `scripts/run_classification.py`). Non-overlapping AVT windows;
   leave-one-scenario-out and stratified grouped 5-fold by recording; 3 seeds (reduced from 5 after the first
   LOSO runs showed a seed s.d. of the ROC-AUC well below the bootstrap CI width; ablations use 1 seed); balanced class weights;
   XGBoost, random forest, RBF-SVM (<= 20 000 training rows per fold), logistic regression, and a recording-level
   natural-frequency shift threshold (Youden threshold chosen on training folds). Cluster bootstrap over
   recordings for 95% CIs. XGBoost with monotone constraints on f_k (and a signed frequency-decrease threshold)
   were added after the first LOSO results as physically motivated variants and are reported separately
   (modal features only; ablations use LOSO only, to bound the runtime).
5. **SHAP** (`scripts/run_shap.py`): exact TreeSHAP for every held-out LOSO window (seed 0); sensors ranked by
   channel-level attribution and aggregated by sensor group/direction/point.
6. **Risk** (`scripts/run_risk.py`): classifier term included only if LOSO ROC-AUC >= 0.7.
7. **EMS** (`scripts/run_ems.py`): hourly FDD tracking over the monitoring year, temperature compensation
   (hinge regression at 0 degC; PCA), BOCPD/CUSUM on daily novelty; onset window 05-14.08.1998 from the PDT reports.
8. **Novelty** (`scripts/run_novelty.py`): Mahalanobis distance of f1-f4 from a baseline (scenario 01, or scenarios 01+02,
   both measured before any intervention); threshold chi2(99%) and an empirical threshold, the largest leave-one-out
   distance of the baseline records. The empirical threshold and the 01+02 baseline were added after the chi2 threshold
   flagged every recording.
9. **SSI cross-check** (`scripts/run_ssi.py`, `src/modal_analysis.ssi_cov`): covariance-driven SSI on the reference
   sensors (50 Hz, 40 block rows, orders 20-60, stabilisation df < 1%, dzeta < 30%), nearest stable pole within 5% of FDD.
10. **Mode-shape changes** (`scripts/run_mode_shapes.py`): FDD singular vectors at the matched modes, phase-aligned and
    scaled on the reference sensors; |dphi_k| relative to the scenario-01 record of the same setup as channel-level
    features (scenario 01 = baseline, excluded from evaluation). Added after SHAP showed near-uniform sensor attributions.
11. **Dashboard** (`app.py`): replays the stored study outputs along the scenario sequence (frequencies, sequence BOCPD,
    novelty, gated sensor states, SHAP, detection log); nothing is recomputed in the app.
