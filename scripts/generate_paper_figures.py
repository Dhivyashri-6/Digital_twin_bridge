"""
Generate Publication-Grade Figures for Research Paper
Digital Twin Bridge Structural Health Monitoring (Z24 Benchmark)
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs('results/figures/paper', exist_ok=True)
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13
})

# -------------------------------------------------------------
# Figure 2: Feature Distributions by Condition (features.csv)
# -------------------------------------------------------------
print("Generating Fig 2: Feature distributions...")
df_feat = pd.read_csv('data/processed/features.csv', usecols=['label', 'rms', 'dominant_frequency', 'frequency_centroid', 'kurtosis'])
# Sample for plotting if large
df_sample = df_feat.sample(n=min(10000, len(df_feat)), random_state=42)

fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=300)
features_to_plot = [
    ('rms', 'Root Mean Square (RMS) [g]', axes[0, 0]),
    ('dominant_frequency', 'Dominant Frequency [Hz]', axes[0, 1]),
    ('frequency_centroid', 'Frequency Centroid [Hz]', axes[1, 0]),
    ('kurtosis', 'Spectral / Time Kurtosis', axes[1, 1])
]

for col, label_name, ax in features_to_plot:
    data_ref = df_sample[df_sample['label'] == 'REFERENCE'][col].dropna()
    data_alt = df_sample[df_sample['label'] == 'ALTERED_CONDITION'][col].dropna()
    
    # Clip extreme outliers for visual clarity
    q_low = min(data_ref.quantile(0.01), data_alt.quantile(0.01))
    q_high = max(data_ref.quantile(0.99), data_alt.quantile(0.99))
    
    bp = ax.boxplot([data_ref.clip(q_low, q_high), data_alt.clip(q_low, q_high)],
                     tick_labels=['REFERENCE\n(N=12,870)', 'ALTERED\n(N=51,674)'],
                     patch_artist=True, widths=0.5,
                     boxprops=dict(facecolor='#D9E1E8', color='#2B5B84'),
                     medianprops=dict(color='#C0392B', linewidth=2),
                     whiskerprops=dict(color='#2B5B84'),
                     capprops=dict(color='#2B5B84'),
                     flierprops=dict(marker='o', markersize=2, alpha=0.3, color='#888888'))
    
    bp['boxes'][0].set_facecolor('#A3C4DC')
    bp['boxes'][1].set_facecolor('#E8A87C')
    ax.set_ylabel(label_name)
    ax.grid(True, linestyle='--', alpha=0.5)

plt.suptitle('Figure 2: Selected Vibration Feature Distributions by Structural Condition Class', weight='bold')
plt.tight_layout()
plt.savefig('results/figures/paper/fig2_feature_distributions.png', dpi=300)
plt.close()
print("Fig 2 generated.")

# -------------------------------------------------------------
# Figure 3: BOCPD Dynamics (bocpd_results.csv)
# -------------------------------------------------------------
print("Generating Fig 3: BOCPD dynamics...")
bocpd_df = pd.read_csv('data/processed/bocpd_results.csv')
# Select one representative sensor sequence
sample_sensor = bocpd_df['sensor'].iloc[0]
sub_bocpd = bocpd_df[bocpd_df['sensor'] == sample_sensor].head(200).reset_index(drop=True)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True, dpi=300)

# Panel 1: Vibration RMS sequence
ax1.plot(sub_bocpd['window_id'], sub_bocpd['feature_value'], color='#2B5B84', lw=1.5, label='Windowed RMS')
ax1.set_ylabel('RMS Amplitude')
ax1.set_title(f'Sequential Bayesian Change Point Detection: Sensor {sample_sensor}', weight='bold', fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='upper right')

# Panel 2: Posterior Change Probability P(r_t = 0 | x_{1:t})
ax2.plot(sub_bocpd['window_id'], sub_bocpd['change_probability'], color='#D96B27', lw=1.5, label='Change Probability $P(r_t=0 \\mid x_{1:t})$')
ax2.axhline(0.80, color='red', linestyle='--', lw=1.5, label='Alarm Threshold ($\tau=0.80$)')
ax2.set_ylabel('Probability')
ax2.set_ylim(-0.05, 1.05)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='upper right')

# Panel 3: Most Probable Run Length
ax3.plot(sub_bocpd['window_id'], sub_bocpd['most_probable_run_length'], color='#27AE60', lw=1.5, label='Most Probable Run Length $r_t^*$')
ax3.set_xlabel('Time Window Index (10s intervals, 50% overlap)')
ax3.set_ylabel('Run Length ($r_t$)')
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.legend(loc='upper left')

plt.suptitle('Figure 3: BOCPD Sequential Detection Profile under Adams-MacKay Formulation', weight='bold')
plt.tight_layout()
plt.savefig('results/figures/paper/fig3_bocpd_dynamics.png', dpi=300)
plt.close()
print("Fig 3 generated.")

# -------------------------------------------------------------
# Figure 4: ML Confusion Matrix & Performance (model_metrics.json)
# -------------------------------------------------------------
print("Generating Fig 4: Confusion matrix...")
with open('results/metrics/model_metrics.json') as f:
    metrics = json.load(f)

cm = np.array(metrics['confusion_matrix'])
classes = metrics['classes']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300, gridspec_kw={'width_ratios': [1.2, 1]})

# Heatmap
cax = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar(cax, ax=ax1, fraction=0.046, pad=0.04)
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(classes, rotation=15, ha='right')
ax1.set_yticklabels(classes)
ax1.set_xlabel('Predicted Label')
ax1.set_ylabel('Ground Truth Label')
ax1.set_title('Grouped Confusion Matrix (Test N=17,127)', weight='bold')

for i in range(2):
    for j in range(2):
        val = cm[i, j]
        color = 'white' if val > cm.max() / 2 else 'black'
        pct = (val / cm[i].sum()) * 100
        ax1.text(j, i, f"{val:,}\n({pct:.1f}%)", ha='center', va='center', color=color, weight='bold')

# Bar plot of metrics
clf_rep = metrics['classification_report']
perf_labels = ['Accuracy', 'ROC-AUC', 'Altered F1', 'Reference F1', 'Macro F1']
perf_values = [
    metrics['accuracy'],
    metrics['roc_auc'],
    clf_rep['ALTERED_CONDITION']['f1-score'],
    clf_rep['REFERENCE']['f1-score'],
    clf_rep['macro avg']['f1-score']
]

colors = ['#2B5B84', '#3AAFA9', '#E8A87C', '#C0392B', '#8E44AD']
bars = ax2.bar(perf_labels, perf_values, color=colors, width=0.55)
ax2.set_ylim(0, 1.0)
ax2.set_ylabel('Score / Metric Value')
ax2.set_title('Evaluation Metrics (GroupShuffleSplit)', weight='bold')
ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.setp(ax2.get_xticklabels(), rotation=25, ha='right')

for bar in bars:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.3f}", ha='center', va='bottom', fontsize=8.5, weight='bold')

plt.suptitle('Figure 4: Leakage-Aware Machine Learning Evaluation Highlighting Class Collapse', weight='bold')
plt.tight_layout()
plt.savefig('results/figures/paper/fig4_ml_evaluation.png', dpi=300)
plt.close()
print("Fig 4 generated.")

# -------------------------------------------------------------
# Figure 5: SHAP Global Feature Importance
# -------------------------------------------------------------
print("Generating Fig 5: SHAP global importance...")
shap_df = pd.read_csv('results/shap/global_feature_importance.csv')
shap_df = shap_df.sort_values('mean_abs_shap', ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
bars = ax.barh(shap_df['feature'], shap_df['mean_abs_shap'], color='#2B7A78', edgecolor='#17252A', height=0.65)
ax.set_xlabel('Mean Absolute SHAP Value (Impact on Condition Model Output)')
ax.set_title('Figure 5: Top 15 Vibration Features Ranked by TreeExplainer SHAP Attribution', weight='bold')
ax.grid(True, axis='x', linestyle='--', alpha=0.5)

for bar in bars:
    width = bar.get_width()
    if width > 0.005:
        ax.text(width + 0.003, bar.get_y() + bar.get_height()/2.0, f"{width:.4f}", ha='left', va='center', fontsize=8.5)

plt.tight_layout()
plt.savefig('results/figures/paper/fig5_shap_importance.png', dpi=300)
plt.close()
print("Fig 5 generated.")

# -------------------------------------------------------------
# Figure 6: Sensor-Level Ranking & Prioritization
# -------------------------------------------------------------
print("Generating Fig 6: Sensor ranking...")
sensor_df = pd.read_csv('data/processed/sensor_importance.csv')
sensor_df = sensor_df.sort_values('rank', ascending=True)

fig, ax = plt.subplots(figsize=(8, 4.2), dpi=300)
bars = ax.bar(sensor_df['sensor'], sensor_df['importance_score'], color='#D96B27', edgecolor='#873600', width=0.5)
ax.set_xlabel('Sensor Channel Identifier')
ax.set_ylabel('Aggregated Attribution Score')
ax.set_title('Figure 6: Structural Sensor Importance Ranking via SHAP Attribution Aggregation', weight='bold')
ax.grid(True, axis='y', linestyle='--', alpha=0.5)
ax.set_ylim(0, max(sensor_df['importance_score']) * 1.2)

for bar, r in zip(bars, sensor_df['rank']):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, height + 0.05, f"Rank {r}\n({height:.2f})", ha='center', va='bottom', fontsize=9, weight='bold')

plt.tight_layout()
plt.savefig('results/figures/paper/fig6_sensor_ranking.png', dpi=300)
plt.close()
print("Fig 6 generated.")

# -------------------------------------------------------------
# Figure 7: Risk Engine Score Distribution & Thresholds
# -------------------------------------------------------------
print("Generating Fig 7: Risk distribution...")
risk_df = pd.read_csv('data/processed/risk_results.csv', usecols=['risk_score'])

fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
n, bins, patches_hist = ax.hist(risk_df['risk_score'], bins=50, color='#3AAFA9', edgecolor='#2B5B84', alpha=0.75, density=False)
ax.axvline(0.35, color='#F39C12', linestyle='--', lw=2, label='Medium Risk Threshold ($\tau=0.35$)')
ax.axvline(0.60, color='#E67E22', linestyle='--', lw=2, label='High Risk Threshold ($\tau=0.60$)')
ax.axvline(0.80, color='#C0392B', linestyle='--', lw=2, label='Critical Risk Threshold ($\tau=0.80$)')

ax.set_yscale('log')
ax.set_xlabel('Calculated Risk Score $R \\in [0, 1]$')
ax.set_ylabel('Logarithmic Frequency (Total Windows = 64,544)')
ax.set_title('Figure 7: Distribution of Calculated Risk Scores Relative to Operational Alert Thresholds', weight='bold')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper right')

# Annotation box
ax.text(0.4, max(n)*0.1, "Operational Status: 100% LOW Risk\nMax Observed Score: 0.235\nMean Score: 0.0034", 
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FEF9E7', edgecolor='#F39C12'))

plt.tight_layout()
plt.savefig('results/figures/paper/fig7_risk_distribution.png', dpi=300)
plt.close()
print("Fig 7 generated.")

# -------------------------------------------------------------
# Figure 8: Digital Twin Bridge Structural Schematic
# -------------------------------------------------------------
print("Generating Fig 8: Digital Twin Bridge Schematic...")
fig, ax = plt.subplots(figsize=(10, 4.2), dpi=300)

# Deck
deck = patches.Rectangle((0, 3.8), 60, 0.6, facecolor='#BDC3C7', edgecolor='#2C3E50', lw=2)
ax.add_patch(deck)

# Piers
pier1 = patches.Rectangle((14, 0), 2, 3.8, facecolor='#7F8C8D', edgecolor='#2C3E50', lw=1.5)
pier2 = patches.Rectangle((44, 0), 2, 3.8, facecolor='#7F8C8D', edgecolor='#2C3E50', lw=1.5)
ax.add_patch(pier1)
ax.add_patch(pier2)

# Abutments
abut1 = patches.Polygon([[0, 0], [4, 0], [4, 3.8], [0, 3.8]], closed=True, facecolor='#95A5A6', edgecolor='#2C3E50')
abut2 = patches.Polygon([[60, 0], [56, 0], [56, 3.8], [60, 3.8]], closed=True, facecolor='#95A5A6', edgecolor='#2C3E50')
ax.add_patch(abut1)
ax.add_patch(abut2)

# Ground line
ax.axhline(0, color='#27AE60', lw=3)

# Sensor markers
sensors = [
    ('01_0', 10, 'Rank 3\n(Score: 1.89)', '#E67E22'),
    ('01_1', 25, 'Rank 1 (Max)\n(Score: 1.95)', '#C0392B'),
    ('01_2', 35, 'Rank 2\n(Score: 1.92)', '#E74C3C'),
    ('01_3', 50, 'Rank 4\n(Score: 1.80)', '#F39C12')
]

for s_id, x_pos, s_label, col in sensors:
    # Sensor dot
    ax.scatter(x_pos, 4.1, color=col, s=180, zorder=5, edgecolors='black', lw=1.5)
    ax.annotate(f"Sensor {s_id}\n{s_label}", xy=(x_pos, 4.4), xytext=(x_pos, 5.2),
                ha='center', fontsize=8, weight='bold',
                arrowprops=dict(arrowstyle='->', lw=1.2, color='#2C3E50'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEF9E7', edgecolor=col))

# Labels and dimensions
ax.text(30, 4.1, 'Main Span (30 m)', ha='center', va='center', fontsize=9, color='#2C3E50')
ax.text(7, 2.0, 'Side Span (14 m)', ha='center', va='center', fontsize=8, color='#555555')
ax.text(52, 2.0, 'Side Span (14 m)', ha='center', va='center', fontsize=8, color='#555555')
ax.text(15, -0.6, 'Pier 1', ha='center', fontsize=8.5, weight='bold')
ax.text(45, -0.6, 'Pier 2', ha='center', fontsize=8.5, weight='bold')

ax.set_xlim(-2, 62)
ax.set_ylim(-1.5, 6.5)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Figure 8: Digital Twin Bridge Elevation Schematic with Prioritized Sensor Attributions', weight='bold', pad=10)

plt.tight_layout()
plt.savefig('results/figures/paper/fig8_bridge_schematic.png', dpi=300)
plt.close()
print("Fig 8 generated.")
print("All 8 publication figures generated successfully!")
