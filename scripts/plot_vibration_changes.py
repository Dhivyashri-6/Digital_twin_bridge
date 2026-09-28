"""Plot vibration behaviour together with BOCPD change points.

Example:
    python scripts/plot_vibration_changes.py --sensor 01_0 --recording 01setup01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = {
    "sensor",
    "recording",
    "window_id",
    "feature_value",
    "change_probability",
}


def plot_vibration_changes(
    data: pd.DataFrame,
    *,
    sensor: str | None = None,
    recording: str | None = None,
    threshold: float = 0.8,
    output: str | Path | None = None,
    show: bool = False,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Create aligned vibration and change-probability plots.

    Returns the Matplotlib figure and the filtered rows used in the plot.
    A change point is any row whose probability is at least ``threshold``.
    """
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    frame = data.copy()
    frame["sensor"] = frame["sensor"].astype(str)
    frame["recording"] = frame["recording"].astype(str)
    if sensor is not None:
        frame = frame[frame["sensor"] == str(sensor)]
    if recording is not None:
        frame = frame[frame["recording"] == str(recording)]
    if frame.empty:
        raise ValueError("No rows match the requested sensor and recording")

    # Avoid connecting the end of one recording to the start of another.
    frame = frame.sort_values(["sensor", "recording", "window_id"]).copy()
    frame["change_point"] = frame["change_probability"] >= threshold
    groups = list(frame.groupby(["sensor", "recording"], sort=False))

    fig, (signal_ax, probability_ax) = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True,
        gridspec_kw={"height_ratios": (2, 1)}, constrained_layout=True,
    )
    colors = plt.get_cmap("tab10")
    offset = 0
    tick_positions: list[float] = []
    tick_labels: list[str] = []

    for index, ((sensor_id, recording_id), group) in enumerate(groups):
        group = group.copy()
        x = range(offset, offset + len(group))
        group["plot_position"] = list(x)
        frame.loc[group.index, "plot_position"] = group["plot_position"]
        colour = colors(index % 10)
        label = f"{sensor_id} / {recording_id}"

        signal_ax.plot(group["plot_position"], group["feature_value"], color=colour,
                       linewidth=1.2, label=label)
        probability_ax.plot(group["plot_position"], group["change_probability"],
                            color=colour, linewidth=1.2)
        changes = group[group["change_point"]]
        if not changes.empty:
            signal_ax.scatter(changes["plot_position"], changes["feature_value"],
                              color="crimson", marker="x", s=60, linewidth=2,
                              zorder=4, label="Detected change" if index == 0 else None)
            for change_x in changes["plot_position"]:
                signal_ax.axvline(change_x, color="crimson", alpha=0.3, linestyle="--")
                probability_ax.axvline(change_x, color="crimson", alpha=0.3, linestyle="--")

        tick_positions.extend(group["plot_position"].tolist())
        tick_labels.extend(group["window_id"].astype(str).tolist())
        offset += len(group)
        if index < len(groups) - 1:
            signal_ax.axvline(offset - 0.5, color="0.65", linewidth=0.8)
            probability_ax.axvline(offset - 0.5, color="0.65", linewidth=0.8)

    feature_name = str(frame["feature"].iloc[0]) if "feature" in frame else "vibration feature"
    signal_ax.set_title("Vibration behaviour and Bayesian change points")
    signal_ax.set_ylabel(feature_name.upper())
    signal_ax.grid(alpha=0.25)
    signal_ax.legend(loc="best", fontsize=8)

    probability_ax.axhline(threshold, color="darkorange", linestyle="--",
                           linewidth=1.4, label=f"Threshold = {threshold:g}")
    probability_ax.set_ylabel("Change probability")
    probability_ax.set_xlabel("Window ID")
    probability_ax.set_ylim(-0.02, 1.02)
    probability_ax.grid(alpha=0.25)
    probability_ax.legend(loc="best", fontsize=8)

    # Keep labels readable for long recordings.
    stride = max(1, len(tick_positions) // 20)
    probability_ax.set_xticks(tick_positions[::stride], tick_labels[::stride], rotation=45)

    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return fig, frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot vibration behaviour and BOCPD change points."
    )
    parser.add_argument("--input", default="data/processed/bocpd_results.csv")
    parser.add_argument("--sensor", help="Sensor ID to plot (for example, 01_0)")
    parser.add_argument("--recording", help="Recording ID to plot")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--output", default="results/figures/vibration_change_points.png")
    parser.add_argument("--show", action="store_true", help="Open an interactive plot window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(
            f"{input_path} does not exist. Run scripts/run_bocpd.py first."
        )
    data = pd.read_csv(input_path)
    _, plotted = plot_vibration_changes(
        data,
        sensor=args.sensor,
        recording=args.recording,
        threshold=args.threshold,
        output=args.output,
        show=args.show,
    )
    print(
        f"Saved {args.output} using {len(plotted)} windows; "
        f"detected {int(plotted['change_point'].sum())} change points."
    )


if __name__ == "__main__":
    main()
