"""Plot TensorBoard scalar runs as PNG grids (e.g. MAPPO training)."""

from __future__ import annotations

import math
from pathlib import Path

from loguru import logger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def latest_run_dir(log_root: Path, subdir_name: str) -> Path | None:
    """Newest immediate subdirectory under ``log_root/subdir_name`` by mtime."""
    base = log_root / subdir_name
    if not base.is_dir():
        return None
    dirs = [p for p in base.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)


def load_scalar_series(run_dir: Path) -> dict[str, tuple[list[int], list[float]]]:
    ea = EventAccumulator(str(run_dir))
    ea.Reload()
    out: dict[str, tuple[list[int], list[float]]] = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        out[tag] = ([e.step for e in events], [e.value for e in events])
    return out


def plot_scalar_groups(
    series_by_tag: dict[str, tuple[list[int], list[float]]],
    title: str,
    out_path: Path,
    max_per_figure: int = 12,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tags = sorted(series_by_tag.keys())
    if not tags:
        return []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base_stem = out_path.stem
    suffix = out_path.suffix if out_path.suffix else ".png"
    n_figures = math.ceil(len(tags) / max_per_figure)
    written: list[Path] = []

    for fig_idx in range(n_figures):
        chunk = tags[fig_idx * max_per_figure : (fig_idx + 1) * max_per_figure]
        n_plots = len(chunk)
        ncols = min(3, n_plots)
        nrows = math.ceil(n_plots / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
        if n_plots == 1:
            ax_list = [axes]
        else:
            ax_list = list(axes.flatten()) if hasattr(axes, "flatten") else [axes]
        for ax, tag in zip(ax_list, chunk):
            steps, vals = series_by_tag[tag]
            ax.plot(steps, vals, linewidth=0.8)
            ax.set_title(tag, fontsize=8)
            ax.set_xlabel("step")
            ax.grid(True, alpha=0.3)
        for j in range(len(chunk), len(ax_list)):
            ax_list[j].set_visible(False)
        fig.suptitle(f"{title} (part {fig_idx + 1}/{n_figures})")
        fig.tight_layout()
        op = (
            out_path
            if n_figures == 1
            else out_path.with_name(f"{base_stem}_p{fig_idx + 1}{suffix}")
        )
        fig.savefig(op, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(op)
    return written


def plot_tensorboard_subdir(
    log_root: Path,
    subdir: str,
    output_path: Path | None = None,
) -> list[Path]:
    """Plot the latest SB3 run under ``log_root/subdir`` into a PNG grid."""
    if output_path is None:
        output_path = log_root / "plots" / f"{subdir}_scalars.png"
    run_dir = latest_run_dir(log_root, subdir)
    if run_dir is None:
        logger.warning("No TensorBoard run directory under {}", log_root / subdir)
        return []
    series = load_scalar_series(run_dir)
    if not series:
        return []
    written = plot_scalar_groups(series, f"TensorBoard scalars ({subdir})", output_path)
    logger.info("Wrote metric plots to {}", output_path.parent)
    return written


def plot_training_tensorboard_runs(
    log_root: Path,
    *,
    primary_subdir: str = "mappo_bots",
    primary_output: Path | None = None,
    secondary_subdir: str | None = None,
    secondary_output: Path | None = None,
) -> list[Path]:
    """Plot one or two TensorBoard run folders (e.g. MAPPO plus an optional legacy folder)."""
    written: list[Path] = []
    written.extend(plot_tensorboard_subdir(log_root, primary_subdir, primary_output))
    if secondary_subdir:
        written.extend(
            plot_tensorboard_subdir(
                log_root,
                secondary_subdir,
                secondary_output,
            )
        )
    return written


def main_cli(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Plot TensorBoard scalars (default: MAPPO logs/mappo_bots/).",
    )
    p.add_argument(
        "--log-root",
        type=Path,
        default=Path("logs"),
        help="Root directory that contains the TensorBoard subfolder (e.g. mappo_bots)",
    )
    p.add_argument(
        "--subdir",
        default="mappo_bots",
        help="Subfolder under log-root with SB3 run dirs",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: log-root/plots/<subdir>_scalars.png)",
    )
    p.add_argument(
        "--second-subdir",
        default="",
        help="Optional second subfolder to plot (e.g. old unified_bots)",
    )
    p.add_argument("--second-output", type=Path, default=None, help="Output path for second plot")
    args = p.parse_args(argv)
    root = args.log_root.resolve()
    sec = args.second_subdir.strip() or None
    plot_training_tensorboard_runs(
        root,
        primary_subdir=args.subdir,
        primary_output=args.output,
        secondary_subdir=sec,
        secondary_output=args.second_output,
    )


if __name__ == "__main__":
    main_cli()
