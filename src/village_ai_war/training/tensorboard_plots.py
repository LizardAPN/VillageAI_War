"""Plot TensorBoard scalar runs as PNG grids (unified training and similar)."""

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


def plot_unified_tensorboard_runs(
    log_root: Path,
    *,
    bots_subdir: str = "unified_bots",
    village_subdir: str = "unified_village",
    output_bots: Path | None = None,
    output_village: Path | None = None,
) -> list[Path]:
    """Load latest SB3 runs under ``log_root`` and write PNG grids."""
    written: list[Path] = []
    if output_bots is None:
        output_bots = log_root / "plots" / "unified_bots_scalars.png"
    if output_village is None:
        output_village = log_root / "plots" / "unified_village_scalars.png"

    bot_dir = latest_run_dir(log_root, bots_subdir)
    if bot_dir is not None:
        s = load_scalar_series(bot_dir)
        if s:
            written.extend(
                plot_scalar_groups(s, f"TensorBoard scalars ({bots_subdir})", output_bots)
            )
            logger.info("Wrote bot phase metric plots to {}", output_bots.parent)
    else:
        logger.warning("No TensorBoard run directory under {}", log_root / bots_subdir)

    vil_dir = latest_run_dir(log_root, village_subdir)
    if vil_dir is not None:
        s = load_scalar_series(vil_dir)
        if s:
            written.extend(
                plot_scalar_groups(s, f"TensorBoard scalars ({village_subdir})", output_village)
            )
            logger.info("Wrote village phase metric plots to {}", output_village.parent)
    else:
        logger.warning("No TensorBoard run directory under {}", log_root / village_subdir)

    return written


def main_cli(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Plot TensorBoard scalars from unified training runs.")
    p.add_argument("--log-root", type=Path, default=Path("logs"), help="Root containing unified_bots / unified_village")
    p.add_argument("--bots-subdir", default="unified_bots")
    p.add_argument("--village-subdir", default="unified_village")
    p.add_argument("--output-bots", type=Path, default=None)
    p.add_argument("--output-village", type=Path, default=None)
    args = p.parse_args(argv)
    plot_unified_tensorboard_runs(
        args.log_root.resolve(),
        bots_subdir=args.bots_subdir,
        village_subdir=args.village_subdir,
        output_bots=args.output_bots,
        output_village=args.output_village,
    )


if __name__ == "__main__":
    main_cli()
