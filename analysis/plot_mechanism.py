#!/usr/bin/env python3
import argparse
from pathlib import Path


def plot_mechanism(out_path, dpi=320):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, Rectangle
    except ImportError as exc:
        raise SystemExit("matplotlib is required. Install with: pip install matplotlib") from exc

    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    boxes = [
        (0.8, 7.3, 2.6, 1.2, "Input / Prompt"),
        (0.8, 5.4, 2.6, 1.2, r"$L_{task}$"),
        (3.9, 5.4, 2.9, 1.6, "Information Bottleneck\n$\\beta$ schedule"),
        (7.2, 5.4, 2.3, 1.6, "Gradient Reversal\n$\\lambda_{GR}$"),
        (7.2, 2.9, 2.3, 1.2, "Updated\nParameters"),
    ]

    for x, y, w, h, label in boxes:
        rect = Rectangle((x, y), w, h, facecolor="#f7f7f7", edgecolor="#333333", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)

    arrows = [
        ((2.1, 7.3), (2.1, 6.6)),
        ((3.4, 6.0), (3.9, 6.0)),
        ((6.8, 6.0), (7.2, 6.0)),
        ((8.35, 5.4), (8.35, 4.1)),
    ]

    for (x1, y1), (x2, y2) in arrows:
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color="#333333")
        ax.add_patch(arrow)

    caption = (
        "RLCP combines gradient reversal with dynamic information bottleneck scheduling\n"
        "to progressively decouple factual recall from reasoning pathways."
    )
    ax.text(5.0, 1.2, caption, ha="center", va="center", fontsize=10, color="#222222")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(dpi), bbox_inches="tight")
    print(f"Saved plot to: {out}")


def main():
    parser = argparse.ArgumentParser(description="Plot RLCP mechanism diagram")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dpi", type=int, default=320)
    args = parser.parse_args()

    plot_mechanism(out_path=args.out, dpi=args.dpi)


if __name__ == "__main__":
    main()
