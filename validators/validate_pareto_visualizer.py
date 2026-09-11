"""validate_pareto_visualizer.py - Plots the Pareto front(s) to visually compare optimization runs."""
import sys, os, json, math

import matplotlib
matplotlib.use("Agg")   # headless-safe; change to "TkAgg" or remove for interactive
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


MCTS_FRONT_PATH   = "./pipeline_artifacts/phase4_pareto_front.json"
RANDOM_FRONT_PATH = "./pipeline_artifacts/phase4_pareto_front_RANDOM.json"
OUTPUT_DIR        = "./pipeline_artifacts/figures"


def load_front(path):
    with open(path) as f: return json.load(f)["pareto_front"]


def compute_hypervolume_2d(front, ref_point):
    """
    2D hypervolume indicator. ref_point = (worst_structural, worst_mpb).
    Structural: maximize, so flip sign for standard min-based computation.
    """
    # Convert to minimization: negate structural_score
    points = sorted([(-c["structural_score"], c["mpb_score"]) for c in front],
                    key=lambda p: p[0])
    hv, prev_x = 0.0, ref_point[0]
    for x, y in points:
        if y < ref_point[1]:
            hv += (ref_point[0] - x) * (ref_point[1] - y)
    return hv


def plot_pareto_comparison(mcts_front, random_front, wt_structural, wt_mpb, out_dir):
    fig, ax = plt.subplots(figsize=(9, 6))

    def _plot_front(front, color, label, marker):
        xs = [c["structural_score"] for c in front]
        ys = [c["mpb_score"]        for c in front]
        ax.scatter(xs, ys, c=color, label=label, marker=marker, s=60, alpha=0.85, zorder=3)

    if mcts_front:   _plot_front(mcts_front,   "#2196F3", "MCTS-guided NSGA-II", "o")
    if random_front: _plot_front(random_front, "#FF5722", "Random mutation NSGA-II", "^")

    # Wild-type reference
    ax.scatter([wt_structural], [wt_mpb], c="#4CAF50", marker="*", s=250,
               # >>> PEPTIDE_SPECIFIC: replace for a new target peptide
               label=f"Wild-type 1CHL", zorder=5)
               # <<< END_PEPTIDE_SPECIFIC
    ax.annotate("WT", (wt_structural, wt_mpb), textcoords="offset points",
                xytext=(6, 4), fontsize=8, color="#4CAF50")

    ax.set_xlabel("Structural Viability Score (log P,  higher = better)", fontsize=11)
    ax.set_ylabel("Metabolic Production Burden (% growth reduction,  lower = better)", fontsize=11)
    # >>> PEPTIDE_SPECIFIC: replace for a new target peptide
    ax.set_title("Pareto Frontier: Structure vs. Manufacturability\n(1CHL-targeted GBM Peptide Optimization)", fontsize=12)
    # <<< END_PEPTIDE_SPECIFIC
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "pareto_comparison.png")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"[SAVED] {path}")
    return path


def plot_hypervolume_history(history_path, out_dir):
    """Plots per-generation hypervolume from a checkpoint metadata JSON."""
    if not os.path.exists(history_path):
        print(f"[SKIP] No history file at {history_path}")
        return

    with open(history_path) as f: data = json.load(f)
    history = data.get("metadata", {}).get("history", [])
    if not history: print("[SKIP] No history data in checkpoint."); return

    gens  = [h["generation"]       for h in history]
    sizes = [h["pareto_front_size"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(gens, sizes, color="#2196F3", linewidth=2)
    ax.fill_between(gens, sizes, alpha=0.15, color="#2196F3")
    ax.set_xlabel("Generation"); ax.set_ylabel("Pareto Front Size")
    ax.set_title("Pareto Front Growth Over Generations")
    ax.grid(True, alpha=0.3)

    path = os.path.join(out_dir, "pareto_front_growth.png")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"[SAVED] {path}")
    return path


def run_visualization():
    print("="*60)
    print("  VALIDATOR: Pareto Frontier Visualization")
    print("="*60)

    fmap         = load_feasibility_map()
    wt_structural = compute_structural_score(WILD_TYPE, fmap)

    mcts_front   = load_front(MCTS_FRONT_PATH)   if os.path.exists(MCTS_FRONT_PATH)   else []
    random_front = load_front(RANDOM_FRONT_PATH) if os.path.exists(RANDOM_FRONT_PATH) else []

    # Wild-type MPB from the saved front (it's always included in initial pop)
    wt_entries = [c for c in mcts_front if c["sequence"] == WILD_TYPE]
    wt_mpb = wt_entries[0]["mpb_score"] if wt_entries else 3.52   # Phase 3 confirmed value

    plot_pareto_comparison(mcts_front, random_front, wt_structural, wt_mpb, OUTPUT_DIR)

    # Latest checkpoint for history
    ckpt_dir = "./phase4_checkpoints"
    if os.path.isdir(ckpt_dir):
        ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".json")])
        if ckpts:
            plot_hypervolume_history(os.path.join(ckpt_dir, ckpts[-1]), OUTPUT_DIR)

    print(f"\n  MCTS front:   {len(mcts_front)} candidates")
    print(f"  Random front: {len(random_front)} candidates")
    print(f"  Figures saved to: {OUTPUT_DIR}")


#  RUN
run_visualization()
