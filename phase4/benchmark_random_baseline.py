"""benchmark_random_baseline.py - Runs an NSGA-II baseline using random mutation instead of MCTS, as a control for comparison."""
import sys, os


POPULATION_SIZE = 30
N_GENERATIONS   = 50
OUTPUT_PATH     = "./pipeline_artifacts/phase4_pareto_front_RANDOM.json"


def run_random_baseline():
    print("="*72)
    print("  BENCHMARK: NSGA-II with RANDOM mutation (no MCTS)")
    print("="*72)

    feasibility_map = load_feasibility_map()
    mask            = load_mask()
    model           = load_iml1515()
    baseline_result = evaluate_candidate(model, WILD_TYPE)
    baseline_growth = baseline_result["baseline_growth"]

    engine = NSGA2Engine(
        feasibility_map = feasibility_map,
        mask            = mask,
        model           = model,
        baseline_growth = baseline_growth,
        population_size = POPULATION_SIZE,
        n_generations   = N_GENERATIONS,
        checkpoint_dir  = "./phase4_checkpoints/random_baseline",
        use_mcts        = False,   # <-- the only difference from run_phase4.py
    )
    archive, history = engine.run()
    archive.save_final(OUTPUT_PATH)

    print(f"\n  Random baseline Pareto front: {len(archive.front)} candidates")
    print(f"  Saved to: {OUTPUT_PATH}")
    return archive, history


#  RUN
run_random_baseline()
