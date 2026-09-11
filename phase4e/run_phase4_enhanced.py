"""run_phase4_enhanced.py - Orchestrates the enhanced Graph-NSGA-II plus learned-perturbation optimization run."""
import sys, os


POPULATION_SIZE = 30
N_GENERATIONS   = 50
FINAL_OUTPUT    = "./pipeline_artifacts/phase4_enhanced_pareto_front.json"


def execute_enhanced_phase_4():
    print("="*72)
    print("  PHASE 4 (ENHANCED): Graph-NSGA-II + Learned Perturbation")
    print("="*72)

    print("\n[1/4] Loading Phase 2 feasibility map and mask...")
    feasibility_map = load_feasibility_map()
    mask            = load_mask()

    print("\n[2/4] Loading iML1515 and computing baseline...")
    model           = load_iml1515()
    baseline_result = evaluate_candidate(model, WILD_TYPE)
    baseline_growth = baseline_result["baseline_growth"]
    wt_mpb          = baseline_result["growth_percent_reduction"]
    print(f"      Baseline growth: {baseline_growth:.4f} 1/h | WT MPB: {wt_mpb:.4f}%")

    print("\n[3/4] Running Graph-NSGA-II + Learned Perturbation...")
    engine = GraphNSGA2Engine(
        feasibility_map = feasibility_map,
        mask            = mask,
        model           = model,
        baseline_growth = baseline_growth,
        population_size = POPULATION_SIZE,
        n_generations   = N_GENERATIONS,
    )
    archive, history = engine.run()

    print("\n[4/4] Saving results...")
    path  = archive.save_final(FINAL_OUTPUT)
    front = archive.front

    print(f"\n{'='*72}")
    print(f"  PHASE 4 ENHANCED COMPLETE")
    print(f"{'='*72}")
    print(f"  Pareto front candidates: {len(front)}")
    print(f"  Wild-type MPB:           {wt_mpb:.4f}%")
    if front:
        best_struct = max(front, key=lambda c: c["structural_score"])
        best_mpb    = min(front, key=lambda c: c["mpb_score"])
        print(f"  Best structural:  {best_struct['structural_score']:.4f} -> {best_struct['sequence']}")
        print(f"  Best MPB:         {best_mpb['mpb_score']:.4f}% -> {best_mpb['sequence']}")
    print(f"  Output: {path}")
    print(f"{'='*72}\n")

    return archive, history


#  RUN
execute_enhanced_phase_4()
