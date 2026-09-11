"""run_phase4.py - Orchestrates the primary NSGA-II plus MCTS optimization run and writes the Pareto front artifact."""
import json

# ── Configuration ─────────────────────────────────────────────────────────
POPULATION_SIZE  = 30
N_GENERATIONS    = 50
N_MCTS_ROLLOUTS  = 50
CHECKPOINT_DIR   = "./phase4_checkpoints"
FINAL_OUTPUT     = "./pipeline_artifacts/phase4_pareto_front.json"


def execute_pipeline_phase_4():
    print("="*72)
    print("  PHASE 4: PARETO EVOLUTIONARY LOOP")
    print("  Feasibility-Conditioned Multi-Objective GA with MCTS Mutations")
    print("="*72)

    # Load Phase 2 artifacts
    print("\n[1/4] Loading Phase 2 feasibility map and mask...")
    feasibility_map = load_feasibility_map()
    mask            = load_mask()
    print(f"      Feasibility map: {sum(1 for v in feasibility_map.values() if v['status']=='MUTABLE')} mutable positions")

    # Load Phase 3 metabolic model
    print("\n[2/4] Loading iML1515 and computing baseline growth rate...")
    model = load_iml1515()
    baseline_result  = evaluate_candidate(model, WILD_TYPE)
    baseline_growth  = baseline_result["baseline_growth"]
    wt_mpb           = baseline_result["growth_percent_reduction"]
    print(f"      Baseline growth: {baseline_growth:.4f} 1/h")
    print(f"      Wild-type MPB:   {wt_mpb:.4f}%")

    # Run NSGA-II + MCTS
    print("\n[3/4] Running evolutionary optimization...")
    engine = NSGA2Engine(
        feasibility_map  = feasibility_map,
        mask             = mask,
        model            = model,
        baseline_growth  = baseline_growth,
        population_size  = POPULATION_SIZE,
        n_generations    = N_GENERATIONS,
        n_mcts_rollouts  = N_MCTS_ROLLOUTS,
        checkpoint_dir   = CHECKPOINT_DIR,
        use_mcts         = True,
    )
    pareto_archive, history = engine.run()

    # Save final output
    print("\n[4/4] Saving final Pareto front...")
    path = pareto_archive.save_final(FINAL_OUTPUT)

    # Audit
    front = pareto_archive.front
    print(f"\n{'='*72}")
    print(f"  PHASE 4 COMPLETE")
    print(f"{'='*72}")
    print(f"  Pareto front candidates: {len(front)}")
    print(f"  Wild-type reference:     struct={baseline_result.get('structural_score','N/A')} | mpb={wt_mpb:.4f}%")
    if front:
        best_struct = max(front, key=lambda c: c["structural_score"])
        best_mpb    = min(front, key=lambda c: c["mpb_score"])
        print(f"  Best structural score:   {best_struct['structural_score']:.4f} -> {best_struct['sequence']}")
        print(f"  Best MPB (lowest cost):  {best_mpb['mpb_score']:.4f}% -> {best_mpb['sequence']}")
    print(f"  Output: {path}")
    print(f"{'='*72}\n")

    return pareto_archive, history


#  RUN
execute_pipeline_phase_4()
