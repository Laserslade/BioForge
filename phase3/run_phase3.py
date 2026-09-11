# run_phase3.py
"""
Phase 3 Orchestrator / Smoke Test.

Unlike run_phase1.py and run_phase2.py, this is NOT a "real" production
run -- Phase 3's actual deliverable is the reusable fba_evaluator.py
function that Phase 4 will call per-candidate. Since Phase 4 doesn't
exist yet, there are no real evolved candidates to score.

This script instead proves the mechanism works end-to-end:
  1. Load iML1515 once.
  # >>> PEPTIDE_SPECIFIC: replace for a new target peptide
  2. Evaluate the wild-type 1CHL sequence itself (sanity baseline --
     the unmodified natural peptide should get *some* reasonable,
     non-zero, non-infeasible burden score).
  # <<< END_PEPTIDE_SPECIFIC
  3. Evaluate 1-2 synthetically mutated sequences (single substitutions
     at mutable positions from Phase 2's mask) to confirm the score
     actually changes with composition, in a plausible direction.
"""

import json


ARTIFACTS_DIR = "./pipeline_artifacts"


def load_wild_type_sequence() -> str:
    with open(f"{ARTIFACTS_DIR}/phase1_manifest.json") as f:
        manifest = json.load(f)
    return manifest["sequence"]


def make_test_mutant(wild_type: str) -> str:
    """
    Builds one synthetic mutant for smoke-testing only: swaps the first
    mutable (non-frozen) position to Tryptophan (W) -- the most
    metabolically expensive residue (highest ATP cost in Phase 1's
    table), so we should see the burden score visibly increase versus
    wild-type if the mechanism is working correctly.
    """
    with open(f"{ARTIFACTS_DIR}/phase2_mask.json") as f:
        mask = json.load(f)

    for i in range(len(wild_type)):
        entry = mask[f"pos_{i}"]
        if entry["status"] == "MUTABLE" and entry["wt"] != "W":
            mutant = wild_type[:i] + "W" + wild_type[i + 1:]
            return mutant, i
    raise RuntimeError("No suitable mutable position found for smoke test.")


def execute_pipeline_phase_3():
    print("================================================================================")
    print("      PHASE 3 SMOKE TEST: METABOLIC CONSTRAINT FEEDBACK (COBRApy / iML1515)")
    print("================================================================================")

    print("\n[STEP 1/3] Loading iML1515...")
    model = load_iml1515()

    # >>> PEPTIDE_SPECIFIC: replace for a new target peptide
    print("\n[STEP 2/3] Evaluating wild-type 1CHL sequence...")
    # <<< END_PEPTIDE_SPECIFIC
    wild_type = load_wild_type_sequence()
    wt_result = evaluate_candidate(model, wild_type)
    print(f"[SUCCESS] Wild-type MPB score: {wt_result['growth_percent_reduction']:.4f}% growth reduction")

    print("\n[STEP 3/3] Evaluating a synthetic high-cost mutant (smoke test)...")
    mutant_seq, mutated_pos = make_test_mutant(wild_type)
    mutant_result = evaluate_candidate(model, mutant_seq, baseline_growth=wt_result["baseline_growth"])
    print(f"[SUCCESS] Mutant (pos_{mutated_pos} -> W) MPB score: {mutant_result['growth_percent_reduction']:.4f}% growth reduction")

    print("\n================================================================================")
    print("                PHASE 3 SMOKE TEST AUDIT")
    print("================================================================================")
    print(f"* Baseline (unconstrained) growth: {wt_result['baseline_growth']:.4f} 1/h")
    print(f"* Wild-type candidate growth:      {wt_result['candidate_growth']:.4f} 1/h  ({wt_result['growth_percent_reduction']:.4f}% reduction)")
    print(f"* Mutant candidate growth:         {mutant_result['candidate_growth']:.4f} 1/h  ({mutant_result['growth_percent_reduction']:.4f}% reduction)")

    if mutant_result["growth_percent_reduction"] >= wt_result["growth_percent_reduction"]:
        print(f"* Sanity check PASSED: substituting a cheap residue for Trp increased (or held) the burden score, as expected.")
    else:
        print(f"* [WARNING] Sanity check FAILED: Trp substitution decreased the burden score. "
              f"This is suspicious -- Trp should be among the most metabolically expensive residues. "
              f"Investigate before trusting fba_evaluator.py's output.")
    print("================================================================================\n")


#  RUN
execute_pipeline_phase_3()
