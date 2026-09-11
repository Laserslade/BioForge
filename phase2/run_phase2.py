# run_phase2.py
"""
Phase 2 Orchestrator.

Wires together:
  1. mask_builder.py        -- builds + validates the audited design mask
  2. protein_mpnn_runner.py -- runs ProteinMPNN once, unconditional
                                per-position probabilities
  3. feasibility_map.py     -- converts raw probabilities into the final
                                continuous-score feasibility map (Option A)

Output: ./pipeline_artifacts/phase2_mask.json
        ./pipeline_artifacts/phase2_feasibility_map.json

Both mask_builder.py and feasibility_map.py were independently tested
against synthetic/real Phase 1 output during development. protein_mpnn_runner.py
was NOT executable in the development sandbox (no network, no repo, no GPU) --
its subprocess calls are built from verified real ProteinMPNN documentation,
but this is the first genuinely untested link in the chain. Expect this
step specifically to need debugging on first real run.
"""


def execute_pipeline_phase_2():
    print("================================================================================")
    print("      PRODUCTION RUN: PHASE 2 SEQUENCE INVERSION & MASKING")
    print("================================================================================")

    # 1. Build and validate the design mask
    print("\n[STEP 1/3] Building design mask...")
    mask_builder = MaskBuilder()
    mask = mask_builder.build_mask()
    mask_path = mask_builder.save_mask(mask)
    frozen_ct = sum(1 for v in mask.values() if v["status"] == "FROZEN")
    mutable_ct = sum(1 for v in mask.values() if v["status"] == "MUTABLE")
    print(f"[SUCCESS] Mask validated: {frozen_ct} frozen / {mutable_ct} mutable. Saved to {mask_path}")

    # 2. Run ProteinMPNN inference
    print("\n[STEP 2/3] Running ProteinMPNN (single forward pass, unconditional probs)...")
    mpnn_runner = ProteinMPNNRunner()
    mpnn_result = mpnn_runner.run()
    print(f"[SUCCESS] Inference complete. Probability matrix shape: {mpnn_result['probs'].shape}")

    # 3. Build the final feasibility map
    print("\n[STEP 3/3] Building feasibility map (Option A: continuous scores)...")
    fmap_builder = FeasibilityMapBuilder()
    fmap = fmap_builder.build(mpnn_result["probs"], mpnn_result["alphabet"], mask=mask)
    fmap_path = fmap_builder.save(fmap)

    # 4. Audit summary
    print("\n================================================================================")
    print("                PHASE 2 PIPELINE AUDIT & INTEGRITY REVIEW")
    print("================================================================================")
    print(f"* Design mask:        {frozen_ct} frozen / {mutable_ct} mutable positions")
    print(f"* Feasibility map:    {fmap_path}")

    sample_mutable = next((k for k, v in fmap.items() if v["status"] == "MUTABLE"), None)
    if sample_mutable:
        top3 = list(fmap[sample_mutable]["probabilities"].items())[:3]
        print(f"* Sample [{sample_mutable}] (wt={fmap[sample_mutable]['wt']}): top candidates -> {top3}")
    print("================================================================================\n")


#  RUN
execute_pipeline_phase_2()
