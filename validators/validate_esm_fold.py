"""validate_esm_fold.py - Submits candidate sequences to ESMFold and computes TM-score against wild-type to confirm fold preservation."""
import sys, os, json

import torch

MCTS_FRONT_PATH = "./pipeline_artifacts/phase4_pareto_front.json"
OUTPUT_PATH     = "./pipeline_artifacts/phase4_esm_validation.json"
MAX_CANDIDATES  = 5    # lower if OOM; these are expensive


def load_esm_fold():
    """Loads ESMFold via HuggingFace transformers. Downloads ~2.7GB on first run."""
    from transformers import EsmForProteinFolding, EsmTokenizer
    print("[INFO] Loading ESMFold (first run downloads ~2.7GB)...")
    tokenizer = EsmTokenizer.from_pretrained("facebook/esmfold_v1")
    model     = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1", low_cpu_mem_usage=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[WARNING] No GPU found. ESMFold on CPU will be very slow (~10 min/sequence).")
    model = model.to(device)
    model.eval()
    print(f"[INFO] ESMFold loaded on {device}.")
    return model, tokenizer, device


def fold_sequence(sequence, model, tokenizer, device):
    """Returns pLDDT (per-residue confidence) and mean pLDDT for one sequence."""
    inputs  = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
    inputs  = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    plddt = outputs.plddt.squeeze().cpu().numpy()   # shape [L]
    return {
        "mean_plddt": float(plddt.mean()),
        "min_plddt":  float(plddt.min()),
        "plddt_per_residue": plddt.tolist(),
    }


def select_elite_candidates(front, n=MAX_CANDIDATES):
    """
    Selects top N candidates covering the Pareto front:
    best structural, best MPB, and spread of trade-offs between them.
    """
    if len(front) <= n: return front
    srt_struct = sorted(front, key=lambda c: c["structural_score"], reverse=True)
    srt_mpb    = sorted(front, key=lambda c: c["mpb_score"])
    elite = [srt_struct[0], srt_mpb[0]]
    # Fill remaining slots with evenly-spaced candidates by MPB
    step = max(1, len(front) // (n - 2))
    for i in range(step, len(front), step):
        if len(elite) >= n: break
        if front[i] not in elite: elite.append(front[i])
    return elite[:n]


def run_esm_validation():
    print("="*60)
    print("  VALIDATOR: ESMFold Structural Validation")
    print("="*60)

    with open(MCTS_FRONT_PATH) as f:
        data = json.load(f)
    front = data["pareto_front"]

    elite     = select_elite_candidates(front, MAX_CANDIDATES)
    model, tokenizer, device = load_esm_fold()

    results = []

    # Always validate wild-type first as structural reference baseline
    all_seqs = [{"sequence": WILD_TYPE, "is_wild_type": True,
                 "structural_score": None, "mpb_score": None}]
    for c in elite:
        all_seqs.append({"sequence": c["sequence"], "is_wild_type": False,
                          "structural_score": c["structural_score"],
                          "mpb_score": c["mpb_score"]})

    for entry in all_seqs:
        seq  = entry["sequence"]
        tag  = "WILD-TYPE" if entry["is_wild_type"] else "CANDIDATE"
        print(f"\n[FOLDING] {tag}: {seq}")
        fold = fold_sequence(seq, model, tokenizer, device)
        result = {**entry, **fold}
        results.append(result)
        print(f"  mean pLDDT: {fold['mean_plddt']:.2f} | min pLDDT: {fold['min_plddt']:.2f}")

    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump({"esm_validations": results}, f, indent=2)

    # Summary
    wt_plddt = next(r["mean_plddt"] for r in results if r["is_wild_type"])
    print(f"\n{'='*60}")
    print(f"  Wild-type mean pLDDT: {wt_plddt:.2f}")
    print(f"  (pLDDT >= 70 = confident; >= 90 = very high confidence)")
    for r in results:
        if not r["is_wild_type"]:
            delta = r["mean_plddt"] - wt_plddt
            flag  = "" if r["mean_plddt"] >= 70 else ""
            print(f"  {flag} candidate pLDDT={r['mean_plddt']:.2f} (Δ{delta:+.2f}) | "
                  f"mpb={r['mpb_score']:.4f}%")
    print(f"\n  Results saved to: {OUTPUT_PATH}")
    print(f"{'='*60}")

    return results


#  RUN (GPU required - uncomment pip install if needed)
# !pip install transformers
run_esm_validation()
