"""validate_sequence_analyzer.py - Analyzes mutation patterns, physicochemical properties, and mask compliance across the Pareto front."""
import sys, os, json

import pandas as pd

MCTS_FRONT_PATH = "./pipeline_artifacts/phase4_pareto_front.json"
OUTPUT_DIR      = "./pipeline_artifacts"

# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY = {
    'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,
    'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,
    'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2
}
# Charge at pH 7.4 (simplified: R,K=+1; D,E=-1; H=+0.1)
CHARGE = {'R':1.0,'K':1.0,'H':0.1,'D':-1.0,'E':-1.0}

# Approximate residue molecular weights (Da)
MW = {
    'A':89,'R':174,'N':132,'D':133,'C':121,'Q':146,'E':147,'G':75,
    'H':155,'I':131,'L':131,'K':146,'M':149,'F':165,'P':115,'S':105,
    'T':119,'W':204,'Y':181,'V':117
}

def sequence_properties(seq):
    hydro  = sum(HYDROPHOBICITY.get(aa,0) for aa in seq) / len(seq)
    charge = sum(CHARGE.get(aa,0) for aa in seq)
    mw     = sum(MW.get(aa,0) for aa in seq)
    return {"hydrophobicity": round(hydro,3),
            "net_charge":     round(charge,2),
            "molecular_weight_Da": mw}

def find_mutations(sequence, wild_type=WILD_TYPE):
    return [f"{wt}{i+1}{aa}"
            for i,(aa,wt) in enumerate(zip(sequence, wild_type))
            if aa != wt]

def run_sequence_analysis():
    print("="*60)
    print("  VALIDATOR: Sequence Analysis")
    print("="*60)

    with open(MCTS_FRONT_PATH) as f:
        data = json.load(f)
    front = data["pareto_front"]
    mask  = load_mask()

    rows = []
    violations_found = False

    for cand in front:
        seq   = cand["sequence"]
        valid, violations = validate_sequence(seq, mask)
        if not valid:
            print(f"[WARNING] Mask violation in candidate: {violations}")
            violations_found = True

        mutations = find_mutations(seq)
        props     = sequence_properties(seq)
        rows.append({
            "sequence":         seq,
            "structural_score": round(cand["structural_score"], 4),
            "mpb_score":        round(cand["mpb_score"], 4),
            "generation":       cand.get("generation","?"),
            "n_mutations":      len(mutations),
            "mutations":        ", ".join(mutations) if mutations else "wild-type",
            "mask_valid":       valid,
            **props,
        })

    df = pd.DataFrame(rows).sort_values("mpb_score")

    csv_path = os.path.join(OUTPUT_DIR, "phase4_pareto_analysis.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n  Pareto front candidates: {len(df)}")
    print(f"  Mask violations: {'NONE' if not violations_found else 'SEE ABOVE'}")
    print(f"\n  Top 5 by lowest MPB (cheapest to manufacture):")
    print(df[["mutations","structural_score","mpb_score","net_charge","hydrophobicity"]].head().to_string(index=False))
    print(f"\n  Top 5 by highest structural score:")
    print(df.sort_values("structural_score",ascending=False)[
        ["mutations","structural_score","mpb_score","net_charge","hydrophobicity"]
    ].head().to_string(index=False))
    print(f"\n  Full table saved to: {csv_path}")

    return df


#  RUN
run_sequence_analysis()
