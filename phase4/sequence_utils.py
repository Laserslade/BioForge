"""sequence_utils.py - Shared helper functions for mutating, scoring, and comparing candidate sequences."""
import json, math, random
from collections import Counter

FEASIBILITY_MAP_PATH = "./pipeline_artifacts/phase2_feasibility_map.json"
MASK_PATH            = "./pipeline_artifacts/phase2_mask.json"
# >>> PEPTIDE_SPECIFIC: replace for a new target peptide
WILD_TYPE            = "MCMPCFTTDHQMARKCDDCCGGKGRGKCYGPQCLCR"
# <<< END_PEPTIDE_SPECIFIC

def load_feasibility_map(path=FEASIBILITY_MAP_PATH):
    with open(path) as f: return json.load(f)

def load_mask(path=MASK_PATH):
    with open(path) as f: return json.load(f)

def get_mutable_positions(mask):
    return sorted([int(k.split("_")[1]) for k,v in mask.items() if v["status"]=="MUTABLE"])

def compute_structural_score(sequence, feasibility_map):
    """Sum of log P(aa | backbone) at mutable positions. Higher = better."""
    score = 0.0
    for i, aa in enumerate(sequence):
        entry = feasibility_map.get(f"pos_{i}")
        if entry and entry["status"] == "MUTABLE":
            score += math.log(max(entry["probabilities"].get(aa, 1e-10), 1e-10))
    return score

def sample_sequence(feasibility_map, mask, wild_type=WILD_TYPE):
    seq = list(wild_type)
    for i in range(len(wild_type)):
        entry = feasibility_map.get(f"pos_{i}")
        if entry and entry["status"] == "MUTABLE":
            probs = entry["probabilities"]
            aas, weights = list(probs.keys()), list(probs.values())
            seq[i] = random.choices(aas, weights=weights, k=1)[0]
    return "".join(seq)

def apply_mutation(sequence, position, new_residue):
    seq = list(sequence); seq[position] = new_residue; return "".join(seq)

def validate_sequence(sequence, mask, wild_type=WILD_TYPE):
    violations = [
        f"pos_{i}: frozen {wt} -> {aa}"
        for i,(aa,wt) in enumerate(zip(sequence, wild_type))
        if mask[f"pos_{i}"]["status"]=="FROZEN" and aa!=wt
    ]
    return len(violations)==0, violations

def dominates(a, b):
    not_worse = (a["structural_score"] >= b["structural_score"] and
                 a["mpb_score"] <= b["mpb_score"])
    strictly_better = (a["structural_score"] > b["structural_score"] or
                       a["mpb_score"] < b["mpb_score"])
    return not_worse and strictly_better

def compute_population_entropy(population, mutable_positions):
    """
    Shannon entropy of amino acid distribution at each mutable position
    across the current population. Returns mean entropy in bits across
    all mutable positions.

    Maximum possible entropy = log2(20) ≈ 4.32 bits (all 20 AAs equally
    likely at every position). Low entropy = homogeneous population.

    Used by NSGA2Engine to detect stagnation: if entropy falls below a
    threshold AND fitness has plateaued, the adaptive boost fires.
    """
    if not population or not mutable_positions:
        return 0.0

    entropies = []
    for pos in mutable_positions:
        counts = Counter(c["sequence"][pos] for c in population)
        total  = sum(counts.values())
        h = -sum((n/total) * math.log2(n/total) for n in counts.values() if n > 0)
        entropies.append(h)

    return sum(entropies) / len(entropies)
