"""existence_analysis.py - Runs five independent lines of evidence testing whether a dual-objective-superior variant exists in the design space."""

import os, json, math, random, glob
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Constants

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX   = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# >>> PEPTIDE_SPECIFIC: replace for a new target peptide
# Wild-type chlorotoxin (1CHL chain A, 36 residues)
WILD_TYPE = "MCMPCFTTDHQMARRCDDCCGGKGRGKCYGPQCLCR"
# <<< END_PEPTIDE_SPECIFIC

# WT MPB from the Phase 4 Enhanced run (growth_percent_reduction, %)
# If your run produced a different value, update this.
WT_MPB_PERCENT = 3.5199

# >>> PEPTIDE_SPECIFIC: replace for a new target peptide
# Known positions that vary across natural chlorotoxin-family homologs.
# <<< END_PEPTIDE_SPECIFIC
# These are the positions where evolution has already accepted substitutions
HOMOLOG_VARIABLE_POSITIONS = {
    0:  set("ML"),       # M/L - N-terminal variability
    5:  set("FY"),       # F/Y - aromatic, structurally equivalent
    7:  set("TD"),       # T/D - polar substitutions seen
    8:  set("TAS"),      # T/A/S - small polar
    9:  set("DEN"),      # D/E/N - acidic/amide
    12: set("MAL"),      # M/A/L - hydrophobic
    13: set("AR"),       # A/R - flexible position
    24: set("GS"),       # G/S - glycine loop, small residues tolerated
    26: set("KR"),       # K/R - basic, functionally equivalent
    28: set("CY"),       # C/Y - in some homologs CY (non-disulfide isoform)
    31: set("PAS"),      # P/A/S - proline bend tolerance
    32: set("QE"),       # Q/E - polar
}

# Amino acid biosynthetic cost in ATP equivalents
# Source: Akashi & Gojobori (2002) PNAS, Table 1
AA_BIOSYNTHETIC_COST = {
    'G': 11.7, 'A': 11.7, 'S': 11.7, 'D': 12.7, 'C': 24.7,
    'T': 18.7, 'P': 20.3, 'E': 15.3, 'Q': 16.2, 'N': 14.7,
    'V': 23.3, 'L': 27.3, 'I': 32.3, 'M': 34.3, 'H': 38.3,
    'F': 52.0, 'Y': 50.0, 'W': 74.3, 'K': 30.3, 'R': 27.3,
}

# ─────────────────────────────────────────────────────────────────────────────
# Artifact Loading

def _load_json(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[ERROR] {label} not found at: {path}\n"
            f"        Make sure Phase 2 and Phase 4 Enhanced have been run first."
        )
    with open(path) as f:
        return json.load(f)

def load_artifacts():
    """Load mask, feasibility map, Pareto front, and any checkpoint candidates."""
    mask   = _load_json("./pipeline_artifacts/phase2_mask.json",
                        "Phase 2 mask")
    fmap   = _load_json("./pipeline_artifacts/phase2_feasibility_map.json",
                        "Phase 2 feasibility map")
    raw    = _load_json("./pipeline_artifacts/phase4_enhanced_pareto_front.json",
                        "Phase 4 Enhanced Pareto front")

    pareto_front = raw.get("pareto_front", raw if isinstance(raw, list) else [])

    mutable_positions = sorted([
        int(k.split("_")[1])
        for k, v in fmap.items()
        if v.get("status") == "MUTABLE"
    ])

    # Aggregate candidates across all checkpoints for a richer dataset
    all_candidates = list(pareto_front)
    seen_seqs      = {c["sequence"] for c in all_candidates}

    ckpt_dir = "./phase4_checkpoints/enhanced"
    if os.path.isdir(ckpt_dir):
        for ckpt_path in sorted(glob.glob(os.path.join(ckpt_dir, "*.json"))):
            try:
                ckpt = json.loads(open(ckpt_path).read())
                for c in ckpt.get("pareto_front", []):
                    if c.get("sequence") and c["sequence"] not in seen_seqs:
                        all_candidates.append(c)
                        seen_seqs.add(c["sequence"])
            except Exception:
                pass

    print(f"[LOAD] Mutable positions:  {len(mutable_positions)}")
    print(f"[LOAD] Final Pareto front: {len(pareto_front)} candidates")
    print(f"[LOAD] Total unique evaluated (incl. checkpoints): {len(all_candidates)}")

    return mask, fmap, pareto_front, all_candidates, mutable_positions


# ─────────────────────────────────────────────────────────────────────────────
# Wild-Type Baseline

def compute_wt_scores(fmap, mutable_positions):
    """
    Replicate compute_structural_score(WILD_TYPE, fmap) exactly as the
    engine does - sum of log P(wt_aa | backbone) at mutable positions only.
    """
    wt_struct = 0.0
    for pos in mutable_positions:
        key   = f"pos_{pos}"
        entry = fmap.get(key, {})
        probs = entry.get("probabilities", {})
        aa    = WILD_TYPE[pos] if pos < len(WILD_TYPE) else "G"
        p     = max(probs.get(aa, 1e-10), 1e-10)
        wt_struct += math.log(p)

    wt_mpb = WT_MPB_PERCENT   # from Phase 4 Enhanced run output
    return wt_struct, wt_mpb


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 0: Direct Pareto Check

def direct_pareto_check(pareto_front, wt_struct, wt_mpb):
    """
    Fastest possible existence proof: does the current Pareto front already
    contain a candidate that beats WT on BOTH axes?
    """
    print("\n" + "=" * 62)
    print("CHECK 0: Direct Pareto Front Inspection")
    print("=" * 62)
    print(f"  Wild-type structural score : {wt_struct:.4f}")
    print(f"  Wild-type MPB              : {wt_mpb:.4f}%")
    print()

    dominates_wt = []
    struct_scores = []
    mpb_scores    = []

    for c in pareto_front:
        s = c.get("structural_score", -999)
        m = c.get("mpb_score", 999)
        struct_scores.append(s)
        mpb_scores.append(m)
        if s > wt_struct and m <= wt_mpb:
            dominates_wt.append(c)

    best_struct = max(struct_scores) if struct_scores else None
    best_mpb    = min(mpb_scores)    if mpb_scores    else None

    print(f"  Best structural score in front : {best_struct:.4f}  "
          f"({'BETTER' if best_struct and best_struct > wt_struct else 'WORSE'} than WT)")
    print(f"  Best MPB in front              : {best_mpb:.4f}%  "
          f"({'BETTER' if best_mpb and best_mpb < wt_mpb else 'WORSE'} than WT)")
    print()

    if dominates_wt:
        print(f"   EMPIRICAL PROOF: {len(dominates_wt)} candidate(s) in the current Pareto")
        print(f"    front already beat wild-type on BOTH objectives.")
        print(f"    The solution exists - it has been found.")
        for c in dominates_wt[:3]:
            print(f"     seq: {c['sequence']}  "
                  f"struct={c['structural_score']:.3f}  mpb={c['mpb_score']:.4f}%")
    else:
        print(f"   No Pareto candidate dominates WT on both axes yet.")
        print(f"    Struct gap to close:  {wt_struct - (best_struct or wt_struct):.4f} units")
        print(f"    MPB already better:   {wt_mpb - (best_mpb or wt_mpb):.4f}% improvement")
        print(f"    The search is improving on both axes but hasn't converged to")
        print(f"    a single candidate that simultaneously beats WT on both.")

    return {"dominates_wt": dominates_wt, "best_struct": best_struct, "best_mpb": best_mpb}


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 1: Basin Sampling

def basin_sampling_analysis(fmap, mutable_positions, wt_struct, wt_mpb,
                            n_samples=10_000):
    """
    Monte Carlo over the ProteinMPNN feasibility prior.

    Estimates P(struct > WT  AND  biosynthetic_cost ≤ WT_cost), where cost
    is used as a proxy for MPB because we cannot run FBA on each sample.
    The proxy is conservative - if anything it underestimates overlap,
    since WT's sequence was not selected for E. coli expression efficiency.
    """
    print("\n" + "=" * 62)
    print("ANALYSIS 1: Basin Sampling  (Monte Carlo, n={:,})".format(n_samples))
    print("=" * 62)

    wt_cost = sum(AA_BIOSYNTHETIC_COST.get(aa, 30.0) for aa in WILD_TYPE)

    n_better_struct = 0
    n_cheaper       = 0
    n_both          = 0
    all_struct      = []
    all_cost_ratio  = []

    for _ in range(n_samples):
        seq          = list(WILD_TYPE)
        struct_score = 0.0
        sample_cost  = 0.0

        for pos in mutable_positions:
            entry = fmap.get(f"pos_{pos}", {})
            probs = entry.get("probabilities", {})
            aas   = list(probs.keys())
            ws    = [max(probs[a], 1e-10) for a in aas]
            total = sum(ws)
            ws    = [w / total for w in ws]
            chosen = random.choices(aas, weights=ws, k=1)[0]
            seq[pos] = chosen

        seq_str = "".join(seq)

        # Structural score (same formula as engine)
        for pos in mutable_positions:
            entry = fmap.get(f"pos_{pos}", {})
            probs = entry.get("probabilities", {})
            p     = max(probs.get(seq_str[pos], 1e-10), 1e-10)
            struct_score += math.log(p)

        # Biosynthetic cost (WT-normalised)
        sample_cost = sum(AA_BIOSYNTHETIC_COST.get(aa, 30.0) for aa in seq_str)
        cost_ratio  = sample_cost / wt_cost   # <1.0 means cheaper than WT

        all_struct.append(struct_score)
        all_cost_ratio.append(cost_ratio)

        if struct_score > wt_struct:
            n_better_struct += 1
        if cost_ratio <= 1.0:
            n_cheaper += 1
        if struct_score > wt_struct and cost_ratio <= 1.0:
            n_both += 1

    p_struct = n_better_struct / n_samples
    p_cheap  = n_cheaper       / n_samples
    p_both   = n_both          / n_samples

    mean_struct = float(np.mean(all_struct))
    std_struct  = float(np.std(all_struct))
    mean_ratio  = float(np.mean(all_cost_ratio))

    print(f"  P(structural_score > WT) under feasibility prior : {p_struct:.4f}")
    print(f"  P(biosyn. cost ≤ WT)     under feasibility prior : {p_cheap:.4f}")
    print(f"  P(both simultaneously)                           : {p_both:.4f}  "
          f"({n_both:,} / {n_samples:,} samples)")
    print()
    print(f"  Mean sampled structural score : {mean_struct:.3f}  (σ={std_struct:.3f})")
    print(f"  Mean biosyn. cost ratio vs WT : {mean_ratio:.4f}  "
          f"({'cheaper on average' if mean_ratio < 1 else 'more expensive on average'})")
    print()

    # Independence test: if the two events were uncorrelated,
    # P(both) ≈ P(struct) × P(cheap).  Actual vs expected ratio shows correlation.
    p_if_independent = p_struct * p_cheap
    if p_if_independent > 0:
        lift = p_both / p_if_independent
        print(f"  Correlation lift (actual / if-independent): {lift:.3f}")
        if lift > 1.1:
            print(f"   The two objectives are POSITIVELY correlated under this prior.")
            print(f"    Sequences that score well structurally also tend to be cheaper.")
        elif lift < 0.9:
            print(f"   The two objectives are NEGATIVELY correlated (antagonistic).")
        else:
            print(f"   The two objectives are approximately independent.")

    if n_both > 0:
        print(f"\n   {n_both:,} feasibility-prior samples satisfy both criteria.")
        print(f"    The target region is non-empty - a solution exists in principle.")
    else:
        print(f"\n   No overlap found in {n_samples:,} samples.")
        print(f"    Either objectives are strongly antagonistic, or the proxy is too coarse.")

    return {"p_struct": p_struct, "p_cheap": p_cheap, "p_both": p_both,
            "n_samples": n_samples, "n_both": n_both,
            "mean_struct": mean_struct, "std_struct": std_struct}


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 2: IVT / Lipschitz Argument

def ivt_lipschitz_argument(all_candidates, wt_struct, wt_mpb):
    """
    Intermediate Value Theorem existence argument.

    If f (structural score) and g (MPB score) are Lipschitz continuous
    in Hamming distance, and there exists:
      - candidate A with f(A) > f(WT)     [better structure]
      - candidate B with g(B) ≤ g(WT)     [better or equal MPB]

    then by IVT there is a point along any Hamming path from A to B where
    both f > f(WT) and g ≤ g(WT) hold simultaneously - provided the score
    functions don't oscillate faster than the Lipschitz bound allows.

    We estimate the empirical Lipschitz constant L from observed pairwise
    score differences / Hamming distances in the evaluated population.
    """
    print("\n" + "=" * 62)
    print("ANALYSIS 2: IVT / Lipschitz Continuity Argument")
    print("=" * 62)

    valid = [c for c in all_candidates
             if c.get("sequence") and len(c["sequence"]) == len(WILD_TYPE)
             and c.get("structural_score") is not None
             and c.get("mpb_score") is not None]

    if not valid:
        print("  [SKIP] No valid candidates available.")
        return {}

    best_struct_cand = max(valid, key=lambda c: c["structural_score"])
    best_mpb_cand    = min(valid, key=lambda c: c["mpb_score"])

    bs = best_struct_cand["structural_score"]
    bm = best_mpb_cand["mpb_score"]

    print(f"  Best structural score : {bs:.4f}  "
          f"(WT: {wt_struct:.4f}  |  diff: {bs - wt_struct:+.4f})")
    print(f"  Best MPB score        : {bm:.4f}%  "
          f"(WT: {wt_mpb:.4f}%  |  diff: {bm - wt_mpb:+.4f}%)")
    print()

    struct_beats_wt = bs > wt_struct
    mpb_beats_wt    = bm < wt_mpb

    # Empirical Lipschitz constant from random pairs
    sample_size = min(300, len(valid))
    sample      = random.sample(valid, sample_size)

    L_struct_obs, L_mpb_obs = [], []
    for i in range(len(sample) - 1):
        a, b = sample[i], sample[i + 1]
        sa, sb = a["sequence"], b["sequence"]
        hamming = sum(x != y for x, y in zip(sa, sb))
        if hamming == 0:
            continue
        L_struct_obs.append(abs(a["structural_score"] - b["structural_score"]) / hamming)
        L_mpb_obs.append(abs(a["mpb_score"] - b["mpb_score"]) / hamming)

    if not L_struct_obs:
        print("  [SKIP] Not enough sequence diversity to estimate Lipschitz constant.")
        return {}

    # Use 90th percentile as a conservative upper bound
    L_struct = float(np.percentile(L_struct_obs, 90))
    L_mpb    = float(np.percentile(L_mpb_obs,    90))

    seq_a = best_struct_cand["sequence"]
    seq_b = best_mpb_cand["sequence"]
    d_AB  = sum(x != y for x, y in zip(seq_a, seq_b))

    # Worst-case score floor along AB path
    struct_floor = bs  - L_struct * d_AB
    mpb_ceiling  = bm  + L_mpb    * d_AB   # MPB can rise by at most L_mpb * d

    print(f"  Empirical Lipschitz constant (struct) : {L_struct:.4f} score/mutation  [90th pct]")
    print(f"  Empirical Lipschitz constant (MPB)    : {L_mpb:.4f} %/mutation        [90th pct]")
    print(f"  Hamming distance between best-struct and best-MPB : {d_AB}")
    print()
    print(f"  Worst-case structural floor along AB : {struct_floor:.4f}  (must exceed {wt_struct:.4f})")
    print(f"  Worst-case MPB ceiling  along AB     : {mpb_ceiling:.4f}%  (must stay ≤ {wt_mpb:.4f}%)")
    print()

    if struct_floor > wt_struct and mpb_ceiling <= wt_mpb:
        verdict = "STRONG"
        msg = ("Both score functions remain above/below WT thresholds along\n"
               "    the entire path. A dual-objective superior sequence provably\n"
               "    exists somewhere between these two candidates.")
    elif struct_beats_wt and mpb_beats_wt:
        verdict = "MODERATE"
        msg = ("Both axes have found candidates beating WT, but the Lipschitz\n"
               "    bound doesn't guarantee scores stay above WT along the full path.\n"
               "    IVT still implies a crossing point; the score may dip briefly\n"
               "    below WT in the interior. A focused local search should find it.")
    elif struct_beats_wt or mpb_beats_wt:
        verdict = "WEAK"
        msg = ("Only one axis currently beats WT. IVT argument is one-sided.\n"
               "    More generations are needed for the other axis to cross WT.")
    else:
        verdict = "NOT MET"
        msg = "Neither axis beats WT yet - IVT conditions not met."

    print(f"  IVT verdict: {verdict}")
    print(f"    {msg}")

    return {
        "struct_beats_wt": struct_beats_wt, "mpb_beats_wt": mpb_beats_wt,
        "L_struct": L_struct, "L_mpb": L_mpb, "d_AB": d_AB,
        "struct_floor": struct_floor, "mpb_ceiling": mpb_ceiling,
        "verdict": verdict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 3: Gaussian Process Surrogate

def gp_surrogate_analysis(all_candidates, mutable_positions, wt_struct, wt_mpb):
    """
    Fits independent GPs to structural_score and mpb_score as functions of
    sequence. Encodes sequences as binary one-hot vectors at mutable positions.

    Queries P(struct > WT AND mpb ≤ WT) across the GP posterior via Monte
    Carlo sampling of the predictive distribution. Returns a point estimate
    and 95% credible interval.
    """
    print("\n" + "=" * 62)
    print("ANALYSIS 3: Gaussian Process Surrogate Model")
    print("=" * 62)

    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
    except ImportError:
        print("  [SKIP] scikit-learn not installed.  Run:  !pip install scikit-learn")
        return {}

    valid = [c for c in all_candidates
             if c.get("sequence") and len(c["sequence"]) == len(WILD_TYPE)
             and c.get("structural_score") is not None
             and c.get("mpb_score") is not None]

    if len(valid) < 10:
        print(f"  [SKIP] Only {len(valid)} valid candidates - need ≥10 for a meaningful GP.")
        return {}

    print(f"  Training GP on {len(valid)} evaluated sequences...")

    def encode(seq):
        """One-hot encoding at mutable positions  vector of length n_mut × 20."""
        vec = []
        for pos in mutable_positions:
            aa  = seq[pos] if pos < len(seq) else "G"
            vec.extend([1.0 if AMINO_ACIDS[i] == aa else 0.0 for i in range(20)])
        return vec

    X        = np.array([encode(c["sequence"])       for c in valid])
    y_struct = np.array([c["structural_score"]        for c in valid])
    y_mpb    = np.array([c["mpb_score"]               for c in valid])
    X_wt     = np.array([encode(WILD_TYPE)])

    kernel = C(1.0, (1e-3, 1e3)) * Matern(nu=2.5)

    gp_s = GaussianProcessRegressor(kernel=kernel, alpha=0.1,  n_restarts_optimizer=5)
    gp_m = GaussianProcessRegressor(kernel=kernel, alpha=0.01, n_restarts_optimizer=5)

    gp_s.fit(X, y_struct)
    gp_m.fit(X, y_mpb)

    wt_s_pred, wt_s_std = gp_s.predict(X_wt, return_std=True)
    wt_m_pred, wt_m_std = gp_m.predict(X_wt, return_std=True)

    print(f"  GP struct at WT:  predicted {wt_s_pred[0]:.3f} ± {wt_s_std[0]:.3f}  "
          f"(actual: {wt_struct:.3f})")
    print(f"  GP MPB at WT:     predicted {wt_m_pred[0]:.4f} ± {wt_m_std[0]:.4f}%  "
          f"(actual: {wt_mpb:.4f}%)")
    print()

    # Predict across the top-struct candidates (most likely neighbourhood
    # where dual improvement exists)
    top_idx = np.argsort(y_struct)[-min(200, len(valid)):]
    X_test  = X[top_idx]

    mu_s, sig_s = gp_s.predict(X_test, return_std=True)
    mu_m, sig_m = gp_m.predict(X_test, return_std=True)

    # Monte Carlo over GP posterior uncertainty
    n_mc            = 2000
    p_exists_trials = []
    for _ in range(n_mc):
        s_s = mu_s + sig_s * np.random.randn(len(X_test))
        s_m = mu_m + sig_m * np.random.randn(len(X_test))
        p_exists_trials.append(float(np.any((s_s > wt_struct) & (s_m <= wt_mpb))))

    p_exists  = float(np.mean(p_exists_trials))
    ci_lo     = float(np.percentile(p_exists_trials, 2.5))
    ci_hi     = float(np.percentile(p_exists_trials, 97.5))

    print(f"  P(∃ candidate: struct > WT AND mpb ≤ WT) - GP posterior:")
    print(f"    Point estimate : {p_exists:.3f}")
    print(f"    95% CI         : [{ci_lo:.3f},  {ci_hi:.3f}]")
    print()

    if p_exists > 0.7:
        verdict = f" HIGH ({p_exists*100:.0f}%)"
        msg     = "GP strongly supports existence of a dual-superior solution."
    elif p_exists > 0.4:
        verdict = f"~ MODERATE ({p_exists*100:.0f}%)"
        msg     = ("GP uncertainty is high. More evaluations would tighten the CI.\n"
                   "    Existence is plausible but not firmly established yet.")
    else:
        verdict = f" LOW ({p_exists*100:.0f}%)"
        msg     = ("GP doesn't support existence in the tested neighbourhood.\n"
                   "    The dual-objective region may be outside what the current\n"
                   "    population has explored.")

    print(f"  GP verdict: {verdict}")
    print(f"    {msg}")

    return {"p_exists": p_exists, "ci": (ci_lo, ci_hi)}


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 4: Biological Prior - Homolog Comparison

def biological_prior_analysis(pareto_front, all_candidates, mutable_positions):
    """
    Checks whether the optimizer independently found mutations at positions
    # >>> PEPTIDE_SPECIFIC: replace for a new target peptide
    that natural evolution has already accepted in chlorotoxin-family homologs.
    # <<< END_PEPTIDE_SPECIFIC

    Independent rediscovery is strong evidence the optimizer is finding
    real structural biology, not just gaming the ProteinMPNN scoring function.

    NOTE: HOMOLOG_VARIABLE_POSITIONS at the top of this file is derived from
    structural alignment literature. Verify entries against UniProt / PDB
    before citing in a paper.
    """
    print("\n" + "=" * 62)
    print("ANALYSIS 4: Biological Prior - Homolog Position Comparison")
    print("=" * 62)
    print("  (Homolog positions are approximate - verify vs UniProt before publishing)")
    print()

    # Gather all mutations found in Pareto front
    opt_mutations = {}   # (pos, aa)  count
    for c in pareto_front:
        seq = c.get("sequence", "")
        if not seq or len(seq) != len(WILD_TYPE):
            continue
        for pos in mutable_positions:
            wt_aa  = WILD_TYPE[pos]
            opt_aa = seq[pos]
            if opt_aa != wt_aa:
                key = (pos, opt_aa)
                opt_mutations[key] = opt_mutations.get(key, 0) + 1

    # Compare against known variable positions
    known_variable = set(HOMOLOG_VARIABLE_POSITIONS.keys()) & set(mutable_positions)
    opt_mut_set    = set(opt_mutations.keys())

    # Rediscovered = optimizer mutated a position that varies in nature
    #              AND chose a residue actually seen in natural homologs
    rediscovered_positions  = set()
    rediscovered_residues   = set()
    novel_positions         = set()

    for (pos, aa), count in opt_mutations.items():
        if pos in HOMOLOG_VARIABLE_POSITIONS:
            rediscovered_positions.add(pos)
            if aa in HOMOLOG_VARIABLE_POSITIONS[pos]:
                rediscovered_residues.add((pos, aa))
        else:
            novel_positions.add(pos)

    print(f"  Unique (position, residue) mutations in Pareto front  : {len(opt_mut_set)}")
    print(f"  Mutable positions overlapping with known variable sites: "
          f"{len(rediscovered_positions)} / {len(known_variable)}")
    print(f"  Optimizer chose a naturally-seen residue at those sites : "
          f"{len(rediscovered_residues)}")
    print(f"  Positions mutated that are novel (not in homolog set)   : "
          f"{len(novel_positions)}")
    print()

    if rediscovered_residues:
        print("  Exact natural-homolog rediscoveries:")
        for pos, aa in sorted(rediscovered_residues):
            count  = opt_mutations.get((pos, aa), 0)
            wt_aa  = WILD_TYPE[pos]
            nat_aa = sorted(HOMOLOG_VARIABLE_POSITIONS[pos] - {wt_aa})
            print(f"    pos {pos:2d}: {wt_aa}{aa}  "
                  f"(seen in {count}/{len(pareto_front)} Pareto candidates; "
                  f"natural variants at this site: {nat_aa})")
        print()

    # Hypergeometric significance test
    try:
        from scipy.stats import hypergeom
        N = len(mutable_positions) * 19      # total possible (pos,aa) pairs
        K = sum(len(v) - 1 for v in HOMOLOG_VARIABLE_POSITIONS.values()
                if any(p in mutable_positions
                       for p in HOMOLOG_VARIABLE_POSITIONS))  # natural non-WT variants
        n = len(opt_mut_set)                 # optimizer choices
        k = len(rediscovered_residues)       # overlap
        if K > 0 and n > 0:
            p_val = hypergeom.sf(max(k - 1, 0), N, K, n)
            print(f"  Hypergeometric p-value (overlap by chance): {p_val:.4f}")
            if p_val < 0.05:
                print("   Statistically significant overlap (p < 0.05).")
                print("    Optimizer is independently recovering biologically meaningful")
                print("    substitutions, not just fitting the structural scoring function.")
            elif p_val < 0.20:
                print("  ~ Marginal significance (p < 0.20). Suggestive but not conclusive.")
            else:
                print("  ~ Not statistically significant. Overlap may be coincidental,")
                print("    or the homolog set is too small to capture full natural variation.")
    except ImportError:
        print("  [INFO] scipy not available for significance test.  pip install scipy")

    if novel_positions:
        print()
        print(f"  Novel positions (not in homolog variable set): "
              f"{sorted(novel_positions)}")
        print("  These may be genuinely new design space - or artifacts of the")
        print("  ProteinMPNN scoring function. ESMFold validation would clarify.")

    return {
        "opt_mutations":          opt_mutations,
        "rediscovered_positions": rediscovered_positions,
        "rediscovered_residues":  rediscovered_residues,
        "novel_positions":        novel_positions,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary Report

def print_summary(direct, basin, ivt, gp, bio):
    print()
    print("=" * 62)
    print("  EXISTENCE ANALYSIS - SUMMARY REPORT")
    print("=" * 62)
    print()
    # >>> PEPTIDE_SPECIFIC: replace for a new target peptide
    print("  QUESTION: Does a chlorotoxin variant exist that simultaneously")
    # <<< END_PEPTIDE_SPECIFIC
    print("  beats wild-type on structural score AND metabolic burden?")
    print()

    evidence = []

    # Direct check
    if direct.get("dominates_wt"):
        n = len(direct["dominates_wt"])
        evidence.append(("", f"DIRECT:    {n} Pareto candidate(s) already beat WT on both axes."))
    else:
        bs = direct.get("best_struct")
        bm = direct.get("best_mpb")
        if bs is not None and bm is not None:
            evidence.append(("~", f"DIRECT:    No single candidate beats WT on both yet "
                                   f"(struct gap: {abs(bs):.1f} vs WT {abs(direct.get('wt_struct', bs)):.1f}; "
                                   f"MPB already {bm:.2f}% vs WT)."))
        else:
            evidence.append(("", "DIRECT:    Could not evaluate (missing score data)."))

    # Basin sampling
    p_both = basin.get("p_both", 0)
    n_both = basin.get("n_both", 0)
    ns     = basin.get("n_samples", 0)
    if p_both > 0:
        evidence.append(("", f"SAMPLING:  {n_both:,}/{ns:,} feasibility-prior samples "
                               f"satisfy both criteria (p={p_both:.4f})."))
    else:
        evidence.append(("", f"SAMPLING:  No overlap in {ns:,} Monte Carlo samples."))

    # IVT
    v = ivt.get("verdict", "NOT MET")
    if v == "STRONG":
        evidence.append(("", f"IVT:       Strong - scores stay above/below WT along entire AB path."))
    elif v == "MODERATE":
        evidence.append(("~", f"IVT:       Moderate - both axes beat WT individually; "
                               "crossing point implied."))
    elif v == "WEAK":
        evidence.append(("~", f"IVT:       Weak - only one axis beats WT so far."))
    else:
        evidence.append(("", f"IVT:       Conditions not met."))

    # GP
    p_ex = gp.get("p_exists")
    if p_ex is not None:
        ci = gp.get("ci", (0, 1))
        if p_ex > 0.7:
            evidence.append(("", f"GP:        P(exists) = {p_ex:.2f}  "
                                   f"[95% CI {ci[0]:.2f}-{ci[1]:.2f}]  - high confidence."))
        elif p_ex > 0.4:
            evidence.append(("~", f"GP:        P(exists) = {p_ex:.2f}  "
                                   f"[95% CI {ci[0]:.2f}-{ci[1]:.2f}]  - moderate confidence."))
        else:
            evidence.append(("", f"GP:        P(exists) = {p_ex:.2f}  - low confidence."))
    else:
        evidence.append(("~", "GP:        Skipped (scikit-learn not available or insufficient data)."))

    # Biological
    nr = len(bio.get("rediscovered_residues", set()))
    if nr > 0:
        evidence.append(("", f"BIOLOGY:   Optimizer independently rediscovered {nr} naturally-"
                               "occurring substitution(s) from homolog alignment."))
    else:
        evidence.append(("~", "BIOLOGY:   No exact natural-homolog rediscoveries detected."))

    n_positive = sum(1 for mark, _ in evidence if mark == "")

    for mark, text in evidence:
        print(f"  {mark}  {text}")

    print()
    print(f"  Lines of evidence: {n_positive}/{len(evidence)} positive")
    print()

    if n_positive >= 4:
        print("  ═══════════════════════════════════════════════════════")
        print("  CONCLUSION: Strong convergent evidence.")
        print("  A structurally valid, metabolically competitive variant")
        # >>> PEPTIDE_SPECIFIC: replace for a new target peptide
        print("  of chlorotoxin almost certainly exists. The optimizer")
        # <<< END_PEPTIDE_SPECIFIC
        print("  may have already found one - inspect the Pareto front")
        print("  direct check above. If not yet, it is very close.")
        print("  Recommended: run ESMFold on the top 5 Pareto candidates")
        print("  to confirm the structural score proxy reflects real fold.")
        print("  ═══════════════════════════════════════════════════════")
    elif n_positive >= 2:
        print("  ═══════════════════════════════════════════════════════")
        print("  CONCLUSION: Moderate evidence.")
        print("  The solution likely exists but hasn't been pinpointed.")
        print("  Consider: larger population (pop=60+), more generations,")
        print("  or adding MPB >= WT as a hard constraint rather than a")
        print("  free objective to force the optimizer into that region.")
        print("  ═══════════════════════════════════════════════════════")
    else:
        print("  ═══════════════════════════════════════════════════════")
        print("  CONCLUSION: Weak evidence. The objectives may be more")
        print("  antagonistic than expected, or the structural score")
        print("  proxy is drifting from actual fold quality. ESMFold")
        print("  validation and objective rebalancing are recommended.")
        print("  ═══════════════════════════════════════════════════════")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point

def run_existence_analysis(seed=42):
    print("=" * 62)
    print("  BioForge - Existence Analysis")
    # >>> PEPTIDE_SPECIFIC: replace for a new target peptide
    print("  Chlorotoxin Structural + Metabolic Optimization")
    # <<< END_PEPTIDE_SPECIFIC
    print("=" * 62)

    random.seed(seed)
    np.random.seed(seed)

    mask, fmap, pareto_front, all_candidates, mutable_positions = load_artifacts()
    wt_struct, wt_mpb = compute_wt_scores(fmap, mutable_positions)

    print(f"\n  Wild-type structural score : {wt_struct:.4f}")
    print(f"  Wild-type MPB              : {wt_mpb:.4f}%")

    direct = direct_pareto_check(pareto_front, wt_struct, wt_mpb)
    # Patch wt_struct into direct dict for summary
    direct["wt_struct"] = wt_struct

    basin  = basin_sampling_analysis(fmap, mutable_positions, wt_struct, wt_mpb)
    ivt    = ivt_lipschitz_argument(all_candidates, wt_struct, wt_mpb)
    gp     = gp_surrogate_analysis(all_candidates, mutable_positions, wt_struct, wt_mpb)
    bio    = biological_prior_analysis(pareto_front, all_candidates, mutable_positions)

    print_summary(direct, basin, ivt, gp, bio)


run_existence_analysis()
