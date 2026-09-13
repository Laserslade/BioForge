"""fba_evaluator.py - Runs flux balance analysis to score a candidate sequence's metabolic production burden."""

try:
    from .peptide_demand_builder import DEMAND_REACTION_ID, build_demand_reaction
except ImportError:  # pragma: no cover - direct script execution fallback
    from peptide_demand_builder import DEMAND_REACTION_ID, build_demand_reaction


class FBAEvaluatorError(Exception):
    pass


# How much peptide-equivalent flux to force through the demand reaction.
PEPTIDE_DEMAND_FLUX = 0.01


def evaluate_candidate(model, sequence: str, baseline_growth: float = None,
                      include_translation_energy: bool = True) -> dict:
    """
    Scores a single candidate sequence's Metabolic Production Burden (MPB).

    model:            a loaded cobra.Model (from fba_model_loader.load_iml1515),
                       reused across calls -- NOT reloaded per candidate.
    sequence:          candidate amino acid string.
    baseline_growth:   unconstrained growth rate to compare against. If not
                        supplied, computed fresh each call (slightly slower,
                        but avoids relying on caller-supplied state going stale).
    include_translation_energy:
                        When True, includes ATP/GTP translation overhead in the
                        same demand reaction. When False, preserves the legacy
                        amino-acid-only demand reaction for direct A/B comparison.

    Returns:
        {
            "sequence": str,
            "baseline_growth": float,
            "candidate_growth": float,
            "growth_delta": float,
            "growth_percent_reduction": float,   # the Phase 4 optimization objective
        }
    """
    if baseline_growth is None:
        baseline_solution = model.optimize()
        if baseline_solution.status != "optimal":
            raise FBAEvaluatorError(
                f"[ERROR] Baseline FBA (no candidate constraint) did not reach "
                f"an optimal solution (status: {baseline_solution.status}). "
                f"Cannot compute a meaningful percent reduction against this."
            )
        baseline_growth = baseline_solution.objective_value

    with model:
        demand_reaction = build_demand_reaction(
            model,
            sequence,
            include_translation_energy=include_translation_energy,
        )
        model.add_reactions([demand_reaction])

        # Force flux through the demand reaction: the model must actually
        # "pay" to produce PEPTIDE_DEMAND_FLUX units of the candidate
        model.reactions.get_by_id(DEMAND_REACTION_ID).lower_bound = PEPTIDE_DEMAND_FLUX

        candidate_solution = model.optimize()

        if candidate_solution.status != "optimal":
            # Infeasible/no-growth candidates are a legitimate, meaningful
            # result (this candidate's amino acid demands can't be met at
            print(f"[WARNING] Candidate infeasible at PEPTIDE_DEMAND_FLUX={PEPTIDE_DEMAND_FLUX} "
                  f"(status: {candidate_solution.status}). Scoring as 100% growth reduction. "
                  f"If this happens for most/all candidates, PEPTIDE_DEMAND_FLUX is still too "
                  f"high -- lower it in fba_evaluator.py.")
            candidate_growth = 0.0
        else:
            candidate_growth = candidate_solution.objective_value

    # model automatically reverted to baseline state here (demand
    # reaction removed, bounds restored) -- confirmed COBRApy behavior,

    growth_delta = baseline_growth - candidate_growth
    growth_percent_reduction = (
        100.0 * (1.0 - candidate_growth / baseline_growth) if baseline_growth > 0 else 100.0
    )

    return {
        "sequence": sequence,
        "baseline_growth": baseline_growth,
        "candidate_growth": candidate_growth,
        "growth_delta": growth_delta,
        "growth_percent_reduction": growth_percent_reduction,
    }
