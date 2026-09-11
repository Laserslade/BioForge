"""peptide_demand_builder.py - Builds the synthetic demand reaction representing heterologous expression of the target peptide."""

import cobra

# One-letter amino acid code -> BiGG cytosolic metabolite ID.
# Pattern: '<name>__L_c' for all except glycine ('gly_c', achiral).
AA_TO_BIGG_ID = {
    'A': 'ala__L_c',
    'R': 'arg__L_c',
    'N': 'asn__L_c',
    'D': 'asp__L_c',
    'C': 'cys__L_c',
    'Q': 'gln__L_c',
    'E': 'glu__L_c',
    'G': 'gly_c',
    'H': 'his__L_c',
    'I': 'ile__L_c',
    'L': 'leu__L_c',
    'K': 'lys__L_c',
    'M': 'met__L_c',
    'F': 'phe__L_c',
    'P': 'pro__L_c',
    'S': 'ser__L_c',
    'T': 'thr__L_c',
    'W': 'trp__L_c',
    'Y': 'tyr__L_c',
    'V': 'val__L_c',
}

DEMAND_REACTION_ID = "DM_candidate_peptide"


class DemandBuilderError(Exception):
    pass


def validate_metabolite_ids(model: cobra.Model, residues: set) -> None:
    """
    Confirms every BiGG ID we're about to use actually exists in the
    loaded model, for exactly the residues present in this sequence.
    Fails with a specific, actionable error naming the bad ID(s) rather
    than letting COBRApy raise a generic KeyError deep in add_boundary.
    """
    missing = []
    for residue in residues:
        if residue not in AA_TO_BIGG_ID:
            missing.append(f"'{residue}' has no entry in AA_TO_BIGG_ID at all")
            continue
        bigg_id = AA_TO_BIGG_ID[residue]
        if bigg_id not in model.metabolites:
            missing.append(f"'{residue}' -> '{bigg_id}' not found in loaded model")

    if missing:
        raise DemandBuilderError(
            "[ERROR] One or more amino acid metabolite IDs could not be validated "
            "against the loaded model:\n  " + "\n  ".join(missing) +
            "\nThis means AA_TO_BIGG_ID's naming pattern doesn't hold for these "
            "residues in this model version. Look up the correct ID(s) at "
            "http://bigg.ucsd.edu/models/iML1515/metabolites before proceeding -- "
            "do not guess a fix."
        )


def build_demand_reaction(model: cobra.Model, sequence: str) -> cobra.Reaction:
    """
    Builds (but does not add to the model) a demand reaction draining
    one unit of each amino acid precursor per occurrence in `sequence`.
    Intended to be added inside a `with model:` block by fba_evaluator.py
    so it's automatically reverted after each evaluation.
    """
    if not sequence:
        raise DemandBuilderError("[ERROR] Cannot build a demand reaction for an empty sequence.")

    residue_counts = {}
    for residue in sequence:
        residue_counts[residue] = residue_counts.get(residue, 0) + 1

    validate_metabolite_ids(model, set(residue_counts.keys()))

    reaction = cobra.Reaction(DEMAND_REACTION_ID)
    reaction.name = "Candidate peptide synthesis demand (amino-acid precursor burden only)"
    reaction.lower_bound = 0
    reaction.upper_bound = 1000  # effectively unconstrained forward flux

    metabolite_coefficients = {}
    for residue, count in residue_counts.items():
        bigg_id = AA_TO_BIGG_ID[residue]
        metabolite = model.metabolites.get_by_id(bigg_id)
        # Negative coefficient = consumed (this is the demand side).
        metabolite_coefficients[metabolite] = -float(count)

    reaction.add_metabolites(metabolite_coefficients)
    return reaction
