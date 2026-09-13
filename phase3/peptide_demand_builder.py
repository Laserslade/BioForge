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

# Required translation-associated metabolites in the cytosol for this request.
REQUIRED_TRANSLATION_METABOLITES = {
    'atp_c',
    'adp_c',
    'amp_c',
    'gtp_c',
    'gdp_c',
    'pi_c',
    'ppi_c',
    'h2o_c',
    'h_c',
}

DEMAND_REACTION_ID = "DM_candidate_peptide"


class DemandBuilderError(Exception):
    pass


def validate_metabolite_ids(model: cobra.Model, residues: set, required_metabolites: set | None = None) -> None:
    """
    Confirms every BiGG ID we're about to use actually exists in the
    loaded model, for exactly the residues present in this sequence and the
    model-level translation-energy cofactors required by this demand reaction.
    """
    missing = []

    if required_metabolites is not None:
        for metabolite_id in sorted(required_metabolites):
            if metabolite_id not in model.metabolites:
                missing.append(f"Missing required metabolite '{metabolite_id}' in loaded model")

    for residue in residues:
        if residue not in AA_TO_BIGG_ID:
            missing.append(f"'{residue}' has no entry in AA_TO_BIGG_ID at all")
            continue
        bigg_id = AA_TO_BIGG_ID[residue]
        if bigg_id not in model.metabolites:
            missing.append(f"'{residue}' -> '{bigg_id}' not found in loaded model")

    if missing:
        raise DemandBuilderError(
            "[ERROR] One or more metabolite IDs could not be validated against the loaded model:\n  "
            + "\n  ".join(missing)
            + "\nLook up the correct ID(s) at http://bigg.ucsd.edu/models/iML1515/metabolites "
            "before proceeding -- do not guess a fix."
        )


def build_demand_reaction(model: cobra.Model, sequence: str, include_translation_energy: bool = True) -> cobra.Reaction:
    """
    Builds (but does not add to the model) a demand reaction for the peptide.

    When include_translation_energy is True, the reaction drains the amino acid
    precursors plus the ATP/GTP costs of tRNA charging and ribosomal elongation.
    When False, it reproduces the exact legacy amino-acid-only behavior.
    """
    if not sequence:
        raise DemandBuilderError("[ERROR] Cannot build a demand reaction for an empty sequence.")

    residue_counts = {}
    for residue in sequence:
        residue_counts[residue] = residue_counts.get(residue, 0) + 1

    validate_metabolite_ids(
        model,
        set(residue_counts.keys()),
        required_metabolites=REQUIRED_TRANSLATION_METABOLITES if include_translation_energy else None,
    )

    reaction = cobra.Reaction(DEMAND_REACTION_ID)
    if include_translation_energy:
        reaction.name = (
            "Candidate peptide synthesis demand "
            "(amino-acid precursors + tRNA charging + ribosomal elongation energy)"
        )
    else:
        reaction.name = "Candidate peptide synthesis demand (amino-acid precursor burden only)"
    reaction.lower_bound = 0
    reaction.upper_bound = 1000  # effectively unconstrained forward flux

    metabolite_coefficients = {}
    for residue, count in residue_counts.items():
        bigg_id = AA_TO_BIGG_ID[residue]
        metabolite = model.metabolites.get_by_id(bigg_id)
        # Negative coefficient = consumed (this is the demand side).
        metabolite_coefficients[metabolite] = metabolite_coefficients.get(metabolite, 0.0) - float(count)

    if include_translation_energy:
        n_charging_events = len(sequence)
        n_elongation_events = len(sequence) - 1 if len(sequence) > 1 else 0

        # The model represents ATP maintenance as ATP + H2O -> ADP + H + Pi.
        # We mirror that exact convention for GTP hydrolysis in the elongation term.
        # Initiation/termination GTP costs are intentionally omitted as a minor
        # refinement and are not part of the current demand reaction scope.
        for metabolite_id, delta in {
            'atp_c': -n_charging_events,
            'amp_c': +n_charging_events,
            'ppi_c': +n_charging_events,
            'gtp_c': -(2 * n_elongation_events),
            'h2o_c': -(2 * n_elongation_events),
            'gdp_c': +(2 * n_elongation_events),
            'pi_c': +(2 * n_elongation_events),
            'h_c': +(2 * n_elongation_events),
        }.items():
            metabolite = model.metabolites.get_by_id(metabolite_id)
            metabolite_coefficients[metabolite] = metabolite_coefficients.get(metabolite, 0.0) + float(delta)

    reaction.add_metabolites(metabolite_coefficients)
    return reaction
