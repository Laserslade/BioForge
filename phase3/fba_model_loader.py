"""fba_model_loader.py - Loads the genome-scale metabolic model used for flux balance analysis."""

import os
import cobra

MODEL_PATH = "./models/iML1515.xml"

# Known published statistics for iML1515 (Monk et al. 2017), used as a
# sanity check that we loaded the right model and it parsed correctly --
EXPECTED_GENE_COUNT = 1515
EXPECTED_REACTION_COUNT = 2719
EXPECTED_METABOLITE_COUNT = 1192


class ModelLoadError(Exception):
    pass


def load_iml1515(model_path: str = MODEL_PATH) -> cobra.Model:
    if not os.path.exists(model_path):
        raise ModelLoadError(
            f"[ERROR] iML1515 model not found at '{model_path}'. Download it first:\n"
            f"  mkdir -p ./models\n"
            f"  wget -O {model_path} http://bigg.ucsd.edu/static/models/iML1515.xml"
        )

    print(f"[INFO] Loading iML1515 from '{model_path}'...")
    model = cobra.io.read_sbml_model(model_path)

    n_genes, n_reactions, n_metabolites = len(model.genes), len(model.reactions), len(model.metabolites)
    print(f"[INFO] Loaded model: {n_genes} genes, {n_reactions} reactions, {n_metabolites} metabolites.")

    # Sanity check against the model's known published dimensions. A
    # mismatch here would mean we loaded a corrupted file, a different
    if n_genes != EXPECTED_GENE_COUNT or n_reactions != EXPECTED_REACTION_COUNT or n_metabolites != EXPECTED_METABOLITE_COUNT:
        print(
            f"[WARNING] Model dimensions ({n_genes}g/{n_reactions}r/{n_metabolites}m) "
            f"don't match published iML1515 statistics "
            f"({EXPECTED_GENE_COUNT}g/{EXPECTED_REACTION_COUNT}r/{EXPECTED_METABOLITE_COUNT}m). "
            f"This could be a different model version -- proceeding, but verify this is intentional."
        )

    if model.objective is None:
        raise ModelLoadError("[ERROR] Loaded model has no objective function set.")

    baseline_solution = model.optimize()
    print(f"[INFO] Baseline (unconstrained) growth rate: {baseline_solution.objective_value:.4f} 1/h")

    if baseline_solution.status != "optimal":
        raise ModelLoadError(
            f"[ERROR] Baseline FBA did not reach an optimal solution "
            f"(status: {baseline_solution.status}). Something is wrong with the "
            f"model before we've even added a candidate demand reaction."
        )

    return model
