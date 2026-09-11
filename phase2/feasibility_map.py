"""feasibility_map.py - Converts ProteinMPNN's raw probability output into the feasibility map that later phases query directly."""

import os
import json

import numpy as np

ARTIFACTS_DIR = "./pipeline_artifacts"

# The 20 standard amino acids, excluding ProteinMPNN's 21st token 'X' (gap /
# unknown), which is never a valid design choice for a real peptide.
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


class FeasibilityMapError(Exception):
    pass


class FeasibilityMapBuilder:
    def __init__(self, artifacts_dir: str = ARTIFACTS_DIR):
        self.artifacts_dir = artifacts_dir

    def _load_mask(self) -> dict:
        mask_path = os.path.join(self.artifacts_dir, "phase2_mask.json")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"[ERROR] Missing {mask_path}. Run mask_builder.py first.")
        with open(mask_path) as f:
            return json.load(f)

    def build(self, probs: np.ndarray, alphabet: str, mask: dict = None) -> dict:
        """
        probs:    [L, 21] array of per-position probabilities from
                  protein_mpnn_runner.py (already exp()'d from log_p).
        alphabet: the 21-token alphabet string used to index probs
                  ('ACDEFGHIKLMNPQRSTVWYX'), so column j corresponds to
                  alphabet[j].
        mask:     the mask_builder.py output; loaded from disk if not given.
        """
        if mask is None:
            mask = self._load_mask()

        if probs.shape[0] != len(mask):
            raise FeasibilityMapError(
                f"[ERROR] Probability matrix has {probs.shape[0]} positions but "
                f"mask has {len(mask)} positions. These must match 1:1 -- "
                f"something upstream (parsing, chain selection) diverged."
            )
        if probs.shape[1] != len(alphabet):
            raise FeasibilityMapError(
                f"[ERROR] Probability matrix has {probs.shape[1]} columns but "
                f"alphabet has {len(alphabet)} tokens. Column-to-residue mapping "
                f"would be silently wrong if we proceeded."
            )

        feasibility_map = {}

        for i in range(probs.shape[0]):
            key = f"pos_{i}"
            entry = mask[key]

            if entry["status"] == "FROZEN":
                feasibility_map[key] = {
                    "status": "FROZEN",
                    "wt": entry["wt"],
                    "allowed": [entry["wt"]],
                    "layer": entry.get("layer"),
                }
                continue

            # Mutable position: build the full continuous distribution over
            # the 20 standard amino acids, dropping the 'X' gap token and
            aa_probs = {}
            for j, token in enumerate(alphabet):
                if token in STANDARD_AA:
                    aa_probs[token] = float(probs[i, j])

            total = sum(aa_probs.values())
            if total <= 0:
                raise FeasibilityMapError(
                    f"[ERROR] Position {i}: all standard-amino-acid probability "
                    f"mass is zero after dropping 'X'. Something is wrong with "
                    f"the input probability matrix at this position."
                )
            aa_probs = {aa: p / total for aa, p in aa_probs.items()}

            feasibility_map[key] = {
                "status": "MUTABLE",
                "wt": entry["wt"],
                "probabilities": dict(
                    sorted(aa_probs.items(), key=lambda kv: kv[1], reverse=True)
                ),
            }

        return feasibility_map

    def save(self, feasibility_map: dict, output_path: str = None) -> str:
        if output_path is None:
            output_path = os.path.join(self.artifacts_dir, "phase2_feasibility_map.json")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(feasibility_map, f, indent=2)
        return output_path
