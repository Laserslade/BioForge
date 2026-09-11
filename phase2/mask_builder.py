"""mask_builder.py - Builds and validates the frozen/mutable position mask that gates which residues the optimizer is allowed to touch."""

import os
import json


# ---------------------------------------------------------------------------
# LOCKED BIOLOGICAL MASK DEFINITION (0-indexed, verified against the real

# >>> PEPTIDE_SPECIFIC: replace for a new target peptide
EXPECTED_SEQUENCE = "MCMPCFTTDHQMARKCDDCCGGKGRGKCYGPQCLCR"
# <<< END_PEPTIDE_SPECIFIC

# Layer 1: Disulfide Fold Lock -- all 8 cysteines, preserves the core
# alpha/beta toxin fold.
# >>> PEPTIDE_SPECIFIC: replace for a new target peptide
LAYER_1_DISULFIDE_LOCK = {
    1: "C", 4: "C", 15: "C", 18: "C", 19: "C", 27: "C", 32: "C", 34: "C",
}
# <<< END_PEPTIDE_SPECIFIC

# >>> PEPTIDE_SPECIFIC: replace for a new target peptide
# Layer 2: Solvent-Exposed Target Patch -- KGRGK basic recognition patch +
# Y29 aromatic packing anchor. Protects neuropilin-1 / receptor binding.
LAYER_2_TARGET_PATCH = {
    22: "K", 24: "R", 26: "K", 28: "Y",
}
# <<< END_PEPTIDE_SPECIFIC

# Layer 3: Local Geometry Anchors -- alpha-helix initiation residues +
# P31 rigid turn geometry.
# >>> PEPTIDE_SPECIFIC: replace for a new target peptide
LAYER_3_GEOMETRY_ANCHORS = {
    12: "A", 13: "R", 14: "K", 30: "P",
}
# <<< END_PEPTIDE_SPECIFIC

# >>> PEPTIDE_SPECIFIC: replace for a new target peptide
FROZEN_LAYERS = {
    "disulfide_fold_lock": LAYER_1_DISULFIDE_LOCK,
    "target_patch": LAYER_2_TARGET_PATCH,
    "geometry_anchors": LAYER_3_GEOMETRY_ANCHORS,
}
# <<< END_PEPTIDE_SPECIFIC


class MaskBuilderError(Exception):
    """Raised when the mask fails validation against the actual sequence."""
    pass


class MaskBuilder:
    def __init__(self, artifacts_dir: str = "./pipeline_artifacts"):
        self.artifacts_dir = artifacts_dir

    def _load_manifest(self) -> dict:
        manifest_path = os.path.join(self.artifacts_dir, "phase1_manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"[ERROR] Phase 1 manifest not found at {manifest_path}. "
                f"Run Phase 1 first."
            )
        with open(manifest_path, "r") as f:
            return json.load(f)

    def _flatten_frozen_indices(self) -> dict:
        """Merge all three layers into a single {index: expected_residue} map,
        raising if any layer accidentally overlaps another."""
        flattened = {}
        for layer_name, layer_indices in FROZEN_LAYERS.items():
            for idx, expected_res in layer_indices.items():
                if idx in flattened:
                    raise MaskBuilderError(
                        f"[ERROR] Index {idx} appears in more than one frozen "
                        f"layer (last seen in '{layer_name}'). Layers must be "
                        f"mutually exclusive."
                    )
                flattened[idx] = expected_res
        return flattened

    def build_mask(self, sequence: str = None) -> dict:
        """
        Builds and validates the position-level mask against the given
        sequence (or the Phase 1 manifest's sequence if none supplied).

        Returns a dict keyed by 0-indexed position string, e.g.:
          {
            "pos_0": {"status": "MUTABLE", "wt": "M", "designable": True},
            "pos_1": {"status": "FROZEN",  "wt": "C", "designable": False,
                       "layer": "disulfide_fold_lock"},
            ...
          }
        """
        if sequence is None:
            manifest = self._load_manifest()
            sequence = manifest["sequence"]

        if sequence != EXPECTED_SEQUENCE:
            raise MaskBuilderError(
                f"[ERROR] Sequence mismatch. Mask was audited against:\n"
                f"  {EXPECTED_SEQUENCE}\n"
                f"but received:\n"
                f"  {sequence}\n"
                f"The frozen-position rationale (cysteines, binding patch, "
                f"geometry anchors) is only valid for the audited sequence. "
                f"Re-audit the mask before proceeding with a different target."
            )

        frozen_map = self._flatten_frozen_indices()

        # Reverse lookup: index -> layer name, for annotation in the output.
        index_to_layer = {}
        for layer_name, layer_indices in FROZEN_LAYERS.items():
            for idx in layer_indices:
                index_to_layer[idx] = layer_name

        mask = {}
        for i, residue in enumerate(sequence):
            if i in frozen_map:
                expected_res = frozen_map[i]
                if residue != expected_res:
                    raise MaskBuilderError(
                        f"[ERROR] Position {i} expected residue '{expected_res}' "
                        f"(per audited mask) but sequence has '{residue}'. "
                        f"Aborting -- freezing the wrong residue would silently "
                        f"corrupt the design space."
                    )
                mask[f"pos_{i}"] = {
                    "status": "FROZEN",
                    "wt": residue,
                    "designable": False,
                    "layer": index_to_layer[i],
                }
            else:
                mask[f"pos_{i}"] = {
                    "status": "MUTABLE",
                    "wt": residue,
                    "designable": True,
                }

        self._validate_mask_summary(mask)
        return mask

    def _validate_mask_summary(self, mask: dict) -> None:
        frozen_count = sum(1 for v in mask.values() if v["status"] == "FROZEN")
        mutable_count = sum(1 for v in mask.values() if v["status"] == "MUTABLE")

        # >>> PEPTIDE_SPECIFIC: replace for a new target peptide
        if frozen_count != 16:
        # <<< END_PEPTIDE_SPECIFIC
            raise MaskBuilderError(
                f"[ERROR] Expected 16 frozen positions, got {frozen_count}. "
                f"Mask does not match the audited spec."
            )
        # >>> PEPTIDE_SPECIFIC: replace for a new target peptide
        if mutable_count != 20:
        # <<< END_PEPTIDE_SPECIFIC
            raise MaskBuilderError(
                f"[ERROR] Expected 20 mutable positions, got {mutable_count}. "
                f"Mask does not match the audited spec."
            )

    def save_mask(self, mask: dict, output_path: str = None) -> str:
        if output_path is None:
            output_path = os.path.join(self.artifacts_dir, "phase2_mask.json")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(mask, f, indent=2)
        return output_path
