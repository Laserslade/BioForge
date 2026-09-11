"""protein_mpnn_runner.py - Drives ProteinMPNN's own CLI scripts to generate per-position amino acid probabilities for the target backbone."""

import os
import sys
import json
import shutil
import subprocess

import numpy as np

# Confirmed directly from protein_mpnn_utils.py / protein_mpnn_run.py source.
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"

ARTIFACTS_DIR = "./pipeline_artifacts"
REPO_DIR = "./ProteinMPNN"
WORK_DIR = "./phase2_work"
# >>> PEPTIDE_SPECIFIC: replace for a new target peptide
MODEL_NAME = "v_48_002"  # locked decision: matches 1CHL's high-res experimental structure
# <<< END_PEPTIDE_SPECIFIC
BACKBONE_NOISE = "0.00"
SAMPLING_TEMP = "0.1"


class ProteinMPNNRunnerError(Exception):
    pass


class ProteinMPNNRunner:
    def __init__(self, artifacts_dir: str = ARTIFACTS_DIR, repo_dir: str = REPO_DIR,
                 work_dir: str = WORK_DIR):
        self.artifacts_dir = artifacts_dir
        self.repo_dir = repo_dir
        self.work_dir = work_dir

    def _require_repo(self):
        if not os.path.isdir(self.repo_dir):
            raise ProteinMPNNRunnerError(
                f"[ERROR] ProteinMPNN repo not found at '{self.repo_dir}'. "
                f"Clone it first:\n"
                f"  git clone --depth 1 https://github.com/dauparas/ProteinMPNN.git {self.repo_dir}"
            )
        run_script = os.path.join(self.repo_dir, "protein_mpnn_run.py")
        if not os.path.exists(run_script):
            raise ProteinMPNNRunnerError(
                f"[ERROR] '{run_script}' not found -- repo clone looks incomplete or "
                f"the directory structure has changed upstream."
            )

    def _load_manifest_and_mask(self):
        manifest_path = os.path.join(self.artifacts_dir, "phase1_manifest.json")
        mask_path = os.path.join(self.artifacts_dir, "phase2_mask.json")

        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"[ERROR] Missing {manifest_path}. Run Phase 1 first.")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"[ERROR] Missing {mask_path}. Run mask_builder.py first.")

        with open(manifest_path) as f:
            manifest = json.load(f)
        with open(mask_path) as f:
            mask = json.load(f)

        if "raw_pdb_path" not in manifest:
            raise ProteinMPNNRunnerError(
                "[ERROR] manifest has no 'raw_pdb_path' -- re-run the updated run_phase1.py "
                "(older manifests predate this field)."
            )
        if not os.path.exists(manifest["raw_pdb_path"]):
            raise FileNotFoundError(
                f"[ERROR] raw_pdb_path '{manifest['raw_pdb_path']}' from the manifest "
                f"does not exist on disk. Was ./pdb_cache/ cleared since Phase 1 ran?"
            )

        return manifest, mask

    def _prepare_input_pdb(self, raw_pdb_path: str) -> str:
        """
        ProteinMPNN's parser expects a folder containing exactly the PDB(s)
        to design, so we copy the cached structure into an isolated inputs/
        directory rather than pointing at Phase 1's shared pdb_cache/.

        IMPORTANT: Biopython's PDBList saves files with a '.ent' extension
        # >>> PEPTIDE_SPECIFIC: replace for a new target peptide
        (e.g. 'pdb1chl.ent'), but every documented ProteinMPNN usage example
        # <<< END_PEPTIDE_SPECIFIC
        uses '.pdb' files. parse_multiple_chains.py likely globs for '.pdb'
        specifically, which would silently parse zero structures from a
        '.ent' file -- no crash, just an empty downstream pipeline. We copy
        with a normalized '.pdb' extension to avoid that failure mode.
        """
        inputs_dir = os.path.join(self.work_dir, "inputs")
        os.makedirs(inputs_dir, exist_ok=True)
        basename_no_ext = os.path.splitext(os.path.basename(raw_pdb_path))[0]
        dest = os.path.join(inputs_dir, f"{basename_no_ext}.pdb")
        shutil.copy(raw_pdb_path, dest)
        return inputs_dir

    def _frozen_positions_1indexed(self, mask: dict) -> list:
        """Convert mask_builder's 0-indexed frozen positions to ProteinMPNN's
        1-indexed, chain-relative position convention (confirmed via
        submit_example_4.sh: 'first amino acid in the chain corresponds to 1')."""
        frozen = []
        for key, entry in mask.items():
            if entry["status"] == "FROZEN":
                zero_idx = int(key.split("_")[1])
                frozen.append(zero_idx + 1)
        return sorted(frozen)

    def _run_subprocess(self, cmd: list, step_name: str):
        print(f"[API CALL] {step_name}: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise ProteinMPNNRunnerError(
                f"[ERROR] '{step_name}' failed (exit {result.returncode}).\n"
                f"--- stdout ---\n{result.stdout}\n"
                f"--- stderr ---\n{result.stderr}"
            )
        return result

    def run(self) -> dict:
        self._require_repo()
        manifest, mask = self._load_manifest_and_mask()

        os.makedirs(self.work_dir, exist_ok=True)
        inputs_dir = self._prepare_input_pdb(manifest["raw_pdb_path"])
        parsed_dir = os.path.join(self.work_dir, "parsed")
        os.makedirs(parsed_dir, exist_ok=True)

        chain_id = manifest["chain_id"]
        pdb_basename = os.path.splitext(os.path.basename(manifest["raw_pdb_path"]))[0]

        parsed_jsonl = os.path.join(parsed_dir, "parsed_pdbs.jsonl")
        assigned_jsonl = os.path.join(parsed_dir, "assigned_chains.jsonl")
        fixed_positions_jsonl = os.path.join(parsed_dir, "fixed_positions.jsonl")

        helper_dir = os.path.join(self.repo_dir, "helper_scripts")

        # Step 1: parse the raw PDB into ProteinMPNN's internal jsonl format
        self._run_subprocess([
            sys.executable, os.path.join(helper_dir, "parse_multiple_chains.py"),
            "--input_path", inputs_dir,
            "--output_path", parsed_jsonl,
        ], "Parse PDB into jsonl")

        # Step 2: mark our single chain as the one being designed
        self._run_subprocess([
            sys.executable, os.path.join(helper_dir, "assign_fixed_chains.py"),
            "--input_path", parsed_jsonl,
            "--output_path", assigned_jsonl,
            "--chain_list", chain_id,
        ], "Assign designed chain")

        # Step 3: build the fixed (frozen) position dictionary from our audited mask
        frozen_positions = self._frozen_positions_1indexed(mask)
        position_list_str = " ".join(str(p) for p in frozen_positions)
        self._run_subprocess([
            sys.executable, os.path.join(helper_dir, "make_fixed_positions_dict.py"),
            "--input_path", parsed_jsonl,
            "--output_path", fixed_positions_jsonl,
            "--chain_list", chain_id,
            "--position_list", position_list_str,
        ], "Build fixed-positions dict")

        # Step 4: run ProteinMPNN itself, single forward pass, unconditional
        # per-position probabilities -- exactly what Option A needs.
        out_folder = os.path.join(self.work_dir, "outputs")
        os.makedirs(out_folder, exist_ok=True)

        inference_result = self._run_subprocess([
            sys.executable, os.path.join(self.repo_dir, "protein_mpnn_run.py"),
            "--jsonl_path", parsed_jsonl,
            "--chain_id_jsonl", assigned_jsonl,
            "--fixed_positions_jsonl", fixed_positions_jsonl,
            "--out_folder", out_folder,
            "--model_name", MODEL_NAME,
            "--backbone_noise", BACKBONE_NOISE,
            "--sampling_temp", SAMPLING_TEMP,
            "--num_seq_per_target", "1",
            "--batch_size", "1",
            "--unconditional_probs_only", "1",
            "--save_probs", "1",
        ], "Run ProteinMPNN inference")
        print(f"[DEBUG] ProteinMPNN stdout:\n{inference_result.stdout}")
        print(f"[DEBUG] Full out_folder contents: {list(os.walk(out_folder))}")

        npz_path = os.path.join(out_folder, "unconditional_probs_only", f"{pdb_basename}.npz")
        if not os.path.exists(npz_path):
            candidates = []
            probs_dir = os.path.join(out_folder, "unconditional_probs_only")
            if os.path.isdir(probs_dir):
                candidates = os.listdir(probs_dir)
            raise ProteinMPNNRunnerError(
                f"[ERROR] Expected output at '{npz_path}' but it doesn't exist. "
                f"Files actually present in that folder: {candidates}. "
                f"ProteinMPNN's output naming may differ from what we assumed -- "
                f"inspect the folder and adjust npz_path construction above."
            )

        npz = np.load(npz_path)
        log_p = npz["log_p"]  # expected shape [1, L, 21] or [L, 21]
        if log_p.ndim == 3:
            log_p = log_p[0]

        probs = np.exp(log_p)  # log_p -> probabilities

        # Sanity check: rows should sum close to 1.0 if these are true softmax
        # probabilities. If not, something about our assumption of the output
        row_sums = probs.sum(axis=-1)
        if not np.allclose(row_sums, 1.0, atol=1e-2):
            print(f"[WARNING] Probability rows do not sum to ~1.0 "
                  f"(min={row_sums.min():.4f}, max={row_sums.max():.4f}). "
                  f"Verify log_p is actually log-softmax output before trusting "
                  f"the feasibility map built from this.")

        return {
            "probs": probs,          # shape [L, 21]
            "alphabet": ALPHABET,
            "sequence": manifest["sequence"],
            "npz_path": npz_path,
        }
