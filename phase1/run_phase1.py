# run_phase1.py
"""
Phase 1 Orchestrator.

CHANGE FROM PRIOR VERSION
==========================
Previously this script only printed an audit block and held all outputs
in memory (coordinate_tensor, cleaned_sequence, production_graph never
touched disk). Phase 2 has no way to consume in-memory Python objects
from a separate script/session, so this version adds the persistence
step we locked in as the Phase 1 -> Phase 2 handoff contract:

  ./pipeline_artifacts/phase1_structure.pt   -- torch.save({'bb_coords': tensor})
  ./pipeline_artifacts/phase1_manifest.json  -- sidecar metadata

Everything else (structure parsing, metabolic mining, graph compilation)
is unchanged.
"""

import os
import json
from datetime import datetime, timezone

import torch


# Fixed per the locked handoff contract -- change here if the target
# PDB/chain ever changes, nowhere else.
# >>> PEPTIDE_SPECIFIC: replace for a new target peptide
PDB_CODE = "1chl"
CHAIN_ID = "A"
# <<< END_PEPTIDE_SPECIFIC
ARTIFACTS_DIR = "./pipeline_artifacts"


def persist_phase1_outputs(coordinate_tensor: torch.Tensor, sequence: str, raw_pdb_path: str) -> None:
    """
    Writes the dual-file handoff contract Phase 2 depends on:
      1. phase1_structure.pt  -- {'bb_coords': Tensor[L, 4, 3]}
      2. phase1_manifest.json -- human-readable sidecar, cross-checkable
         against the tensor before Phase 2 trusts it.

    NOTE: raw_pdb_path is included because ProteinMPNN's own PDB parser
    needs the actual .pdb/.ent file on disk -- our extracted tensor isn't
    a substitute input for their pipeline. Without this, Phase 2 would
    have no reliable way to locate the source structure file.
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    coord_file = "phase1_structure.pt"
    coord_path = os.path.join(ARTIFACTS_DIR, coord_file)
    torch.save({"bb_coords": coordinate_tensor}, coord_path)

    manifest = {
        "pdb_id": PDB_CODE.upper(),
        "chain_id": CHAIN_ID,
        "sequence": sequence,
        "num_residues": len(sequence),
        "coord_file": coord_file,
        "raw_pdb_path": os.path.abspath(raw_pdb_path),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    manifest_path = os.path.join(ARTIFACTS_DIR, "phase1_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[SUCCESS] Persisted Phase 1 handoff artifacts to '{ARTIFACTS_DIR}/':")
    print(f"           - {coord_file}  (tensor shape {list(coordinate_tensor.shape)})")
    print(f"           - phase1_manifest.json")


def execute_pipeline_phase_1():
    print("================================================================================")
    print("      PRODUCTION RUN: PHASE 1 DATA COUPLING HARDENED INITIALIZATION")
    print("================================================================================")

    # 1. Parse and download structural metrics safely
    parser = StructureParser(pdb_code=PDB_CODE)
    file_path = parser.fetch_pdb_file()
    coordinate_tensor, cleaned_sequence = parser.extract_backbone_tensors(file_path, chain_id=CHAIN_ID)

    # 2. Query KEGG database using link mapping and persistent cache
    miner = MetabolicMiner()
    metabolic_data_pool = {}
    print("\n[START] Executing Cached / Active Query Sequences across KEGG Links...")
    for residue in set(cleaned_sequence):
        pathway_metadata = miner.fetch_ec_pathway_data(residue)
        metabolic_data_pool[residue] = pathway_metadata

    # 3. Assemble structural network topology model
    compiler = ManufacturingGraphCompiler()
    production_graph = compiler.build_expression_dag(cleaned_sequence, metabolic_data_pool)

    # 4. Persist the handoff artifacts Phase 2 depends on
    print()
    persist_phase1_outputs(coordinate_tensor, cleaned_sequence, raw_pdb_path=file_path)

    # 5. Infrastructure Provenance and Integrity Verification Audit
    print("\n================================================================================")
    print("                PHASE 1 PIPELINE AUDIT & INTEGRITY REVIEWS")
    print("================================================================================")
    kegg_sourced = sum(1 for res in metabolic_data_pool.values() if res["provenance"] == "kegg")
    fallback_sourced = sum(1 for res in metabolic_data_pool.values() if res["provenance"] == "fallback")

    print(f"* Structural Chain Target Length: {len(cleaned_sequence)} residues.")
    print(f"* PyTorch Geometric Coordinates:   Tensor matrix spatial dimension {list(coordinate_tensor.shape)}")
    print(f"* Live API Provenance Registry:   {kegg_sourced} Sourced from KEGG | {fallback_sourced} Used Fallback Parameters")

    sample_residue = list(metabolic_data_pool.keys())[0]
    print(f"* Sample Data Mapping [{sample_residue}]: Path -> {metabolic_data_pool[sample_residue]['pathway_id']} | Verified Enzymes Mapped: {metabolic_data_pool[sample_residue]['associated_enzymes']}")
    print("================================================================================\n")


#  RUN
execute_pipeline_phase_1()
