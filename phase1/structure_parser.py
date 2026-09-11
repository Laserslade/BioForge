import os
from typing import Tuple
from Bio.PDB import PDBList, PDBParser
import torch

class StructureParser:
    def __init__(self, pdb_code: str, storage_dir: str = "./pdb_cache"):
        self.pdb_code = pdb_code.lower()
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        
    def fetch_pdb_file(self) -> str:
        """Downloads the raw PDB file from the RCSB server with strict verification."""
        print(f"[INFO] Fetching PDB file for {self.pdb_code.upper()} from RCSB...")
        pdbl = PDBList()
        file_path = pdbl.retrieve_pdb_file(self.pdb_code, pdir=self.storage_dir, file_format="pdb")
        
        # Check for network-failure resilience or empty file drops
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[ERROR] PDB file failed to download or save at: {file_path}")
        if os.path.getsize(file_path) < 1024:  # Must be greater than 1KB
            raise IOError(f"[ERROR] Downloaded PDB file is corrupt or empty (Size: {os.path.getsize(file_path)} bytes).")
            
        return file_path

    def extract_backbone_tensors(self, file_path: str, chain_id: str = "A") -> Tuple[torch.Tensor, str]:
        """
        Parses PDB geometry and extracts a tensor of shape [L, 4, 3] 
        and the accompanying sequence string.
        """
        print(f"[INFO] Extracting backbone coordinates from Chain {chain_id}...")
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(self.pdb_code, file_path)
        model = structure[0]
        
        if chain_id not in model:
            available_chains = [c.id for c in model.get_chains()]
            raise KeyError(f"Chain {chain_id} not found in PDB. Available chains: {available_chains}")
            
        chain = model[chain_id]
        backbone_coordinates = []
        residue_sequence = []
        
        three_to_one = {
            'ALA':'A', 'CYS':'C', 'ASP':'D', 'GLU':'E', 'PHE':'F',
            'GLY':'G', 'HIS':'H', 'ILE':'I', 'LYS':'K', 'LEU':'L',
            'MET':'M', 'ASN':'N', 'PRO':'P', 'GLN':'Q', 'ARG':'R',
            'SER':'S', 'THR':'T', 'VAL':'V', 'TRP':'W', 'TYR':'Y'
        }

        for residue in chain:
            if residue.id[0] != " ":
                continue
                
            res_name = residue.get_resname().strip()
            if res_name not in three_to_one:
                continue
                
            required_atoms = ["N", "CA", "C", "O"]
            if all(atom in residue for atom in required_atoms):
                residue_coords = [residue[atom].get_coord() for atom in required_atoms]
                backbone_coordinates.append(residue_coords)
                residue_sequence.append(three_to_one[res_name])
                
        if not backbone_coordinates:
            raise ValueError(f"[ERROR] No valid structural residues extracted from Chain {chain_id}.")
            
        coordinate_tensor = torch.tensor(backbone_coordinates, dtype=torch.float32)
        sequence_str = "".join(residue_sequence)
        
        print(f"[SUCCESS] Ingested Sequence: {sequence_str}")
        print(f"[SUCCESS] Coordinates Matrix Shape: {list(coordinate_tensor.shape)}")
        
        return coordinate_tensor, sequence_str
