"""metabolic_miner.py - Looks up per-residue biosynthetic pathway metadata from KEGG, with local fallback values on API failure."""

import os
import json
import time
import requests


# KEGG pathway IDs that are broad "overview" maps rather than specific
# biosynthesis pathways. These get excluded when picking a target
GENERIC_OVERVIEW_PATHWAYS = {
    "eco01100",  # Metabolic pathways (global overview)
    "eco01110",  # Biosynthesis of secondary metabolites
    "eco01120",  # Microbial metabolism in diverse environments
    "eco01200",  # Carbon metabolism
    "eco01210",  # 2-Oxocarboxylic acid metabolism
    "eco01230",  # Biosynthesis of amino acids (overview, not residue-specific)
    "eco01240",  # Biosynthesis of cofactors
    "eco01250",  # Biosynthesis of nucleotide sugars
}


class MetabolicMiner:
    def __init__(self, cache_file: str = "kegg_cache.json"):
        self.kegg_base_url = "https://rest.kegg.jp"
        self.cache_file = cache_file
        self.cache = self._load_cache()

        # Verified against KEGG's live /list/compound and /find/compound
        # endpoints. Each entry below was cross-checked individually.
        self.aa_to_cpd = {
            'A': 'cpd:C00041',  # L-Alanine
            'R': 'cpd:C00062',  # L-Arginine
            'N': 'cpd:C00152',  # L-Asparagine
            'D': 'cpd:C00049',  # L-Aspartate  (FIXED: was C00042 / Succinate)
            'C': 'cpd:C00097',  # L-Cysteine
            'Q': 'cpd:C00064',  # L-Glutamine
            'E': 'cpd:C00025',  # L-Glutamate
            'G': 'cpd:C00037',  # Glycine
            'H': 'cpd:C00135',  # L-Histidine
            'I': 'cpd:C00407',  # L-Isoleucine
            'L': 'cpd:C00123',  # L-Leucine
            'K': 'cpd:C00047',  # L-Lysine    (FIXED: was C00408 / wrong compound)
            'M': 'cpd:C00073',  # L-Methionine
            'F': 'cpd:C00079',  # L-Phenylalanine
            'P': 'cpd:C00148',  # L-Proline
            'S': 'cpd:C00065',  # L-Serine
            'T': 'cpd:C00188',  # L-Threonine
            'W': 'cpd:C00078',  # L-Tryptophan
            'Y': 'cpd:C00082',  # L-Tyrosine
            'V': 'cpd:C00183',  # L-Valine
        }

    def _load_cache(self) -> dict:
        """Load on-disk cache, tolerating a missing or corrupted file."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"[WARNING] Cache file unreadable ({e}); rebuilding cache from scratch.")
                return {}
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=4)
        except OSError as e:
            print(f"[WARNING] Could not persist cache to disk: {e}")

    def _select_specific_pathway(self, eco_pathways: list) -> str:
        """
        Given a list of path:ecoXXXXX IDs linked to a compound, return the
        most specific (non-overview) pathway. Falls back to the first
        entry only if every candidate is a generic overview pathway.
        """
        specific = [
            p for p in eco_pathways
            if p.replace("path:", "") not in GENERIC_OVERVIEW_PATHWAYS
        ]
        if specific:
            return specific[0]
        # Every linked pathway was a generic overview map -- still return
        # something rather than failing, but this is a weaker signal.
        return eco_pathways[0]

    def fetch_ec_pathway_data(self, amino_acid: str) -> dict:
        """Queries KEGG using compound-to-pathway/enzyme linkage strategies with caching."""
        aa_upper = amino_acid.upper()
        if aa_upper not in self.aa_to_cpd:
            return self._get_fallback_stoichiometry(aa_upper, reason="Invalid residue code")

        if aa_upper in self.cache:
            return self.cache[aa_upper]

        cpd_id = self.aa_to_cpd[aa_upper]

        try:
            pathway_url = f"{self.kegg_base_url}/link/pathway/{cpd_id}"
            print(f"[API CALL] Querying pathway links for {aa_upper} ({cpd_id})...")
            p_response = requests.get(pathway_url, timeout=10)
            time.sleep(0.1)

            if p_response.status_code != 200 or not p_response.text.strip():
                raise ConnectionError("Empty or invalid response from KEGG pathway link.")

            # FIXED: this endpoint only ever returns generic "path:mapXXXXX"
            # reference IDs (e.g. cpd:C00001 -> "path:map00190"), never
            eco_pathways = []
            for line in p_response.text.strip().split("\n"):
                parts = line.split("\t")
                if len(parts) == 2 and "path:map" in parts[1]:
                    map_id = parts[1].strip()
                    eco_id = map_id.replace("path:map", "path:eco")
                    eco_pathways.append(eco_id)

            if not eco_pathways:
                raise ValueError(f"No pathway links returned by KEGG for compound {cpd_id}")

            # FIXED: was eco_pathways[0] with no filtering -- now excludes
            # generic overview pathways so we land on the residue-specific
            target_pathway = self._select_specific_pathway(eco_pathways)

            enzyme_url = f"{self.kegg_base_url}/link/enzyme/{target_pathway.replace('path:', '')}"
            print(f"[API CALL] Querying actual host enzyme markers for {target_pathway}...")
            e_response = requests.get(enzyme_url, timeout=10)
            time.sleep(0.1)

            # --- TEMPORARY DEBUG: remove once we've diagnosed the empty-enzyme-list issue ---
            print(f"[DEBUG] Enzyme URL: {enzyme_url}")
            print(f"[DEBUG] Status: {e_response.status_code} | Body (first 300 chars): {e_response.text[:300]!r}")
            # --- END TEMPORARY DEBUG ---

            associated_enzymes = []
            if e_response.status_code == 200 and e_response.text.strip():
                for line in e_response.text.strip().split("\n"):
                    parts = line.split("\t")
                    if len(parts) == 2:
                        associated_enzymes.append(parts[1].replace("ec:", "").strip())

            result = {
                "amino_acid": aa_upper,
                "pathway_id": target_pathway,
                "associated_enzymes": associated_enzymes[:5],
                "stoichiometry": self._get_fallback_stoichiometry(aa_upper)["stoichiometry"],
                "provenance": "kegg"
            }

            self.cache[aa_upper] = result
            self._save_cache()
            return result

        except Exception as e:
            print(f"[WARNING] KEGG query anomaly for {aa_upper}: {str(e)}. Executing fallback framework.")
            return self._get_fallback_stoichiometry(aa_upper, reason=str(e))

    def _get_fallback_stoichiometry(self, aa: str, reason: str = "none") -> dict:
        atp_costs = {
            'A': 11.7, 'R': 27.3, 'N': 14.7, 'D': 12.7, 'C': 24.7,
            'Q': 16.3, 'E': 12.3, 'G': 11.7, 'H': 38.3, 'I': 32.3,
            'L': 27.3, 'K': 30.3, 'M': 34.3, 'F': 39.0, 'P': 20.3,
            'S': 11.7, 'T': 18.7, 'W': 60.3, 'Y': 50.0, 'V': 23.3
        }
        return {
            "amino_acid": aa,
            "pathway_id": "unknown_fallback",
            "associated_enzymes": [],
            "stoichiometry": {"atp_cost": atp_costs.get(aa, 20.0), "carbon_precursor": "Glucose"},
            "provenance": "fallback",
            "fallback_reason": reason
        }
