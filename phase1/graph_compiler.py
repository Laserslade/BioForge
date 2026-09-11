import networkx as nx

class ManufacturingGraphCompiler:
    def __init__(self):
        self.graph = nx.DiGraph()

    def build_expression_dag(self, sequence: str, metabolic_data_pool: dict) -> nx.DiGraph:
        """Compiles sequence parameters and pathway attributes into a Directed Acyclic Graph."""
        print("[INFO] Initiating Construction of NetworkX Manufacturing Graph...")
        self.graph.clear() # Clear state
        
        self.graph.add_node("E_coli_Biomass_Sink", type="Output", capacity="Target_Expression")
        unique_residues = set(sequence)
        
        for residue in unique_residues:
            data = metabolic_data_pool.get(residue, {})
            atp_weight = data.get("stoichiometry", {}).get("atp_cost", 20.0)
            precursor = data.get("stoichiometry", {}).get("carbon_precursor", "Glucose")
            provenance = data.get("provenance", "unknown")
            
            amino_acid_node = f"Residue_{residue}"
            precursor_node = f"Feedstock_{precursor}"
            
            self.graph.add_node(amino_acid_node, type="Amino_Acid_Pool", residue=residue, provenance=provenance)
            self.graph.add_node(precursor_node, type="Raw_Input", cost_index=1.0)
            
            self.graph.add_edge(
                precursor_node, 
                amino_acid_node, 
                weight=atp_weight, 
                active_enzymes=data.get("associated_enzymes", []),
                provenance=provenance
            )
            
            seq_frequency = sequence.count(residue)
            self.graph.add_edge(amino_acid_node, "E_coli_Biomass_Sink", weight=0.0, total_required_units=seq_frequency)
            
        print(f"[SUCCESS] Graph Compiled. Active Nodes: {self.graph.number_of_nodes()}, Directed Edges: {self.graph.number_of_edges()}")
        return self.graph
