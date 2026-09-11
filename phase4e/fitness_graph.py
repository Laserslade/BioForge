"""fitness_graph.py - Maintains a graph of evaluated sequences and their fitness relationships for the enhanced optimizer."""
import sys, os


def hamming_distance_mutable(seq_a, seq_b, mutable_positions):
    """Hamming distance counting only mutable positions."""
    return sum(1 for pos in mutable_positions if seq_a[pos] != seq_b[pos])


class FitnessGraph:
    def __init__(self, mutable_positions, max_edge_distance=2):
        self.mutable_positions  = mutable_positions
        self.max_edge_distance  = max_edge_distance
        self._nodes             = {}   # sequence -> candidate dict
        self._adjacency         = {}   # sequence -> set of neighbor sequences

    def add_candidates(self, candidates):
        """Add a batch of evaluated candidates, building edges lazily."""
        new_seqs = []
        for cand in candidates:
            seq = cand["sequence"]
            if seq not in self._nodes:
                self._nodes[seq]      = cand
                self._adjacency[seq]  = set()
                new_seqs.append(seq)

        # Connect new sequences to all existing nodes within max_edge_distance
        existing = list(self._nodes.keys())
        for new_seq in new_seqs:
            for existing_seq in existing:
                if new_seq == existing_seq: continue
                d = hamming_distance_mutable(new_seq, existing_seq, self.mutable_positions)
                if d <= self.max_edge_distance:
                    self._adjacency[new_seq].add(existing_seq)
                    self._adjacency[existing_seq].add(new_seq)

    def get_neighbors(self, sequence):
        """Return all evaluated sequences within max_edge_distance mutations."""
        return [self._nodes[s] for s in self._adjacency.get(sequence, set())]

    def is_local_optimum(self, candidate):
        """
        Returns True if no neighbor in the graph dominates this candidate.
        In a local optimum, single/double mutations have no improving direction.
        """
        neighbors = self.get_neighbors(candidate["sequence"])
        if not neighbors:
            return False  # isolated node -- not enough info to declare local opt
        return not any(dominates(n, candidate) for n in neighbors)

    def get_underexplored_positions(self, population, top_k=5):
        """
        Returns the mutable positions least represented in the population's
        current sequence diversity. Used to direct perturbations toward
        regions where little variation has been tried.
        """
        from collections import Counter
        position_diversity = {}
        for pos in self.mutable_positions:
            aas_seen = Counter(c["sequence"][pos] for c in population)
            # Entropy-proxy: fewer unique AAs = less explored
            position_diversity[pos] = len(aas_seen)

        # Return positions sorted by least diversity (most underexplored first)
        sorted_positions = sorted(position_diversity.items(), key=lambda x: x[1])
        return [pos for pos, _ in sorted_positions[:top_k]]

    @property
    def size(self):
        return len(self._nodes)
