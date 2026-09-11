"""learned_perturbation.py - Learns a posterior over which mutations tend to improve fitness, and samples from it to propose new candidates."""
import math, random


class LearnedPerturbation:
    def __init__(self, feasibility_map, mask, mutable_positions,
                 prior_weight=5.0):
        """
        prior_weight: how strongly the feasibility map prior anchors the
        initial distribution. Higher = more rollouts needed to shift away
        from ProteinMPNN's suggestions. 5.0 is a reasonable starting point
        (roughly equivalent to having seen 5 pseudo-observations per residue).
        """
        self.feasibility_map   = feasibility_map
        self.mask              = mask
        self.mutable_positions = mutable_positions
        self.prior_weight      = prior_weight

        # Success/attempt counts per (position, residue): Bayesian update
        # initialized from feasibility map probabilities * prior_weight
        self._successes = {}
        self._attempts  = {}
        for pos in mutable_positions:
            entry = feasibility_map[f"pos_{pos}"]
            for aa, prob in entry["probabilities"].items():
                key = (pos, aa)
                self._successes[key] = prob * prior_weight
                self._attempts[key]  = prior_weight

        self.total_updates = 0

    def _posterior_prob(self, pos, aa):
        """Posterior mean = (successes + prior) / (attempts + prior)."""
        key = (pos, aa)
        s   = self._successes.get(key, 0.0)
        a   = self._attempts.get(key, self.prior_weight)
        return s / a if a > 0 else 0.0

    def _sample_mutation(self, sequence, exclude_positions=None):
        """
        Sample a (position, residue) mutation weighted by posterior probabilities.
        Returns (position, new_residue) or None if no valid mutation found.
        """
        exclude = exclude_positions or set()
        candidates = []
        weights    = []

        for pos in self.mutable_positions:
            if pos in exclude: continue
            entry = self.feasibility_map[f"pos_{pos}"]
            current_aa = sequence[pos]
            for aa in entry["probabilities"]:
                if aa == current_aa: continue
                w = self._posterior_prob(pos, aa)
                if w > 0:
                    candidates.append((pos, aa))
                    weights.append(w)

        if not candidates: return None
        total = sum(weights)
        weights = [w / total for w in weights]
        chosen = random.choices(candidates, weights=weights, k=1)[0]
        return chosen

    def perturb(self, sequence, n_mutations=2):
        """
        Apply n_mutations successive substitutions guided by posterior
        probabilities. Returns mutated sequence.
        """
        result       = list(sequence)
        used_positions = set()

        for _ in range(n_mutations):
            mutation = self._sample_mutation("".join(result), exclude_positions=used_positions)
            if mutation is None: break
            pos, aa = mutation
            result[pos] = aa
            used_positions.add(pos)

        return "".join(result)

    def perturb_toward_unexplored(self, sequence, underexplored_positions, n_mutations=2):
        """
        Biased perturbation: preferentially mutate underexplored positions
        (from FitnessGraph.get_underexplored_positions) to push the search
        into genuinely unvisited regions of sequence space.
        """
        result       = list(sequence)
        used_positions = set()
        n_done       = 0

        # First pass: mutate underexplored positions using posterior
        for pos in underexplored_positions:
            if n_done >= n_mutations: break
            if pos in used_positions: continue
            entry      = self.feasibility_map[f"pos_{pos}"]
            current_aa = result[pos]
            candidates = [(aa, self._posterior_prob(pos, aa))
                          for aa in entry["probabilities"]
                          if aa != current_aa]
            if not candidates: continue
            aas, ws = zip(*candidates)
            total   = sum(ws)
            chosen  = random.choices(aas, weights=[w/total for w in ws], k=1)[0]
            result[pos] = chosen
            used_positions.add(pos)
            n_done += 1

        # Second pass: fill remaining mutations normally
        while n_done < n_mutations:
            mutation = self._sample_mutation("".join(result), exclude_positions=used_positions)
            if mutation is None: break
            pos, aa = mutation
            result[pos] = aa
            used_positions.add(pos)
            n_done += 1

        return "".join(result)

    def update(self, parent_candidate, child_candidate):
        """
        Update success/attempt counts based on observed outcome.
        A mutation is a 'success' if the child is not dominated by the parent
        (i.e., it found a genuinely non-worse solution on at least one axis).
        """

        parent_seq = parent_candidate["sequence"]
        child_seq  = child_candidate["sequence"]

        # Find which positions changed
        changed = [(i, child_seq[i]) for i in self.mutable_positions
                   if parent_seq[i] != child_seq[i]]

        if not changed: return

        # Success: child is not dominated by parent (it's at least as good somewhere)
        success = not dominates(parent_candidate, child_candidate)

        for pos, aa in changed:
            key = (pos, aa)
            self._attempts[key]  = self._attempts.get(key, 0)  + 1.0
            self._successes[key] = self._successes.get(key, 0) + (1.0 if success else 0.0)

        self.total_updates += 1

    def top_learned_mutations(self, n=5):
        """Returns the n highest-posterior (position, residue) pairs -- for audit logging."""
        posteriors = {k: self._posterior_prob(k[0], k[1]) for k in self._successes}
        return sorted(posteriors.items(), key=lambda x: x[1], reverse=True)[:n]
