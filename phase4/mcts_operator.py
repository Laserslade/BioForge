"""mcts_operator.py - Two-phase Monte Carlo tree search mutation operator used to propose improved candidate sequences."""
import math, random

UCB1_C       = math.sqrt(2)
STEP1_FRACTION = 0.6   # fraction of rollout budget used for first mutation


class _Node:
    __slots__ = ["sequence","mutation","parent","children","visits","total_reward","_untried"]
    def __init__(self, sequence, mutation=None, parent=None):
        self.sequence     = sequence
        self.mutation     = mutation
        self.parent       = parent
        self.children     = []
        self.visits       = 0
        self.total_reward = 0.0
        self._untried     = None

    @property
    def mean_reward(self):
        return self.total_reward / self.visits if self.visits > 0 else 0.0

    def ucb1(self):
        if self.visits == 0: return float("inf")
        if not self.parent or self.parent.visits == 0: return self.mean_reward
        return (self.mean_reward +
                UCB1_C * math.sqrt(math.log(self.parent.visits) / self.visits))

    def is_fully_expanded(self): return self._untried is not None and len(self._untried) == 0
    def best_ucb_child(self):    return max(self.children, key=lambda c: c.ucb1())
    def best_mean_child(self):   return max(self.children, key=lambda c: c.mean_reward) if self.children else None


class MCTSOperator:
    def __init__(self, feasibility_map, mask, n_rollouts=50):
        self.feasibility_map = feasibility_map
        self.mask            = mask
        self.n_rollouts      = n_rollouts
        self._mutable        = [int(k.split("_")[1])
                                 for k,v in mask.items() if v["status"]=="MUTABLE"]

    def _build_untried(self, sequence, exclude_positions=None):
        """Weighted-shuffle of (pos, residue) candidates, optionally excluding positions."""
        exclude_positions = exclude_positions or set()
        candidates = []
        for pos in self._mutable:
            if pos in exclude_positions: continue
            entry = self.feasibility_map[f"pos_{pos}"]
            for aa, prob in entry["probabilities"].items():
                if aa != sequence[pos]:
                    candidates.append((pos, aa, prob))
        if not candidates: return []
        total = sum(c[2] for c in candidates)
        candidates.sort(
            key=lambda c: -math.log(max(random.random(), 1e-10)) / (c[2] / total)
        )
        return [(pos, aa) for pos, aa, _ in candidates]

    def _run_mcts(self, sequence, n_rollouts, exclude_positions=None):
        """Single-level MCTS returning the best mutated sequence found."""
        root = _Node(sequence); root.visits = 1

        for _ in range(n_rollouts):
            node = root
            # Selection
            while node.is_fully_expanded() and node.children:
                node = node.best_ucb_child()
            # Expansion
            if not node.is_fully_expanded():
                if node._untried is None:
                    node._untried = self._build_untried(node.sequence, exclude_positions)
                if node._untried:
                    pos, aa = node._untried.pop(0)
                    child = _Node(apply_mutation(node.sequence, pos, aa), (pos,aa), node)
                    node.children.append(child); node = child
            # Rollout (instant structural score)
            reward = compute_structural_score(node.sequence, self.feasibility_map)
            # Backprop
            cur = node
            while cur:
                cur.visits += 1; cur.total_reward += reward; cur = cur.parent

        best = root.best_mean_child()
        return best.sequence if best else sequence, best.mutation if best else None

    def mutate(self, parent_sequence):
        """
        Two sequential depth-1 MCTS calls producing a 2-mutation offspring.
        Step 1 finds the best single mutation; Step 2 finds the best second
        mutation from that result, constrained to a different position.
        """
        n1 = max(1, int(self.n_rollouts * STEP1_FRACTION))
        n2 = max(1, self.n_rollouts - n1)

        # Step 1: best single mutation
        mid_seq, first_mutation = self._run_mcts(parent_sequence, n1)

        # Step 2: best second mutation, excluding the position already changed
        already_mutated = set()
        if first_mutation: already_mutated.add(first_mutation[0])

        result_seq, _ = self._run_mcts(mid_seq, n2, exclude_positions=already_mutated)
        return result_seq
