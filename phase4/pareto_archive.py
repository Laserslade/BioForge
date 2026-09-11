"""pareto_archive.py - Tracks and persists the non-dominated Pareto front of candidate sequences across generations."""
import os, json
from datetime import datetime, timezone


class ParetoArchive:
    def __init__(self):
        self.front = []          # current non-dominated set
        self.all_evaluated = []  # every candidate ever seen (for post-analysis)

    def update(self, population):
        """Merge new candidates into the Pareto front, deduplicating by sequence."""
        self.all_evaluated.extend(population)
        combined = self.front + population

        new_front = []
        for i, cand in enumerate(combined):
            if not any(dominates(combined[j], cand) for j in range(len(combined)) if j!=i):
                new_front.append(cand)

        # Deduplicate by sequence, keeping latest version of each
        seen, deduped = set(), []
        for c in reversed(new_front):
            if c["sequence"] not in seen:
                seen.add(c["sequence"]); deduped.append(c)
        self.front = deduped

    def checkpoint(self, path, metadata=None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "timestamp":    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "pareto_front": self.front,
                "metadata":     metadata or {},
            }, f, indent=2)

    def save_final(self, path="./pipeline_artifacts/phase4_pareto_front.json"):
        self.checkpoint(path, {"type": "final_pareto_front",
                                "n_candidates": len(self.front)})
        return path

    @classmethod
    def load_checkpoint(cls, path):
        archive = cls()
        with open(path) as f: data = json.load(f)
        archive.front = data["pareto_front"]
        return archive
