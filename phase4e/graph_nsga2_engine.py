"""graph_nsga2_engine.py - Enhanced NSGA-II loop that combines the fitness graph and learned perturbation model with a stagnation escape mechanism."""
import os, random, sys


# Re-use NSGA-II primitives from original engine unchanged

# ── Hyperparameters ────────────────────────────────────────────────────────
LOCAL_OPT_THRESHOLD   = 0.5   # fraction of top population that must be local
                               # optima to trigger perturbation (0.5 = majority)
PERTURBATION_SIZE     = 8     # candidates replaced by learned perturbations
PERTURBATION_COOLDOWN = 8     # generations before perturbation can re-fire
N_MUTATIONS_NORMAL    = 2     # mutations per offspring in normal operation
N_MUTATIONS_ESCAPE    = 3     # mutations per offspring during escape event


class GraphNSGA2Engine:
    def __init__(self, feasibility_map, mask, model, baseline_growth,
                 population_size=30, n_generations=50,
                 checkpoint_dir="./phase4_checkpoints/enhanced"):

        self.feasibility_map  = feasibility_map
        self.mask             = mask
        self.model            = model
        self.baseline_growth  = baseline_growth
        self.population_size  = population_size
        self.n_generations    = n_generations
        self.checkpoint_dir   = checkpoint_dir
        self.mutable_positions= get_mutable_positions(mask)

        self.graph      = FitnessGraph(self.mutable_positions, max_edge_distance=2)
        self.perturbation = LearnedPerturbation(feasibility_map, mask,
                                                self.mutable_positions)
        self._fba_cache = {}
        self._cooldown  = 0
        self._escape_count = 0
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _mpb(self, sequence):
        if sequence not in self._fba_cache:
            r = evaluate_candidate(self.model, sequence, self.baseline_growth)
            self._fba_cache[sequence] = r["growth_percent_reduction"]
        return self._fba_cache[sequence]

    def _make_candidate(self, sequence, generation):
        return {
            "sequence":          sequence,
            "structural_score":  compute_structural_score(sequence, self.feasibility_map),
            "mpb_score":         self._mpb(sequence),
            "generation":        generation,
            "rank":              None,
            "crowding_distance": 0.0,
        }

    def _eval_pop(self, sequences, generation):
        return [self._make_candidate(s, generation) for s in sequences]

    def _init_population(self):
        seqs = {WILD_TYPE}
        while len(seqs) < self.population_size:
            seqs.add(sample_sequence(self.feasibility_map, self.mask))
        return list(seqs)

    def _detect_local_optima(self, population):
        """
        Returns True if a majority of the top-ranked population members
        are local optima in the fitness landscape graph.
        """
        # Only check top half (front 0) to avoid triggering on genuinely
        # inferior candidates that are correctly stuck
        top_half = sorted(population, key=lambda c: c.get("rank", 99))
        top_half = top_half[:max(1, len(top_half)//2)]
        local_opt_count = sum(1 for c in top_half if self.graph.is_local_optimum(c))
        return (local_opt_count / len(top_half)) >= LOCAL_OPT_THRESHOLD

    def _fire_escape(self, population, generation):
        """
        Replace PERTURBATION_SIZE worst-ranked candidates with learned
        perturbations directed at underexplored graph regions.
        Unlike random injection, these are posterior-weighted mutations
        at positions the graph identifies as least-explored -- giving
        them a real chance to survive selection pressure.
        """
        underexplored = self.graph.get_underexplored_positions(population, top_k=5)

        escaped_seqs = []
        # Pick diverse parents from the Pareto front to perturb from
        front0 = [c for c in population if c.get("rank", 99) == 0]
        parents = random.choices(front0 if front0 else population,
                                 k=PERTURBATION_SIZE)
        for parent in parents:
            new_seq = self.perturbation.perturb_toward_unexplored(
                parent["sequence"], underexplored, n_mutations=N_MUTATIONS_ESCAPE
            )
            escaped_seqs.append(new_seq)

        escaped = self._eval_pop(escaped_seqs, generation)

        # Replace worst-ranked candidates
        sorted_pop = sorted(population, key=lambda c: (c.get("rank",99),
                                                        -c.get("crowding_distance",0)))
        new_pop = sorted_pop[:-PERTURBATION_SIZE] + escaped
        self._cooldown     = PERTURBATION_COOLDOWN
        self._escape_count += 1
        return new_pop

    def run(self, pareto_archive=None):
        if pareto_archive is None: pareto_archive = ParetoArchive()

        print(f"\n{'='*72}")
        print(f"  Graph-NSGA-II + Learned Perturbation")
        print(f"  pop={self.population_size} | gen={self.n_generations}")
        print(f"  Local optima threshold={LOCAL_OPT_THRESHOLD} | "
              f"Escape size={PERTURBATION_SIZE} | Cooldown={PERTURBATION_COOLDOWN}")
        print(f"{'='*72}")

        population = self._eval_pop(self._init_population(), 0)
        self.graph.add_candidates(population)
        pareto_archive.update(population)
        history = []

        for gen in range(1, self.n_generations + 1):

            # Sort + crowding
            fronts = fast_non_dominated_sort(population)
            for f in fronts: crowding_distance(f, population)

            # Generate offspring via learned perturbation
            offspring_seqs = []
            parent_map     = {}   # child_seq -> parent_candidate for update()
            for _ in range(self.population_size):
                parent  = tournament_select(population)
                n_muts  = N_MUTATIONS_NORMAL
                child_seq = self.perturbation.perturb(parent["sequence"], n_mutations=n_muts)
                offspring_seqs.append(child_seq)
                parent_map[child_seq] = parent

            offspring = self._eval_pop(offspring_seqs, gen)
            self.graph.add_candidates(offspring)

            # Update learned perturbation posterior based on outcomes
            for child in offspring:
                parent = parent_map.get(child["sequence"])
                if parent: self.perturbation.update(parent, child)

            # Survivor selection
            population = select_survivors(population + offspring, self.population_size)
            pareto_archive.update(population)

            # Local optima detection + escape
            escape_tag = ""
            if self._cooldown > 0:
                self._cooldown -= 1
            else:
                fronts2 = fast_non_dominated_sort(population)
                for f in fronts2: crowding_distance(f, population)
                if self._detect_local_optima(population):
                    population = self._fire_escape(population, gen)
                    pareto_archive.update(population)
                    escape_tag = f" [ESCAPE #{self._escape_count} FIRED]"

            best_struct = max(c["structural_score"] for c in population)
            best_mpb    = min(c["mpb_score"]        for c in population)
            top_learned = self.perturbation.top_learned_mutations(n=1)
            top_tag     = f"top_learned=({top_learned[0][0][0]},{top_learned[0][0][1]}:{top_learned[0][1]:.2f})" if top_learned else ""

            history.append({
                "generation":        gen,
                "pareto_front_size": len(pareto_archive.front),
                "cache_size":        len(self._fba_cache),
                "graph_size":        self.graph.size,
                "best_structural":   best_struct,
                "best_mpb":          best_mpb,
                "perturbation_updates": self.perturbation.total_updates,
                "escape_count":      self._escape_count,
            })

            print(f"  gen {gen:03d} | pareto={len(pareto_archive.front):3d} | "
                  f"cache={len(self._fba_cache):4d} | graph={self.graph.size:4d} | "
                  f"struct={best_struct:.3f} | mpb={best_mpb:.4f}% | "
                  f"{top_tag}{escape_tag}")

            if gen % 10 == 0:
                ckpt = os.path.join(self.checkpoint_dir, f"checkpoint_gen_{gen:03d}.json")
                pareto_archive.checkpoint(ckpt, {"generation": gen, "history": history})
                print(f"  [CHECKPOINT] {ckpt}")

        print(f"\n  Done. Pareto={len(pareto_archive.front)} | "
              f"FBA={len(self._fba_cache)} | Graph={self.graph.size} | "
              f"Escapes={self._escape_count} | "
              f"Perturbation updates={self.perturbation.total_updates}")
        return pareto_archive, history
