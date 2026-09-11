"""nsga2_engine.py - Core NSGA-II multi-objective evolutionary optimization loop over structural fitness and metabolic cost."""
import os, random, math

# ── Stagnation / boost parameters ─────────────────────────────────────────
STAGNATION_WINDOW  = 5     # generations without improvement to declare stagnation
ENTROPY_THRESHOLD  = 1.5   # bits; max possible = log2(20) ≈ 4.32; below this = low diversity
BOOST_ROLLOUTS     = 150   # MCTS rollouts during boost (3x the default 50)
BOOST_DURATION     = 5     # generations to run boosted MCTS after trigger
BOOST_INJECT_COUNT = 5     # random sequences injected into population on trigger
BOOST_COOLDOWN     = 10    # generations before boost can re-fire


# ── NSGA-II primitives ─────────────────────────────────────────────────────

def fast_non_dominated_sort(population):
    n = len(population)
    dom_count = [0]*n
    dom_set   = [[] for _ in range(n)]
    fronts    = [[]]
    for i in range(n):
        for j in range(n):
            if i==j: continue
            if dominates(population[i], population[j]):   dom_set[i].append(j)
            elif dominates(population[j], population[i]): dom_count[i] += 1
        if dom_count[i] == 0:
            population[i]["rank"] = 0; fronts[0].append(i)
    k = 0
    while fronts[k]:
        nxt = []
        for i in fronts[k]:
            for j in dom_set[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    population[j]["rank"] = k+1; nxt.append(j)
        k += 1; fronts.append(nxt)
    return [f for f in fronts if f]


def crowding_distance(front_indices, population):
    n = len(front_indices)
    for idx in front_indices: population[idx]["crowding_distance"] = 0.0
    if n <= 2:
        for idx in front_indices: population[idx]["crowding_distance"] = float("inf")
        return
    for obj in ["structural_score", "mpb_score"]:
        srt = sorted(front_indices, key=lambda i: population[i][obj])
        population[srt[0]]["crowding_distance"]  = float("inf")
        population[srt[-1]]["crowding_distance"] = float("inf")
        rng = population[srt[-1]][obj] - population[srt[0]][obj]
        if rng == 0: continue
        for i in range(1, n-1):
            population[srt[i]]["crowding_distance"] += (
                population[srt[i+1]][obj] - population[srt[i-1]][obj]) / rng


def tournament_select(population):
    a, b = random.sample(population, 2)
    ra, rb = a.get("rank",float("inf")), b.get("rank",float("inf"))
    if ra < rb: return a
    if rb < ra: return b
    return a if a.get("crowding_distance",0) >= b.get("crowding_distance",0) else b


def select_survivors(combined, target_size):
    fronts = fast_non_dominated_sort(combined)
    for f in fronts: crowding_distance(f, combined)
    survivors = []
    for f in fronts:
        if len(survivors) + len(f) <= target_size:
            survivors.extend([combined[i] for i in f])
        else:
            rem = target_size - len(survivors)
            srt = sorted(f, key=lambda i: combined[i].get("crowding_distance",0), reverse=True)
            survivors.extend([combined[i] for i in srt[:rem]]); break
    return survivors


# ── Engine ─────────────────────────────────────────────────────────────────

class NSGA2Engine:
    def __init__(self, feasibility_map, mask, model, baseline_growth,
                 population_size=30, n_generations=50, n_mcts_rollouts=50,
                 checkpoint_dir="./phase4_checkpoints", use_mcts=True):

        self.feasibility_map  = feasibility_map
        self.mask             = mask
        self.model            = model
        self.baseline_growth  = baseline_growth
        self.population_size  = population_size
        self.n_generations    = n_generations
        self.base_rollouts    = n_mcts_rollouts
        self.checkpoint_dir   = checkpoint_dir
        self.use_mcts         = use_mcts
        self.mutable_positions= get_mutable_positions(mask)

        self.mcts        = MCTSOperator(feasibility_map, mask, n_mcts_rollouts)
        self._fba_cache  = {}
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Stagnation / boost state
        self._fitness_history  = []   # (best_struct, best_mpb) per generation
        self._boost_remaining  = 0    # generations left in current boost
        self._cooldown_remaining = 0  # generations until boost can re-fire
        self._boost_count      = 0    # total times boost has fired (for audit)

    # ── Evaluation ──────────────────────────────────────────────────────────

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

    # ── Mutation ─────────────────────────────────────────────────────────────

    def _mutate(self, parent_seq):
        if self.use_mcts:
            return self.mcts.mutate(parent_seq)
        pos     = random.choice(self.mutable_positions)
        entry   = self.feasibility_map[f"pos_{pos}"]
        choices = [aa for aa in entry["probabilities"] if aa != parent_seq[pos]]
        return apply_mutation(parent_seq, pos, random.choice(choices)) if choices else parent_seq

    # ── Stagnation detection & adaptive boost ────────────────────────────────

    def _check_stagnated(self):
        """
        Returns True if neither best_struct nor best_mpb has improved
        beyond epsilon in the last STAGNATION_WINDOW generations.
        """
        if len(self._fitness_history) < STAGNATION_WINDOW:
            return False
        window   = self._fitness_history[-STAGNATION_WINDOW:]
        best_struct_old, best_mpb_old = window[0]
        best_struct_now, best_mpb_now = window[-1]
        struct_improved = (best_struct_now - best_struct_old) > 0.01
        mpb_improved    = (best_mpb_old - best_mpb_now)       > 0.001
        return not struct_improved and not mpb_improved

    def _maybe_fire_boost(self, population, generation):
        """
        Fires adaptive diversity boost when entropy < threshold AND
        stagnated AND cooldown has expired. Returns (population, boosted: bool).
        """
        if self._cooldown_remaining > 0:
            return population, False

        entropy   = compute_population_entropy(population, self.mutable_positions)
        stagnated = self._check_stagnated()

        if entropy >= ENTROPY_THRESHOLD or not stagnated:
            return population, False

        # ── FIRE BOOST ───────────────────────────────────────────────────
        self._boost_remaining    = BOOST_DURATION
        self._cooldown_remaining = BOOST_COOLDOWN
        self._boost_count       += 1
        self.mcts.n_rollouts     = BOOST_ROLLOUTS

        # Inject fresh random sequences, replacing lowest-ranked survivors
        injected_seqs  = [sample_sequence(self.feasibility_map, self.mask)
                          for _ in range(BOOST_INJECT_COUNT)]
        injected       = self._eval_pop(injected_seqs, generation)
        sorted_pop     = sorted(population,
                                key=lambda c: (c.get("rank", 99),
                                               -c.get("crowding_distance", 0)))
        combined = sorted_pop[:-BOOST_INJECT_COUNT] + injected
        return combined, True

    def _update_boost_state(self):
        """Called once per generation to tick down boost and cooldown counters."""
        if self._boost_remaining > 0:
            self._boost_remaining -= 1
            if self._boost_remaining == 0:
                self.mcts.n_rollouts = self.base_rollouts   # restore normal rollouts
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(self, pareto_archive=None):
        if pareto_archive is None: pareto_archive = ParetoArchive()

        mode = "MCTS" if self.use_mcts else "RANDOM"
        print(f"\n{'='*72}")
        print(f"  NSGA-II [{mode}] | pop={self.population_size} | gen={self.n_generations}")
        print(f"  Entropy threshold={ENTROPY_THRESHOLD} bits | "
              f"Stagnation window={STAGNATION_WINDOW} gen | "
              f"Boost rollouts={BOOST_ROLLOUTS} for {BOOST_DURATION} gen")
        print(f"{'='*72}")

        population = self._eval_pop(self._init_population(), 0)
        pareto_archive.update(population)
        history = []

        for gen in range(1, self.n_generations + 1):

            # Sort current population
            fronts = fast_non_dominated_sort(population)
            for f in fronts: crowding_distance(f, population)

            # Generate offspring
            offspring_seqs = [
                self._mutate(tournament_select(population)["sequence"])
                for _ in range(self.population_size)
            ]
            offspring  = self._eval_pop(offspring_seqs, gen)
            population = select_survivors(population + offspring, self.population_size)
            pareto_archive.update(population)

            # Compute metrics
            best_struct = max(c["structural_score"] for c in population)
            best_mpb    = min(c["mpb_score"]        for c in population)
            entropy     = compute_population_entropy(population, self.mutable_positions)
            self._fitness_history.append((best_struct, best_mpb))

            # Adaptive boost check
            population, boosted = self._maybe_fire_boost(population, gen)
            boost_tag = f" [BOOST #{self._boost_count} FIRED]" if boosted else (
                        f" [boost {self._boost_remaining}gen left]" if self._boost_remaining > 0 else "")

            self._update_boost_state()

            history.append({
                "generation":        gen,
                "pareto_front_size": len(pareto_archive.front),
                "cache_size":        len(self._fba_cache),
                "best_structural":   best_struct,
                "best_mpb":          best_mpb,
                "entropy_bits":      round(entropy, 4),
                "boost_fired":       boosted,
                "total_boosts":      self._boost_count,
            })

            print(f"  gen {gen:03d} | pareto={len(pareto_archive.front):3d} | "
                  f"cache={len(self._fba_cache):4d} | "
                  f"struct={best_struct:.3f} | mpb={best_mpb:.4f}% | "
                  f"H={entropy:.2f}b{boost_tag}")

            if gen % 10 == 0:
                ckpt = os.path.join(self.checkpoint_dir, f"checkpoint_gen_{gen:03d}.json")
                pareto_archive.checkpoint(ckpt, {"generation": gen, "history": history})
                print(f"  [CHECKPOINT] {ckpt}")

        print(f"\n  Done. Pareto={len(pareto_archive.front)} | "
              f"FBA calls={len(self._fba_cache)} | "
              f"Total boosts fired={self._boost_count}")
        return pareto_archive, history
