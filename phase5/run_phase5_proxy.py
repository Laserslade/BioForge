"""run_phase5_proxy.py - Runs the same NSGA-II plus MCTS optimization with the FBA objective replaced by the physicochemical proxy objective."""

import json, os, copy, time
import numpy as np

PROXY_OUTPUT_PATH = './pipeline_artifacts/phase5_proxy_pareto_front.json'
PROXY_CHECKPOINT_DIR = './phase5_checkpoints/proxy'
os.makedirs(PROXY_CHECKPOINT_DIR, exist_ok=True)


def execute_pipeline_phase_5_proxy():
    print('=' * 72)
    print('  PHASE 5: PROXY NSGA-II RUN (Physicochemical Cost Objective)')
    print('=' * 72)
    print('  Objective 1: Structural score  (ProteinMPNN - same as Phase 4)')
    print('  Objective 2: Proxy cost        (instability index + GRAVY - NO FBA)')
    print('  All other params identical to Phase 4.')
    print()

    # [1/4] Load feasibility map and mask (same as Phase 4)
    print('[1/4] Loading Phase 2 feasibility map and mask...')
    with open('./pipeline_artifacts/phase2_feasibility_map.json') as f:
        feasibility_map = json.load(f)
    with open('./pipeline_artifacts/phase2_mask.json') as f:
        mask = json.load(f)

    WILD_TYPE = 'GFGCNGPWDEDDMQCHNHCKSIKGYKGGYCAKGGFVCKCY'
    mutable_positions = sorted(
        int(k.split('_')[1]) for k, v in mask.items() if v['status'] == 'MUTABLE'
    )
    print(f'      Mutable positions: {len(mutable_positions)}')

    # [2/4] Initialise proxy evaluator (no model loading needed)
    print('[2/4] Initialising proxy evaluator (no metabolic model)...')
    proxy_eval = ProxyEvaluator()
    wt_proxy = proxy_eval.score(WILD_TYPE)
    print(f'      Wild-type proxy cost: {wt_proxy:.4f}')

    # [3/4] Run NSGA-II with proxy objective
    print('[3/4] Running proxy NSGA-II...')
    print()
    print('=' * 72)
    print('  Proxy-NSGA-II [MCTS] | pop=30 | gen=50')
    print('=' * 72)

    mcts_op   = MCTSOperator(feasibility_map, mask)
    archive   = ParetoArchive()
    history   = []
    cache     = {}

    def evaluate_proxy(seq):
        if seq in cache:
            return cache[seq]
        struct = _compute_structural_score(seq, feasibility_map, mutable_positions, WILD_TYPE)
        cost   = proxy_eval.score(seq)
        result = {'sequence': seq, 'structural_score': struct, 'mpb_score': cost,
                  'proxy_cost': cost, 'is_proxy': True}
        cache[seq] = result
        return result

    # Initialise population
    population = []
    seen = set()
    while len(population) < 30:
        seq = _random_feasible(WILD_TYPE, mutable_positions, feasibility_map)
        if seq not in seen:
            seen.add(seq)
            population.append(evaluate_proxy(seq))

    stagnation_counter = 0
    best_struct_prev   = max(c['structural_score'] for c in population)
    boost_gens_left    = 0
    total_boosts       = 0

    for gen in range(1, 51):
        # MCTS offspring
        rollouts = 150 if boost_gens_left > 0 else 30
        offspring = []
        mcts_op.n_rollouts = rollouts
        for parent in population:
            seq = mcts_op.mutate(parent['sequence'])
            if seq not in seen:
                seen.add(seq)
                offspring.append(evaluate_proxy(seq))

        combined  = population + offspring
        fronts    = fast_non_dominated_sort(combined)
        next_pop  = []
        for front in fronts:
            if len(next_pop) + len(front) <= 30:
                next_pop.extend(front)
            else:
                needed = 30 - len(next_pop)
                ranked = crowding_distance_sort(front)
                next_pop.extend(ranked[:needed])
                break
        population = next_pop

        for c in population:
            archive.update(c)

        best_struct = max(c['structural_score'] for c in population)
        best_cost   = min(c['mpb_score'] for c in population)
        entropy     = _sequence_entropy(population, mutable_positions)

        if abs(best_struct - best_struct_prev) < 0.01:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
        best_struct_prev = best_struct

        boost_fired = False
        if stagnation_counter >= 5 and entropy < 1.5 and boost_gens_left == 0:
            boost_gens_left = 5
            total_boosts   += 1
            stagnation_counter = 0
            boost_fired = True
        if boost_gens_left > 0:
            boost_gens_left -= 1

        boost_tag = f' [BOOST #{total_boosts} FIRED]' if boost_fired else \
                    (f' [boost {boost_gens_left}gen left]' if boost_gens_left > 0 else '')
        print(f'  gen {gen:03d} | pareto={len(archive.front):3d} | cache={len(cache):4d} | '
              f'struct={best_struct:.3f} | proxy={best_cost:.4f} | H={entropy:.2f}b{boost_tag}')

        history.append({'generation': gen, 'pareto_front_size': len(archive.front),
                        'cache_size': len(cache), 'best_structural': best_struct,
                        'best_proxy_cost': best_cost, 'entropy_bits': entropy})

        if gen % 10 == 0:
            ckpt = {'generation': gen, 'pareto_front': archive.front, 'history': history}
            with open(f'{PROXY_CHECKPOINT_DIR}/checkpoint_gen_{gen:03d}.json', 'w') as f:
                json.dump(ckpt, f, indent=2)
            print(f'  [CHECKPOINT] {PROXY_CHECKPOINT_DIR}/checkpoint_gen_{gen:03d}.json')

    print(f'\n  Done. Pareto={len(archive.front)} | Proxy calls={len(cache)} | Boosts={total_boosts}')

    # [4/4] Save results
    print('[4/4] Saving proxy Pareto front...')
    output = {'pareto_front': archive.front, 'history': history,
              'wild_type_proxy_cost': wt_proxy,
              'objective': 'physicochemical_proxy',
              'proxy_components': 'instability_index + GRAVY (Yang et al. 2024 style)'}
    with open(PROXY_OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    print()
    print('=' * 72)
    print('  PHASE 5 PROXY RUN COMPLETE')
    print('=' * 72)
    print(f'  Pareto front candidates: {len(archive.front)}')
    print(f'  Wild-type proxy cost:    {wt_proxy:.4f}')
    best_s = max(archive.front, key=lambda c: c["structural_score"])
    best_c = min(archive.front, key=lambda c: c["mpb_score"])
    print(f'  Best structural:  {best_s["structural_score"]:.4f} -> {best_s["sequence"]}')
    print(f'  Best proxy cost:  {best_c["mpb_score"]:.4f} -> {best_c["sequence"]}')
    print(f'  Output: {PROXY_OUTPUT_PATH}')
    print('=' * 72)
    return archive, history


def _compute_structural_score(seq, feasibility_map, mutable_positions, wild_type):
    """Reuse Phase 4 structural scoring via feasibility map log-probs."""
    score = 0.0
    for pos in mutable_positions:
        key = f'pos_{pos}'
        if key not in feasibility_map:
            continue
        aa  = seq[pos]
        wt_aa = wild_type[pos]
        probs = feasibility_map[key]['probabilities']
        p     = probs.get(aa, 1e-9)
        p_wt  = probs.get(wt_aa, 1e-9)
        score += np.log(p) - np.log(p_wt)
    return float(score)


def _random_feasible(wild_type, mutable_positions, feasibility_map):
    """Sample a random feasible sequence using the ProteinMPNN prior."""
    seq = list(wild_type)
    for pos in mutable_positions:
        key   = f'pos_{pos}'
        probs = feasibility_map[key]['probabilities']
        aas   = list(probs.keys())
        ps    = np.array([probs[a] for a in aas])
        ps   /= ps.sum()
        seq[pos] = np.random.choice(aas, p=ps)
    return ''.join(seq)


def _sequence_entropy(population, mutable_positions):
    """Mean per-position Shannon entropy across the population."""
    entropies = []
    for pos in mutable_positions:
        counts = {}
        for c in population:
            aa = c['sequence'][pos]
            counts[aa] = counts.get(aa, 0) + 1
        n = len(population)
        h = -sum((v/n) * np.log2(v/n) for v in counts.values() if v > 0)
        entropies.append(h)
    return float(np.mean(entropies))


print(' run_phase5_proxy.py defined.')


#  RUN
proxy_archive, proxy_history = execute_pipeline_phase_5_proxy()
