"""divergence_analyzer.py - Compares the FBA and proxy Pareto fronts to quantify how much the two objectives diverge."""

import json
import numpy as np
from scipy import stats
from collections import Counter

AA_BIOSYNTHETIC_COST = {
    'A':  11.7, 'R':  27.3, 'N':  14.7, 'D':  12.8, 'C':  24.7,
    'Q':  16.3, 'E':  15.3, 'G':  11.7, 'H':  38.8, 'I':  32.7,
    'L':  27.3, 'K':  30.3, 'M':  34.3, 'F':  52.0, 'P':  20.3,
    'S':  18.7, 'T':  18.7, 'W':  74.3, 'Y':  50.0, 'V':  23.3,
}
EXPENSIVE_AA  = {'W', 'Y', 'F', 'M', 'H', 'C', 'R'}
WILD_TYPE     = 'GFGCNGPWDEDDMQCHNHCKSIKGYKGGYCAKGGFVCKCY'


def run_divergence_analysis():
    print('=' * 70)
    print('  PHASE 5 - DIVERGENCE ANALYSIS: FBA vs Physicochemical Proxy')
    print('=' * 70)

    # Load both fronts
    with open('./pipeline_artifacts/phase4_pareto_front.json') as f:
        fba_data  = json.load(f)
    with open('./pipeline_artifacts/phase5_proxy_pareto_front.json') as f:
        proxy_data = json.load(f)

    fba_front   = fba_data.get('pareto_front', fba_data if isinstance(fba_data, list) else [])
    proxy_front = proxy_data.get('pareto_front', proxy_data if isinstance(proxy_data, list) else [])

    fba_seqs   = {c['sequence'] for c in fba_front}
    proxy_seqs = {c['sequence'] for c in proxy_front}

    print(f'\n  FBA front size   : {len(fba_front)}')
    print(f'  Proxy front size : {len(proxy_front)}')

    # ── 1. Sequence Overlap ───────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('CHECK 1: Sequence Overlap (Jaccard Similarity)')
    print('=' * 70)
    intersection = fba_seqs & proxy_seqs
    union        = fba_seqs | proxy_seqs
    jaccard      = len(intersection) / len(union) if union else 0
    only_fba     = fba_seqs   - proxy_seqs
    only_proxy   = proxy_seqs - fba_seqs
    print(f'  Sequences in both fronts      : {len(intersection)}')
    print(f'  Sequences only in FBA front   : {len(only_fba)}')
    print(f'  Sequences only in proxy front : {len(only_proxy)}')
    print(f'  Jaccard similarity            : {jaccard:.4f}  (0=no overlap, 1=identical)')
    if jaccard < 0.2:
        print('   LOW overlap - the two objectives select substantially different sequences.')
    elif jaccard < 0.5:
        print('  ~ MODERATE overlap - objectives partially agree but diverge meaningfully.')
    else:
        print('   HIGH overlap - objectives largely agree; divergence may be weak.')

    # ── 2. Expensive AA Analysis ──────────────────────────────────────────
    print('\n' + '=' * 70)
    print('CHECK 2: Expensive Amino Acid Content (W, Y, F, M, H, C, R)')
    print('=' * 70)

    def expensive_count(seq):
        return sum(1 for aa in seq if aa in EXPENSIVE_AA)

    def true_cost(seq):
        return sum(AA_BIOSYNTHETIC_COST.get(aa, 20.0) for aa in seq)

    fba_exp   = [expensive_count(c['sequence']) for c in fba_front]
    proxy_exp = [expensive_count(c['sequence']) for c in proxy_front]
    fba_cost_true   = [true_cost(c['sequence']) for c in fba_front]
    proxy_cost_true = [true_cost(c['sequence']) for c in proxy_front]

    wt_exp  = expensive_count(WILD_TYPE)
    wt_cost = true_cost(WILD_TYPE)

    print(f'  Wild-type: {wt_exp} expensive AAs | true biosyn cost: {wt_cost:.1f} ATP eq')
    print(f'\n  FBA front   - mean expensive AAs: {np.mean(fba_exp):.2f} (±{np.std(fba_exp):.2f})')
    print(f'                mean true cost:     {np.mean(fba_cost_true):.1f} ATP eq (±{np.std(fba_cost_true):.1f})')
    print(f'\n  Proxy front - mean expensive AAs: {np.mean(proxy_exp):.2f} (±{np.std(proxy_exp):.2f})')
    print(f'                mean true cost:     {np.mean(proxy_cost_true):.1f} ATP eq (±{np.std(proxy_cost_true):.1f})')

    # Mann-Whitney U test
    u_stat, p_val = stats.mannwhitneyu(fba_exp, proxy_exp, alternative='less')
    print(f'\n  Mann-Whitney U (FBA < Proxy on expensive AA count):')
    print(f'    U={u_stat:.1f}  p={p_val:.4f}')
    if p_val < 0.05:
        print('   SIGNIFICANT: FBA front has statistically fewer expensive AAs than proxy front.')
        print('    This is the key empirical result - FBA sees what proxies miss.')
    else:
        print('   Not significant at p<0.05 - expensive AA distributions overlap.')

    # ── 3. Per-AA composition comparison ─────────────────────────────────
    print('\n' + '=' * 70)
    print('CHECK 3: Per-AA Frequency Shift (FBA vs Proxy, mutable positions only)')
    print('=' * 70)
    with open('./pipeline_artifacts/phase2_mask.json') as f:
        mask = json.load(f)
    mutable_pos = sorted(int(k.split('_')[1]) for k, v in mask.items() if v['status'] == 'MUTABLE')

    def aa_freq(front):
        counts = Counter()
        total  = 0
        for c in front:
            for pos in mutable_pos:
                counts[c['sequence'][pos]] += 1
                total += 1
        return {aa: counts[aa]/total for aa in 'ACDEFGHIKLMNPQRSTVWY'}

    fba_freq   = aa_freq(fba_front)
    proxy_freq = aa_freq(proxy_front)

    print(f'  {"AA":<4} {"FBA freq":>10} {"Proxy freq":>12} {"Δ (FBA-Proxy)":>14} {"Biosyn cost":>12}')
    print(f'  {"-"*58}')
    diffs = []
    for aa in sorted('ACDEFGHIKLMNPQRSTVWY', key=lambda a: -AA_BIOSYNTHETIC_COST[a]):
        d    = fba_freq[aa] - proxy_freq[aa]
        flag = '  expensive' if aa in EXPENSIVE_AA else ''
        diffs.append((aa, d))
        print(f'  {aa:<4} {fba_freq[aa]:>10.4f} {proxy_freq[aa]:>12.4f} '
              f'{d:>+14.4f}  ({AA_BIOSYNTHETIC_COST[aa]:>5.1f} ATP){flag}')

    # ── 4. Proxy-FBA cost correlation ─────────────────────────────────────
    print('\n' + '=' * 70)
    print('CHECK 4: Correlation Between FBA Cost and True Biosynthetic Cost')
    print('=' * 70)
    all_seqs  = list({c['sequence'] for c in fba_front + proxy_front})
    all_fba   = [next((c['mpb_score'] for c in fba_front if c['sequence'] == s), None)
                 for s in all_seqs]
    all_true  = [true_cost(s) for s in all_seqs]
    all_proxy = [next((c['mpb_score'] for c in proxy_front if c['sequence'] == s), None)
                 for s in all_seqs]

    # FBA vs true cost correlation
    pairs_fba = [(f, t) for f, t in zip(all_fba, all_true) if f is not None]
    if len(pairs_fba) >= 3:
        r_fba, p_fba = stats.pearsonr([p[0] for p in pairs_fba], [p[1] for p in pairs_fba])
        print(f'  FBA MPB vs true biosyn cost: r={r_fba:.4f}  p={p_fba:.4f}')
        print(f'     FBA cost {"correlates" if abs(r_fba) > 0.5 else "does NOT strongly correlate"} with true ATP cost')

    # Proxy vs true cost correlation
    pairs_proxy = [(p, t) for p, t in zip(all_proxy, all_true) if p is not None]
    if len(pairs_proxy) >= 3:
        r_prx, p_prx = stats.pearsonr([p[0] for p in pairs_proxy], [p[1] for p in pairs_proxy])
        print(f'  Proxy cost vs true biosyn cost: r={r_prx:.4f}  p={p_prx:.4f}')
        print(f'     Proxy cost {"correlates" if abs(r_prx) > 0.5 else "does NOT strongly correlate"} with true ATP cost')

    # ── 5. Sequences unique to proxy front (the blind spots) ──────────────
    print('\n' + '=' * 70)
    print('CHECK 5: Sequences in Proxy Front NOT in FBA Front (Proxy Blind Spots)')
    print('=' * 70)
    print('  These are candidates the proxy rated highly but FBA would reject')
    print('  as metabolically expensive. They reveal what proxies miss.')
    print()
    blind_spots = [c for c in proxy_front if c['sequence'] in only_proxy]
    blind_spots.sort(key=lambda c: true_cost(c['sequence']), reverse=True)
    for c in blind_spots[:5]:
        tc   = true_cost(c['sequence'])
        ec   = expensive_count(c['sequence'])
        exp_aas = [aa for aa in c['sequence'] if aa in EXPENSIVE_AA]
        print(f'  {c["sequence"]}')
        print(f'    proxy_cost={c["mpb_score"]:.4f}  true_cost={tc:.1f} ATP  '
              f'expensive_AAs={ec} ({" ".join(exp_aas)})')
        print()

    # ── Summary ───────────────────────────────────────────────────────────
    print('=' * 70)
    print('  DIVERGENCE ANALYSIS - SUMMARY')
    print('=' * 70)
    print(f'  Jaccard similarity          : {jaccard:.4f}')
    print(f'  FBA   mean expensive AAs    : {np.mean(fba_exp):.2f}')
    print(f'  Proxy mean expensive AAs    : {np.mean(proxy_exp):.2f}')
    print(f'  FBA   mean true cost        : {np.mean(fba_cost_true):.1f} ATP eq')
    print(f'  Proxy mean true cost        : {np.mean(proxy_cost_true):.1f} ATP eq')
    print(f'  Mann-Whitney p-value        : {p_val:.4f}')
    print(f'  Proxy blind spots (n)       : {len(only_proxy)}')
    print()
    if p_val < 0.05 and jaccard < 0.5:
        print('   STRONG DIVERGENCE: FBA objective selects cheaper sequences')
        print('    in ways that physicochemical proxies cannot replicate.')
        print('    This validates the FBA-as-objective framework contribution.')
    elif p_val < 0.1 or jaccard < 0.3:
        print('  ~ MODERATE DIVERGENCE: Some evidence FBA and proxy differ.')
        print('    Report the direction and magnitude honestly.')
    else:
        print('   WEAK DIVERGENCE: FBA and proxy select similar sequences.')
        print('    The framework contribution claim is weakened. Consider')
        print('    refining the FBA objective or switching to shadow-price scoring.')
    print('=' * 70)

    # Save summary
    summary = {
        'jaccard_similarity':      jaccard,
        'fba_mean_expensive_aa':   float(np.mean(fba_exp)),
        'proxy_mean_expensive_aa': float(np.mean(proxy_exp)),
        'fba_mean_true_cost':      float(np.mean(fba_cost_true)),
        'proxy_mean_true_cost':    float(np.mean(proxy_cost_true)),
        'mannwhitney_p':           float(p_val),
        'n_only_fba':              len(only_fba),
        'n_only_proxy':            len(only_proxy),
        'n_intersection':          len(intersection),
        'aa_freq_fba':             fba_freq,
        'aa_freq_proxy':           proxy_freq,
    }
    with open('./pipeline_artifacts/phase5_divergence_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print('  Saved: ./pipeline_artifacts/phase5_divergence_summary.json')


print(' divergence_analyzer.py defined.')


#  RUN
run_divergence_analysis()
