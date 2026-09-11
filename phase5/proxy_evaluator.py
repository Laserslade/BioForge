"""proxy_evaluator.py - Drop-in replacement for the FBA evaluator that scores manufacturability using physicochemical proxies only."""

from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np

# E. coli biosynthetic cost per amino acid (ATP equivalents, Akashi & Gojobori 2002)
# Used for post-hoc annotation only - NOT used as an objective in proxy run.
AA_BIOSYNTHETIC_COST = {
    'A':  11.7, 'R':  27.3, 'N':  14.7, 'D':  12.8, 'C':  24.7,
    'Q':  16.3, 'E':  15.3, 'G':  11.7, 'H':  38.8, 'I':  32.7,
    'L':  27.3, 'K':  30.3, 'M':  34.3, 'F':  52.0, 'P':  20.3,
    'S':  18.7, 'T':  18.7, 'W':  74.3, 'Y':  50.0, 'V':  23.3,
}
EXPENSIVE_AA = {'W', 'Y', 'F', 'M', 'H', 'C', 'R'}  # top 7 by cost


class ProxyEvaluator:
    """
    Computes proxy manufacturability score from physicochemical properties.
    Returns a cost in [0, 100] - lower is better (cheaper to manufacture
    in proxy terms). Designed to be interface-compatible with FBAEvaluator.
    """

    def __init__(self):
        self._cache = {}
        self._calls = 0

    def compute_proxy_cost(self, sequence: str) -> dict:
        """Compute proxy cost and component scores for a sequence."""
        if sequence in self._cache:
            return self._cache[sequence]

        self._calls += 1
        analysis = ProteinAnalysis(sequence)

        instability = analysis.instability_index()   # 0-100+, lower = more stable
        gravy       = analysis.gravy()                # negative = hydrophilic
        aliphatic   = _aliphatic_index(sequence)      # higher = more aliphatic

        # Proxy cost: penalise instability and hydrophobicity.
        # Normalised so typical values land in [0, 100].
        proxy_cost = (
            0.5 * np.clip(instability, 0, 100)
            + 0.5 * np.clip((gravy + 2.0) * 25, 0, 100)  # shift GRAVY ~[-2,2]  [0,100]
        )

        # True biosynthetic cost for annotation (not used as objective)
        true_cost = sum(AA_BIOSYNTHETIC_COST.get(aa, 20.0) for aa in sequence)
        expensive_count = sum(1 for aa in sequence if aa in EXPENSIVE_AA)

        result = {
            'proxy_cost':        float(proxy_cost),
            'instability_index': float(instability),
            'gravy':             float(gravy),
            'aliphatic_index':   float(aliphatic),
            'true_biosyn_cost':  float(true_cost),
            'expensive_aa_count': expensive_count,
        }
        self._cache[sequence] = result
        return result

    def score(self, sequence: str) -> float:
        """Return proxy cost scalar (interface-compatible with FBAEvaluator)."""
        return self.compute_proxy_cost(sequence)['proxy_cost']


def _aliphatic_index(sequence: str) -> float:
    """Ikai 1980 aliphatic index."""
    n  = len(sequence)
    xa = sequence.count('A') / n
    xv = sequence.count('V') / n
    xi = sequence.count('I') / n
    xl = sequence.count('L') / n
    return 100 * (xa + 2.9 * xv + 3.9 * (xi + xl))


print(' ProxyEvaluator defined.')
print(f'   Expensive AAs tracked: {", ".join(sorted(EXPENSIVE_AA))}')
print(f'   Most expensive: W ({AA_BIOSYNTHETIC_COST["W"]:.1f} ATP eq)')
print(f'   Cheapest: G/A ({AA_BIOSYNTHETIC_COST["G"]:.1f} ATP eq)')
