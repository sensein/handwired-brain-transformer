"""Hand-authored weights for fable_exposure.json: a default display window (lo, hi) from FOV statistics.

Input ch0 = raw log(I+1) of the FOV (NOT the display image). Outputs "lo" and "hi" are constant over the FOV.
Only global (uniform-attention) heads and ReLU units; every statistic is a FOV mean.
  L1  head: mu = FOV mean (also seeds lo and hi)            | MLP: dp = relu(v - mu), dn = relu(mu - v)
  L2  heads: P = mean dp -> hi += A_HI[0]*P ; N = mean dn -> lo -= C_N*N
      | MLP: lower-tail deviation dn2 = relu(mu - PIV*N - v);
             indicators I_j = clamp(SHARP*(v - mu - A_HI[j]*P), 0, 1)  ("voxel is brighter than candidate j")
  L3  heads: lo -= C_N2 * mean dn2 ; F_j = mean I_j (fraction of the FOV above candidate j)
      | MLP: thermometer: hi += min(relu(GAIN*(F_j - Q)), (A_HI[j+1] - A_HI[j]) * P)   for each candidate j
             (if more than Q of the FOV is still brighter than candidate j, move hi up to candidate j+1)
  L4  MLP: hi <- max(hi, lo + MIN_WIDTH)
lo: Gaussian case P = N = 0.399 sd, deviation beyond 1 sd = 0.0833 sd -> lo = mu - 3.0*0.399 sd - 10.2*0.0833 sd = mu - 2.05 sd
    (2nd percentile); the tail term stretches lo when there is a dark tail.
hi: the 99.7th percentile is found by counting, because a "mean + k * deviation" rule cannot serve both a unimodal FOV
    (99.7th percentile at mu + 6.9 P) and a FOV that is part black background, part tissue (tissue top at mu + 2..4 P).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fable_common import global_head, mlp, step, write  # noqa: E402

C_N, C_N2 = 3.0, 10.2
PIV = 2.5            # tail pivot = mean - PIV * (mean one-sided deviation)  (= 1 sd for a Gaussian)
A_HI = [1.5, 2.5, 3.5, 4.5, 5.5, 7.0, 9.0, 11.5]   # hi candidates mu + a*P; a Gaussian needs 6.9
Q = 0.003            # tolerated fraction above hi
GAIN = 2000.0        # thermometer steepness: an excess fraction of 0.0005 moves hi by one log unit (capped by the step)
SHARP = 50.0         # indicator ramp width 0.02 log units
MIN_WIDTH = 1.0      # same minimum window width as hwseg.exposure.window

for _a in filter(None, os.environ.get("FABLE_OVR", "").split(";")):
    exec(_a)
NC = len(A_HI) - 1
MU, LO, HI, DP, DN, P, N, DN2 = range(11, 19)
IND = [19 + j for j in range(NC)]
F = [19 + NC + j for j in range(NC)]
D = 19 + 2 * NC


def build(path):
    L1 = {"heads": [global_head(D, {0: 1.0}, {MU: 1.0, LO: 1.0, HI: 1.0})],
          "mlp": mlp(D, [({0: 1.0, MU: -1.0}, 0.0, {DP: 1.0}), ({0: -1.0, MU: 1.0}, 0.0, {DN: 1.0})])}
    units2 = [({0: -1.0, MU: 1.0, N: -PIV}, 0.0, {DN2: 1.0})]
    for j in range(NC):
        units2 += step({0: 1.0, MU: -1.0, P: -A_HI[j]}, 0.0, IND[j], SHARP)
    L2 = {"heads": [global_head(D, {DP: 1.0}, {P: 1.0, HI: A_HI[0]}), global_head(D, {DN: 1.0}, {N: 1.0, LO: -C_N})],
          "mlp": mlp(D, units2)}
    heads3 = [global_head(D, {DN2: 1.0}, {LO: -C_N2})] + [global_head(D, {IND[j]: 1.0}, {F[j]: 1.0}) for j in range(NC)]
    units3 = []
    for j in range(NC):
        units3.append(({F[j]: GAIN}, -GAIN * Q, {HI: 1.0}))
        units3.append(({F[j]: GAIN, P: -(A_HI[j + 1] - A_HI[j])}, -GAIN * Q, {HI: -1.0}))
    L3 = {"heads": heads3, "mlp": mlp(D, units3)}
    L4 = {"heads": [], "mlp": mlp(D, [({LO: 1.0, HI: -1.0}, MIN_WIDTH, {HI: 1.0})])}
    meta = dict(kind="exposure", author="fable (hand-written)", input="raw log(I+1)")
    write(path, "fable_exposure", D, {"lo": LO, "hi": HI}, meta, [L1, L2, L3, L4])


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "fable_exposure.json"))
