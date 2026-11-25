# Hierarchical hand-programmed 3D transformer (`hier1`)

Status: implemented (Python reference `src/hwseg/hier*.py`, JS CPU + WebGPU `demo/hier.js`), evaluated without labels
(see LOG.md). No gradient descent, no optimizer, no labels: every number is placed by a generator (`hier_programs.py`,
`hier_regions.py`) and chosen by reasoning and by looking at outputs.

## 1. What changed relative to the single-resolution programs

| critique | old programs | `hier1` |
|---|---|---|
| single resolution | every head on the voxel grid | 4-5 stages, stride-2 overlapped patch merging; coarse stages carry mm-scale context cheaply |
| preprocessing outside the model | display window lo/hi + gamma, exposure program, level routing, `_x2/_x4/_x8` variants chosen in route.js | input is raw `log(I+1)`; contrast handled by an in-model rank front end; one weight file for every pyramid level; the voxel size is an input token |
| tissue / cortex as threshold pipelines | fixed blur, quantile ramps, a 3.3 mm distance rule | the FOV's own intensity distribution (global heads) sets the tissue threshold; cortex vs white matter by in-context prototypes (attention pooling over geometrically defined regions) and nearest-prototype assignment; depth bands from a potential between two attention-smoothed boundaries |

The level that the viewer fetches is still chosen from the zoom, but only as a data-cost decision (the coarsest level
whose voxel is not larger than a screen pixel); the same program runs at every level.

## 2. Tokens, stages, streams

* Stage 0 = the input voxel grid, shape `n0 = (Z, Y, X)`, spacing `sp0` (um, an input). Stage `s` has
  `ceil(n_{s-1}/2)` tokens per axis (z is kept at full resolution when a program sets `zpool = false`: the regions
  program, because a slab is only ~1 mm thick and merging across its cut faces would mix tissue with outside).
* Token `k` of stage `s` sits at the physical block centre `p_s(k) = (k 2^s + (2^s - 1)/2 - (n0 - 1)/2) sp0`
  (tile-centred um). Every cross-stage op is defined from these positions only.
* Each stage has a residual stream of `d[s]` channels; one global token carries `dg` FOV-level scalars that every MLP
  can read. Fixed channels: stage 0 `0 = log(I+1)`, `1 = 1`; stage s>0 `0 = 1`; global `0 = log2(sp0_x / um)`, `1 = 1`.
* Every op adds `gain * result` into destination channels (residual stream).

## 3. Ops (identical semantics in torch, JS-CPU and WGSL)

| op | definition | transformer reading |
|---|---|---|
| `rank` | `u_i = (1/K) sum_j H(v_i - v_j)`, `H(0) = 1/2`, keys `j` = every `m`-th token (flat index, `m = ceil(N/kmax)`, `kmax = 4096`), optionally only keys with a mask > 0.5 | global attention with a step kernel: sigmoid attention `sigma(beta q_i k_j)` with `q_i = [v_i, 1]`, `k_j = [1, -v_j]` in the limit `beta -> inf`, uniform normalisation, value 1; spatial-reduction keys |
| `gpool` | `g_c += gain * sum_j softmax_j(beta x[score]_j) x[src]_j` over all tokens of a stage | one global softmax head whose query is constant (content-only scores); beta = 0 gives the FOV mean, beta > 0 attends to the top of a score (e.g. the brightest ranks), a mask score restricts it to a region |
| `mlp` | `x += W2 relu(W1 [x; g] + b1)` per token of a stage, or on the global token | the MLP block |
| `attn` | `o_i = sum_j softmax_j(-1/2 sum_a ((p_j,a - p_i,a)/sigma_a)^2) v_j` over `|p_j,a - p_i,a| <= max(3 sigma_a, h_a)` (`h_a` = source voxel), source stage -> destination stage, sigma in um or source voxels | Gaussian-score dot-product attention (`q_i = [P p_i; 1]`, `k_j = [p_j; -1/2 p_j^T P p_j]`); dst = src+1 with sigma = 1 voxel is overlapped patch merging (strided windowed attention pooling); dst < src is the decode-head upsampling; same stage with a large sigma at a coarse stage is spatial-reduction attention |
| `conv` | depthwise separable Gaussian-derivative correlation, scale-normalised (`sigma^|o| d^o G_sigma`, orders <= 2 per axis), replicate padding, sigma isotropic in um (or in stage voxels) | the Mix-FFN 3x3x3 depthwise conv with hand-set kernels (Hessian filter bank) |

Implementation notes. A diagonal-precision Gaussian over a box window factorises exactly, so `attn` and `conv` are
evaluated as three banded 1-D passes with host-built matrices (float64 on the host in Python and JS). The 1-D derivative
kernels are normalised so that `sum k0 = 1`, `sum t k1 = sigma`, `sum k2 = 0`, `sum t^2/2 k2 = sigma^2`; sigma is
clamped to >= half a voxel (a sub-voxel filter is a delta / smallest difference). `rank` in WGSL is a brute-force loop
over the strided keys (O(N K)); Python / JS-CPU sort the keys (same result).

## 4. Normalisation front end (all programs)

* `u = rank(logI)`: each voxel becomes its CDF value in the field of view.
* Global softmax heads read the view: `Qhi`/`Qlo` = mean log-intensity of the brightest / darkest ~1/40 of the ranks
  (`gpool`, score `+-40 u`), `Uhi`/`Ulo` = mean rank there.
* Unreadable views switch every output off (`ok = 0`): flat (`Qhi - Qlo < 0.06` log units, i.e. the brightest and darkest
  2.5 % differ by < 6 %), dark (`Qhi < 3.4`, the brightest voxels below ~30 counts, or more than ~40 % of the FOV tied at
  its minimum), saturated (`Uhi < 0.9`: more than ~20 % of the FOV tied at its maximum).

## 5. Programs

### 5.1 `hier_cells` (4 stages, d = [41, 37, 37, 37], 282 ops, 4761 nonzero parameters)
1. Pyramid of the rank image: `u_s = attn(u_{s-1} -> stage s, sigma = 1 source voxel)`.
2. At (stage, sigma) = (0, 1.3), (0, 1.8), (s, 1.0), (s, 1.5) for s = 1..3 (sigma in stage voxels): 6 Hessian channels
   (`conv`), 13 directional curvatures `c_d = d^T H d` (3 axes, 6 face and 4 body diagonals, physical directions), their
   max / min by ReLU tournaments; blob measure `B = relu(-c_max)` (all curvatures negative).
3. In-context per-scale calibration: `B <- relu(B - max(k_s mean_FOV|tr H|, floor_s))` with `k = 0.4`, floors 0.03
   (stage 0) / 0.012 (coarser) chosen from printed statistics (NN noise p99.9 at stage 0 is 0.038-0.057).
4. Physical size prior computed in-model from the voxel-size token: the effective scale of each response is
   `sigma_eff = sp0 sqrt((c 2^s)^2 + (4^s - 1)/3)`; a trapezoid of `log2(sigma_eff)` gates it into classes nucleus
   (1.6-3.6 um), soma (3.6-8 um), cluster (8-18 um). Responses outside every class are dropped (e.g. 10 um-voxel views
   cannot resolve a nucleus, so stage 0 at level 2 is a 'soma' or 'cluster' scale, and level >= 3 outputs nothing).
5. Decode head: gated responses are upsampled to stage 0 by Gaussian attention (sigma 0.6 source voxel) and fused by a
   per-class maximum (scale selection by response); `seg = step(max_k C_k - tau)` and `ok`.

### 5.2 `hier_vessels` (4 stages, d = [44, 40, 40, 40], 306 ops, 4949 nonzero)
Same skeleton with the tube measure `T = -l_perp - 2|c_max| - (l_perp - c_min)`, `l_perp = (tr - c_max)/2` (two
transverse curvatures strong and equal, axial curvature ~0; zero for an ideal blob or sheet); calibration `k = 0.55 / 0.5`,
floors 0.045 / 0.015; classes capillary (1.2-3 um), vessel (3-7 um), large (7-20 um).

### 5.3 `hier_regions` (5 stages, in-plane merging, d = [53, 3, 7, 1, 1], 155 ops, 3048 nonzero)
1. Tissue. 49 global heads give the quantile function `Q(p_k)`, `p_k = 0.02..0.98` (score `-300 |u - p_k|`, absent
   data `I = 0` masked). Inverse density `D_k = (Q_{k+1} - Q_k)/0.02`. A histogram valley is an interval sparser than
   both neighbourhoods and the FOV average: `W_k = min(D_k - 1.8 (D_{k-2} + D_{k+2})/2, D_k - 2.5 D_ref)`. The tissue
   threshold is the brightest valley (relu-tournament argmax with index transport `s x = relu(x - M(1 - s))`), `T = p_k +
   0.01` (a rank); no valley -> all tissue. `tissue = step(G_25um(u) - T)`.
2. Within-tissue feature `f`: rank (keys restricted to tissue) of the 100 um-smoothed rank image, smoothed in-plane at
   200 um. It shows the laminar structure: NN cortex bright / white matter dark; YO bright outer cortex, dark deep cortex,
   medium-bright white matter.
3. Context at stage 2 (two in-plane patch merges): proximity of background (the pial side) `P_a, P_b, P_p` = in-plane
   Gaussian attention, sigma 0.5 / 1.5 / 1.0 mm, upsampled to stage 0.
4. Prototype regions, declared from the exact response of a straight edge (`P_sigma(d) = Phi(-d/sigma)`): outer cortex
   0.2-0.8 mm below the pia, inner cortex 1.2-2.0 mm, deep tissue > 3.5 mm; restricted to core voxels (tissue across
   +-60 um in z, so the slab's cut faces do not count as pia). Prototype values = attention pooling of `f` over each
   region (white matter: softmax over `-1000 P_b`, i.e. the deepest tissue).
5. Nearest prototype: `wm = step(min(|f - m_O|, |f - m_I|) - |f - m_W| - 0.05)` inside tissue, never within ~0.2 mm of the
   pia, only when both prototypes exist (> 0.3 % of the FOV); majority vote over 150 um (Gaussian attention, > 0.5).
   `cortex = tissue - wm`, only when a pial surface is in view.
6. Depth bands: potential `x = P_wm - P_pia` (both in-plane Gaussian attention, sigma 1 mm, white-matter proximity computed
   at stage 2 from the wm map); band edges are the values of `x` at the declared depth fractions 0.08 / 0.15 / 0.45 /
   0.55 / 0.75 for a straight 2.5 mm cortex (a Laplace-like equipotential). Layers I..VI = bands of `x` inside the cortex.

## 6. Guaranteed by construction vs measured

Guaranteed (exact, independent of data):
* `u` is invariant to every strictly increasing intensity map (tested: gain, power gamma, offset, log-cubed;
  `tests/test_hier.py`). Hence every output that depends on the image only through `u` (cells, vessels, the within-tissue
  feature `f`) is invariant to such maps as long as the gates stay open. Quantisation (uint16 rounding) and clipping are
  not monotone-invertible and are covered only by measurement.
* The tissue valley test and threshold use ratios of log-intensity differences: invariant to intensity gain and power-law
  gamma (log-affine maps), not to general monotone maps (up to the `+1` in `log(I+1)`).
* Physical units: every attention / conv extent is in um or derived from the voxel size, and the size prior reads the
  voxel size token, so the same weight file runs at any spacing. This does NOT imply identical outputs across levels
  (sampling, partial-volume and the pyramid's own blur differ); the scale sweep measures it.
* Separable evaluation equals the dense masked softmax (test); Python float64, JS-CPU float32 and WebGPU float32 agree
  (`demo/test/golden_hier.json`; worst continuous difference 3.5e-4 JS-CPU, 4.6e-3 WebGPU on a steep ramp output).

Measured only (LOG.md, `scripts/hier_eval.py`): detection quality on real windows, robustness to quantised transforms,
consistency across pyramid levels, behaviour on dark / flat / saturated views.

## 7. Known limitations (honest)
* Cells: dense touching YO nuclei are under-detected; at 5 um voxels the coarse 'cluster' class wins around groups of small
  cells; masks are blob cores, not full cell outlines; nothing is detected at >= 10-20 um voxels (by design of the prior).
* Vessels: local Hessian tube measure; thick LEC tubes are found in pieces, the class is mostly 'large' because the
  response peaks at coarse stages; it fires on clusters of YO nuclei.
* Tissue: the brightest-valley rule calls dim tissue lobes (NN_161 L6) background; dark sulci occupying < ~4 % of the FOV
  are not found (the valley test needs neighbours on both sides).
* Cortex / white matter: needs a pial surface and deep tissue in view; YO's three-level profile makes the nearest-prototype
  rule fragile; buried cortex in closed sulci is labelled by intensity only.
* Depth bands are declared fractions on a potential, not detected laminae.
* WebGPU `rank` is O(N K); a sort-based version is future work.
