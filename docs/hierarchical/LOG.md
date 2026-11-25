# Hierarchical hand-programmed transformer: progress log

Scratch outputs: /orcd/scratch/bcs/002/satra/hwseg-tmp/hier/ (PNGs named below). All numbers chosen by reasoning and by
looking at images / printed statistics; no optimizer, no labels.

## 2026-11-24 phase 1 start
* Sibling worktrees (`hier`, `agent-ad3f...`, `hbt-neurodesk`) hold no extra work (fresh checkouts of 992ebb4): nothing reused.
* New format `hier1` (src/hwseg/hier.py), torch executor (hier_model.py), programs (hier_programs.py).
  Ops: rank (global step-kernel attention -> CDF), gpool (global softmax head -> FOV scalar), mlp (stage or global token),
  attn (Gaussian attention between stages: pool / upsample / smooth, physical units, separable exact evaluation),
  conv (scale-normalised Gaussian-derivative depthwise conv, replicate padding).
* cells v1 (c1.png): per-voxel blob measure relu(-max_d d^T H d), 13 directions, 8 scales on a 4-stage pyramid, single
  FOV threshold kappa * mean(score): YO/NN L0-L1 detect blobs but NN_86_L0 (noise-dominated NeuN view) gives 3 % speckle:
  the rank transform spreads background noise over [0,1].
* cells v2 (c2.png): per-scale calibration R_s > 1.0 * FOV-mean|trace_s|: almost nothing detected.
  hier_diag.py statistics: p99(R)/mean|tr| = 0.4-0.7 at stage 0, 0.15-0.35 at stages 1-3; noise (NN_86_L0) p999 0.057 at
  s0 vs 0.07 on YO nuclei. -> threshold_s = max(kappa_s * mean|tr_s|, floor_s), floor 0.03 at s0 (above NN_86 noise
  p999 at sigma 1.6 vox), 0.012 above.
* cells v3/v4 (c3.png, c4.png): kappa_s = (0.4, 0.4): NN_161_L0 somata mostly found (orange = soma, blue = cluster),
  NN_86_L0 0.9 % sparse specks, YO_90_L0 under-detected (dense touching nuclei: R small relative to mean|tr|), L2 views
  (10 um voxels) almost nothing (cells are 1 voxel: below the nucleus prior). Coarse ("cluster") responses win around
  groups of small cells at L1.
* vessels v1-v4 (v1..v4.png): tube = -l_perp - a|c_max| - b(l_perp - c_min). a=3-4 too strict (thick LEC tubes missed);
  a=1.5 fires on nuclei clusters; final a=2, b=1, floors (0.045, 0.015), kappa (0.55, 0.5): LEC tubes found as
  connected pieces, noise specks mostly gone; still fires on YO nuclear clusters (vessels program is meant for LEC).
* front-end gate: LEC_56_L1 has Qhi - Qlo = 0.21 log units (top/bottom 2.5 % ratio 1.24) yet visible vessels -> flat gate
  lowered from 0.25 to 0.06.
