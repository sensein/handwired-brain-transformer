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

## 2026-11-24 regions, engines, evaluation
* regions r1 (r1.png): tissue = brightest histogram valley of the FOV quantile function; correct outlines on YO_90/YO_117/
  NN_86/NN_161 L5 and YO_90 L6; dim NN_161 L6 lobe called background; dim patches inside white matter -> holes.
* within-tissue rank f (r2.png) shows the laminar contrast: NN cortex bright / WM dark; YO bright outer, dark inner cortex,
  medium WM. Prototype cortex/WM (m1-m3.png): first version had no WM because stage pooling mixed out-of-slab planes ->
  `zpool=False` (in-plane merging) and core voxels (tissue across +-60 um in z); bug min(dO,dI) fixed; r4.png: WM cores on
  NN_161_L5_w0, YO_90_L5_w0, YO_117_L5_w2; NN_86_L5_w2 and L6 slabs: no WM (deep-tissue prototype missing).
* scale sweep 1 (sweep.png): 8 mm windows inside tissue gave no background (dark sulci < 4 % of FOV missed) -> tail rule
  for the valley test (dark-tail intervals with a log jump > 2x the bright-end spread). After: NN_161_B3 bg 3 %, NN_86_B3 7 %,
  all-tissue L0/L1 windows still no false background.
* engines: JS-CPU vs Python float64 worst 3.5e-4; WebGPU (SwiftShader) worst 4.6e-3 (steep ramp), no binary flips;
  seg.js/route.js integration path (hier_seg_test.mjs): 0 flips.
* robustness (robust.txt, quantised uint16 transforms of RAW intensity): cells / vessels IoU 1.00 (min >= 0.99) for gain
  x0.25/x4, gamma 0.5/2, log-contrast 0.5/2, offset +200; FLAT/DARK/SAT 0 detected. regions (pre tail-rule): tissue >= 0.99,
  cortex / wm mean 0.94 / 0.88, min 0.64 at gamma 2 and offset +200 (offset is monotone but not log-affine: the valley
  test is only log-affine invariant); FLAT on YO_90_L5_w0 still 25 % tissue (log contrast x0.02 of a 6-log-unit window
  leaves 10 % contrast > the 6 % flat gate).
* cell sweep (same 0.6 mm FOV, levels 0-4): objects/mm2 YO_90 524/237/53/0/0, NN_161 669/218/34/0/0: NOT level-consistent;
  finer cells are lost as the voxel grows and the prior switches them off at >= 20 um voxels.
