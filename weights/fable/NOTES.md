# Hand-written weights: fable_blobs / fable_vessels

All weights are numbers chosen by reasoning and by looking at montages and printed statistics. No optimizer, no
fitting loop, no labels. The generator scripts (`make_blobs.py`, `make_vessels.py`, helper `fable_common.py`) only place
the chosen numbers into the sparse JSON matrices; they do not import `hwseg.programs` or `hwseg.wiring`.
`fable_check.py` prints diagnostics (it only evaluates a weight file; it does not change weights).

## Files

| file | what | d | layers | heads (global) | ReLU units | nonzero params | largest window half-extent |
|---|---|---|---|---|---|---|---|
| `fable_blobs.json` (`make_blobs.py`) | nuclei / somata / clusters, YO and NN | 51 | 7 | 23 (7) | 42 | 375 | 240 um |
| `fable_vessels.json` (`make_vessels.py`) | tubes, LEC | 95 | 9 | 72 (7) | 242 | 2374 | 285 um |
| `fable_exposure.json` (`make_exposure.py`) | display window lo/hi from raw log(I+1) | 33 | 4 | 11 (11) | 32 | 152 | global only |

`figs/` holds the final montages (default view at levels 0,2,4 and 1,3; c=2, c=0.5, gamma=2, dark; vessels run on YO/NN).
`run_final.sh` reproduces the final tables; `sb.sh` submits a job (the session itself runs inside a Slurm job, so `srun`
would only start a step). `reference_builder_cells.json` is the builder-made reference, untouched.

## Design rationale (final versions)

Shared ideas
* **Physical octave scales.** Blobs: Gaussians at 2.5, 5, 10, 20, 40, 80 um, band k = G_k - G_{k+1}. Vessels: transverse
  sigma 3, 6, 12, 24, 48 um, elongation 3, 12 directions (golden-angle hemisphere). With an octave pyramid the finest
  valid scale at level L is always the L-th one, so every level sees the same filter in voxel units.
* **Scale gate without knowing the voxel size.** A Gaussian head applied to the r^2 channels returns r^2 + (variance of
  the sampled kernel). Subtracting r^2 leaves 3 sigma^2 when the kernel is resolved, ~0 when sigma is sub-voxel, and a
  large negative number near FOV borders (truncated kernel). One ReLU unit turns "variance ratio < 0.8-0.9" into a
  threshold penalty, so sub-voxel scales and border zones switch off instead of producing speckle.
* **Thresholds relative to the FOV, with an absolute floor that knows about exposure.** Global heads give the FOV mean of
  the response magnitude; the threshold is max(kappa * that, floor) (in-place max: thr += relu(floor - thr)). The floor
  is small (0.005 blobs / 0.01 vessels, display units) in a normally exposed FOV and rises by relu(0.05 - MAD) when the
  mean absolute deviation of the display image is small (flat, dark or washed-out view). Pure contrast reduction
  therefore leaves the mask unchanged until the view is nearly flat, and then everything goes off together.
* **Three-way class argmax** from weighted per-scale evidence (pairwise max layer, then steep ramps).

Blobs only
* Local relative threshold: a Gaussian head of sigma 3*sigma_k over |band_k| adds 3x the local mean response. A compact
  blob adds ~(R/W)^3 to its own local mean, a straight edge ~(sigma/W), so bright rims of cracks/tissue borders fail
  the test while blobs pass. Cross-band coupling (band k-1 mean added to band k threshold) stops the next band firing
  in a halo around a small bright object.
* Classes: nucleus = band 0 (ball diameter <= 10 um), soma = band 1 (12-20 um), cluster = bands 2-4 (>= 25 um), with
  weights 1 : 2.3 : 5.3 chosen from the response of a uniform ball to octave DoG bands.

Vessels only
* Response = min(strength, 3 * anisotropy): strength = E_d - W - threshold (E_d oriented Gaussian, W = direction mean
  of the next coarser scale = wide isotropic surround); anisotropy = E_d - mean_d E - penalties, where the penalties
  are 0.3*relu(Ebar_s - W_s) (round things), 0.5*relu(Ebar_{s+1} - W_{s+1}) (shell of something round one octave up)
  and relu(W_s - Ebar_s) (voxel darker than its surround). The min is written in place: E_d <- E_d - relu(a - 3b).
* Threshold >= 0.4 * MAD of the display image, so it does not collapse when clipping removes the background noise.
* Dead non-negative channels are recycled (a unit subtracts relu(channel) while new values are written) to stay at
  d = 95 <= 96.
* Classes: capillary = sigma_t 3 (tube diameter <= 8 um), vessel = sigma_t 6 (10-16 um), large = sigma_t >= 12 (>= 20 um).

Exposure
* lo = mean - 3 N - 10.2 N2 (N mean negative deviation, N2 mean deviation below mean - 2.5 N; = mean - 2.05 sd for a
  Gaussian). hi by counting: 7 candidates mean + a*P (a = 1.5 ... 9), each with a steep indicator, a global head for
  the fraction above it and two ReLU units that move hi to the next candidate while more than 0.3% of the FOV is
  still brighter; finally hi >= lo + 1. A first version with hi = mean + 3 P + 18.6 P2 overshot by 1-3 log units on
  FOVs that are part black background (mean abs error 0.67), the counting version has mean abs error 0.18 (lo 0.16)
  over the 36 bank crops, with a median 0.9% of voxels below lo and 0.02% above hi.

## Iteration log

Common tools per iteration: `fable_check.py stats` (band statistics on real crops), `robust` (IoU of `seg` vs the
default view under contrast/gamma/brightness), `sizes` (clean synthetic balls/tubes of known diameter),
`scripts/fable_eval.py` montages + `--phantom` + `--progression`.

### blobs v1 (4 layers, d=37, 204 nonzero)
Design: 6 octave Gaussians (2.5..80 um), band k = G_k - G_{k+1}; threshold max(10 * FOVmean(relu(-band)), 0.035);
a 1-voxel box head over r^2 measured the voxel size to switch off sub-voxel bands; classes by "E_x > sum of others".
Seen:
* Montages (YO, NN, L0/2/4): only the very brightest nuclei marked at L0 (YO ~1%), NN L0 somata fine on bright cells;
  NN_40_L2 has many visible 1-2 voxel neurons and nothing was detected; NN_40_L4 marks the bright rim along a dark
  crack as "cluster" (edge of a dark hole, not a blob); NN_14_L2 marks the flanks of a bright bubble artifact.
* robust: c0.5 0.61, c2 0.41, c4 0.29, g0.5 0.41, g2 0.52, b+-0.15 0.76 mean IoU. Detected fraction grows with contrast
  (YO_123_L0 0.6% -> 2.6% -> 6.0% at c=1,2,4): when the dark background clips to 0 its noise disappears from the
  negative-part statistic, the threshold falls to the floor and clipped-noise speckle is detected.
* phantom: base 0.56 < c0.5 0.91 (same defect as the builder reference): coarser bands fire in a halo around bright
  small cells because their threshold sits at the floor in the sparse phantom.
* sizes: nucleus <=10 um, soma 12-16 um, cluster >=32 um, but 20-25 um balls were mostly unclassified
  (three-way "greater than the sum" rule leaves gaps). Scale gate worked (L2: bands 0,1 off; L4: only band 4 on).

### blobs v2 (6 layers, d=45, 303 nonzero)
Changes: statistic -> FOV mean |band| with kappa 4 and an additive floor 0.02; FOV "visibility gate" (off when the mean
absolute deviation of the display image < 0.03); image floored at FOVmean - 1.5*MAD before filtering (dark cracks);
the box-head scale gate replaced by Gaussian heads over r^2 (kernel variance collapses when sigma is sub-voxel and goes
negative near FOV borders, so one probe gates both); proper three-way argmax (pairwise max layer).
Seen:
* c=2 montage of NN_40_L0: every visible neuron is marked (orange = soma) and that looks right; the default view marks
  only about half of them. So the default threshold is too conservative rather than c=2 too liberal.
* NN_40_L4 crack rim still marked (1.3%, was 2.2%): flooring alone is not enough.
* robust: c0.5 fell to 0.38 (additive floor: halving contrast raises the effective threshold 24% and the tail is steep),
  b-0.15 0.91, c2 0.39.  Phantom: base 0.84, c0.5 0.96, c2 0.85, g0.5 0.94, g2 0.50.
* `sizes`: no ball below 40 um detected: the clean synthetic FOV (one object on a flat background) has MAD 0.016 and the
  FOV visibility gate switched everything off. A FOV-level flatness gate is wrong for sparse clean fields -> removed.
* The border gate removes bands 3 and 4 over most of a 246 um FOV at L0 (they need ~1.9 sigma of context per side).

### blobs v3 (7 layers, d=51, 372 nonzero)
Changes: FOV visibility gate removed; threshold = max(3.5 * FOVmean|band|, 0.035) (in-place max on a threshold
channel); dark-hole penalty heads (Gaussian of the "deficit below the floor" added to the threshold).
Seen:
* YO L0: blue (nucleus) marks sit on the visible bright dots, a few larger merged blobs are orange (soma). NN_40_L0:
  most visible somata orange. NN_40_L2: many of the 1-2 voxel neurons now marked (as "cluster", only bands >= 10 um
  are valid at L2). NN_40_L4: the crack rim got *worse* (3.5%): the 3D crop contains a lot of black outside the
  shown slice (FOV mean 0.35, MAD 0.26), so the floor mean - 1.5*MAD is below zero and neither the flooring nor the
  dark-hole penalty does anything. FOV-mean based flooring is the wrong tool -> removed in v4.
* robust at L0: c0.5 0.86-0.97, c2 0.49-0.67, gamma 0.6-0.77, brightness 0.77-0.99. At L2/L4 thresholds sit at the 0.035
  floor, so c0.5 loses nearly everything there (IoU 0.02-0.5).
* phantom: base 0.64, c2 0.30, g2 0.27: kappa 3.5 lets clipped/gamma-amplified noise through in band 0.
* sizes (clean balls): nucleus <= 10 um, soma 12-20 um, cluster >= 25 um; three-way argmax leaves no gaps.

### blobs v4 (7 layers, d=51, 375 nonzero)
Changes: flooring and dark-hole heads removed; floor made MAD-dependent (0.015 + relu(0.05 - MAD)); threshold =
max(kappa_g * FOVmean|band|, floor) + kappa_l * local mean |band| with the local mean taken by a Gaussian head of
sigma_{k+1} (kappa_g 2-3, kappa_l 2).
Seen: detection collapsed to 0.2% at L0 (YO), 0% at L4. A window of sigma_{k+1} is only about one correlation length of
the band response, so the "local mean |band|" at a blob is mostly the blob itself and the blob suppresses itself.
NN_40_L4 rim: gone. Lesson: the local statistic needs a window several times wider than the band (blob contributes
~(R/W)^3, a straight edge ~ (sigma/W)^1).

### blobs v5 (7 layers, d=51, 375 nonzero)
Changes: local-statistics heads widened to sigma = 3*sigma_k (half-extent 2 sigma); kappa_l 3, kappa_g 2 (band 0) /
0.5 (others); floor 0.005 + relu(0.05 - MAD).
Seen:
* NN_40_L4: crack rim gone (0.0%), NN_14_L2: the flanks of the bright bubble are gone, only its centre and scattered
  single dots remain. NN_40_L0: 8 brightest somata orange. NN_14_L0 (noise-only crop): one big green "cluster" patch
  and a few soma marks - coarse bands too permissive (their threshold is 3.5x the mean, band 0 has 5x).
* phantom cells: P 0.27 R 0.99 F1 0.43; 0.44% "soma" for 4 um-radius cells: band 1 fires in a halo around the bright
  small cells (band-1 response of a small bright ball is ~0.1 of its amplitude, band-1 noise is tiny in the phantom).
* sizes: nucleus <= 10 um, soma 12-16, 20 um half soma / half cluster, cluster >= 25 um.

### blobs v6 (7 layers, d=51)
Change: cross-band coupling: the global head of band k-1 also writes XI=1 times its mean |band| into the threshold of
band k (kappa_g of bands >= 1 set to 0), so a coarser band cannot be far more sensitive than the next finer one.
Seen: phantom cells base 0.81 / c0.5 0.83 / c2 0.83 / g0.5 0.72 / g2 0.38 / b+0.15 0.85 / b-0.15 0.75, flat 0 (halo
gone: 98% of positives within one voxel of a true cell). robust mean IoU: c0.5 0.86, b-0.15 0.80, b+0.15 0.71,
g2 0.53, g0.5 0.41, c2 0.38, c4 0.23. Montage YO_122_L0: only ~15 of ~60 visible nuclei marked, YO_123_L0 almost
none: too conservative in band 0 (threshold 2x global + 3x local mean).

### blobs v7 = final (7 layers, d=51, 375 nonzero; iterations 7 and 8 are its two variants A and B)
Two hand-picked variants looked at side by side: A (kappa_l 2.5, kappa_g0 1.5) and B (kappa_l 3, kappa_g0 1.0).
A: more marks (YO_122_L0 1.3%) but phantom base 0.66 < c0.5 0.69 < g0.5 0.88 and more background positives (12%).
B: YO_122_L0 0.83% (nucleus 0.43%), phantom base 0.76 / c0.5 0.79 / c2 0.75 / g0.5 0.89 / g2 0.33 / b+ 0.86 / b- 0.69.
Chose B (keeps kappa_l = 3, which is what removes crack rims). Final numbers are in the tables below.

### vessels v1 (5 layers, d=96, 1842 nonzero)
Design: 5 scales x 12 directions of elongated Gaussians (rho=3), tubeness = E_d - mean_d E, threshold
max(10 * mean positive deviation, 0.02), same box-head scale gate.
Seen: seg 5-11% of the volume on LEC, big "large" areas along the dark sulcus at L4 and along FOV borders, clean
synthetic *balls* detected (80%+) and 20-30% of the voxels around large objects: direction-mean subtraction alone
does not reject blobs once kernels are sampled on an anisotropic voxel grid, and off-object voxels whose kernels
poke into a bright object pass. Phantom vessel F1 0.09.

### vessels v2 (7 layers, d=94)
Changes: FOV visibility gate, flooring, r^2 variance probes for the scale/border gate, penalties on the direction
contrast C = Ebar_s - Ebar_{s+1}: blob penalty 0.3*relu(C), off-ridge penalty relu(-C); 10 directions, rho 2.5.
Seen: LEC montage almost empty (one mark on the L0 streak); clean synthetic tubes not detected at all (the MAD gate
switched the clean FOV off, as for blobs); phantom vessel F1 0.07.

### vessels v3 (8 layers, d=94)
Changes: gate removed, threshold max(7*m, 0.02) in place, penalties moved into the Ebar channels.
Seen: tubes 50-70% detected (17-45% for the worst orientation), balls 0-23%, but 16-80% of the voxels *around* large
objects and FOV borders fire in the coarse scales ("large" 2% of LEC L0); L2: the thin 1-voxel vessels that the eye
sees in LEC_97_L2 are not marked, green patches along the top border are. Phantom F1 0.05.

### vessels v4 variants (probe 1.5 sigma_t, gate ratio 0.9, MAD floor; kappa 10 / 14, beta 0.3 / 0.6)
Seen: phantom precision 0.03-0.07 at any kappa, 60-88% of positives on background; beta 0.6 removes the tubes too.
A phantom figure (truth vs marks, 5 z-planes) showed the in-plane vessel marked, cells unmarked, and many short
streak marks in pure noise. Reason found by reasoning about the kernels: sigma_t = 2 um is under-sampled in y
(3.6 um voxels), so directions differ in noise level and "max - mean over directions" has poor signal (0.19 of the
line amplitude for the worst of 10 directions, 0.57 aligned) while the blob penalty eats 0.1 more.

### vessels v5 variants (sigma_t 3,6,12,24,48; rho 3; 12 directions; channels recycled to stay at d=95)
Seen: still P 0.06-0.17 with recall 0.19-0.79: thresholding the anisotropy itself cannot separate tubes from noise.

### vessels v6 variants (9 layers): response = min(strength, 3 * anisotropy)
strength = E_d - W - threshold with W = direction mean of the next coarser scale (wide surround); anisotropy =
E_d - Ebar - penalties only has to be positive. Variants kappa 14 / 20, eta 0.3 / 0.5.
Seen: recall 0.97-1.0, clean tubes 98-100% in every orientation; kappa 20: phantom F1 base 0.24, c2 0.52; but clean
balls of 16-32 um fire in 36-65% of their voxels (the shell of a ball is locally anisotropic at a finer scale).
LEC montage (kappa 20): elongated blue marks on the faint streaks at L0 (plausible capillaries); at L2 bright blobby
patches are marked "large" and the thin 1-voxel vessels are not; L4 nothing on the pial vessels.

### vessels v7 (9 layers, d=95, 2369 nonzero)
Changes: cross-scale blob penalty 0.5*relu(Ebar_{s+1} - W_{s+1}) (a voxel that sits in something round one octave
up is not a tube), kappa 22; class weights compared for two hand-picked sets with the tube-diameter table:
(1,2,4): capillary to ~10 um, vessel 10-17 um; (1,3,7): capillary <= 8 um, vessel 10-16 um, large >= 20 um -> chosen.
Phantom vessels: base 0.32, c0.5 0.34, c2 0.64, g0.5 0.60, g2 0.12, b+0.15 0.57, b-0.15 0.30; balls 30-47% (was 60%+).


### vessels v8 = final (9 layers, d=95, 2374 nonzero)
Seen in v7: detected fraction grew with contrast on dark crops (LEC_97_L0 0.87% -> 2.7% -> 7.0% at c = 1, 2, 4): the
background clips to 0, the anisotropy statistic collapses and clipped-noise clusters pass. Change: threshold >=
0.4 * MAD of the display image (one extra weight in the in-place max unit). Seen: default montage unchanged (same
marks), phantom better everywhere (base 0.32 -> 0.41, c0.5 0.34 -> 0.54, c2 0.64 -> 0.70, g0.5 0.60 -> 0.73), but the
c=2 montage of LEC_97_L0 still shows blotchy marks on speckle clusters (3.6% of the slice); see limitations.

## Final metrics (all run through scripts/fable_eval.py or fable_check.py on Slurm)

Phantom F1 of `seg` (scripts/fable_eval.py --phantom; 3 seeds each)

| view | blobs, cells phantom | vessels, vessels phantom (15 cells + 3 vessels) | builder reference cells (given) |
|---|---|---|---|
| base | 0.76 | 0.41 | worse than its c0.5 |
| c0.5 | 0.79 | 0.54 | |
| c2 | 0.75 | 0.70 | |
| g0.5 | 0.89 | 0.73 | |
| g2 | 0.33 | 0.15 | |
| b+0.15 | 0.86 | 0.65 | |
| b-0.15 | 0.69 | 0.39 | |
| c0.1 (flat) | 0.00, nothing detected | 0.00, nothing detected | |

Blobs phantom detail (base): precision 0.62, recall 0.99, 90% of positives within one voxel of a true cell.
Vessels phantom detail (base): precision 0.26, recall 0.96; 64% of positives on vessels, 20% on the round cells, 17%
background. The vessels program run on the cells-only phantom overlaps cells with F1 0.13.

IoU of `seg` against the default view, mean over crops (levels 0, 2, 4)

| view | blobs (12 YO/NN crops) | vessels (6 LEC crops) |
|---|---|---|
| c0.5 | 0.86 | 0.82 |
| c2 | 0.39 | 0.53 |
| c4 | 0.24 | 0.22 |
| g0.5 | 0.43 | 0.47 |
| g2 | 0.55 | 0.37 |
| b+0.15 | 0.72 | 0.72 |
| b-0.15 | 0.81 | 0.73 |
| c0.1 | 0.00 (0.00% detected in all crops) | 0.00 (0.01% in one L4 crop, else 0) |

Progression, blobs, detected % at brightness 0 for contrast 0.1 / 0.25 / 0.5 / 1 / 2 / 4:
phantom 0 / 0.06 / 0.24 / 0.25 / 0.23 / 0.27; YO_122_L0 0 / 0.02 / 0.79 / 0.83 / 0.67 / 0.72;
YO_123_L0 0 / 0 / 0.40 / 0.43 / 0.94 / 1.38; NN_40_L0 0 / 0.01 / 1.21 / 1.23 / 2.32 / 2.77.
Brightness -0.5: 0 at contrast <= 0.25 everywhere, then growing with contrast (phantom 1.35% at c=2).
Brightness +0.5: <= 0.36% up to c=2.
Progression, vessels (grid measured for v7; v8 differs only by the MAD floor, its grid job hit the 15 min limit):
LEC_165_L0 0 / 0.48 / 1.50 / 1.57 / 2.04 / 3.96; LEC_97_L0 0 / 0.03 / 0.81 / 0.87 / 2.74 / 6.96 (v8 robust table gives
the same 2.74 and 6.96 at c=2, 4 for this crop).

Size classes on clean synthetic objects (level 0): balls 6-10 um nucleus, 12-16 um soma, 20 um half soma half
cluster, >= 25 um cluster. Tubes (4 orientations, >= 96% of tube voxels marked): 6 um capillary, 10 um 84% vessel,
16 um 85% vessel, 25 um 93% large. Clean balls in the vessels program (v7): 8 um 0%, 16 um 30%, 32 um 39%, 50 um 47% of
ball voxels marked.

## Behaviours a-d, as observed

a) Contrast / brightness / gamma. Holds for contrast reduction and brightness (IoU 0.7-0.9). Does not reach 0.7 for
   c >= 2 or gamma: clipping or gamma changes the blob-to-noise ratio of the display image itself and the
   detections in these noisy crops sit on a steep tail. Phantom F1 stays 0.69-0.89 for blobs except gamma 2 (0.33).
b) Darkness to brightness. Nothing at c <= 0.1 (all crops, both programs), nothing at b = -0.5 with c <= 0.25, almost
   nothing at b = +0.5; blobs base phantom F1 (0.76) is no longer worse than c0.5 (0.79) by more than noise and the
   detected fraction is flat from c = 0.5 to 4 on the phantom and on YO_122. On dark-background crops (YO_123, NN_40,
   LEC_97) the fraction still grows 2-3x (blobs) and up to 8x (vessels) from c = 1 to 4.
c) Levels 0-4. Sub-voxel scales switch off exactly (L2: bands 0-1 off, L4: only the 40-80 um band on). L0: nuclei and
   somata; L1: somata on NN (many marked); L2: single-voxel neurons marked as "cluster" (only bands >= 10 um are valid);
   L3/L4: almost nothing, and the crack rim that earlier versions marked at NN_40_L4 is gone.
d) Biology. Blob size classes follow apparent size (blurred YO nuclei of ~12 um are called soma). Vessels: clean tubes
   are marked in every orientation and classed by diameter; round cells are rejected only partly (see numbers above).

## Known limitations
* Blobs are conservative at the default view: in YO_122_L0 about a third of the visible nuclei are marked, in
  NN_40_L0 the 8-10 brightest somata. More appear at c = 2, which is why IoU at c >= 2 is low.
* Vessels on real LEC: marks on the brighter streaks at L0; the thin 1-voxel vessels that the eye follows at L1/L2
  and the pial vessels at L4 are mostly missed, and one bright round patch at LEC_97_L2 is marked "large". On the
  noise-only NN_14_L0 crop the vessels program marks one false "large" patch.
* Vessel footprint is wider than the vessel (about one transverse sigma on each side), which caps phantom precision.
* Coarse scales need context: a band/scale is switched off within about 1.9 sigma (blobs) / 2.1 sigma_t (vessels) of
  the FOV border, so the 40-80 um blob band and the 24/48 um vessel scales are off in most of a 246 um L0 crop.
* Cost: the vessels program has 60 oriented heads; the largest windows (sigma_long 144 um, half-extent 285 um) are only
  useful at coarse levels but are evaluated at every level. Not timed on the WebGPU engine. The JS engines were not run.
* The scale gate reads kernel variance from r^2 channels in float32; verified only through the Python FFT path.
* Exposure: lo is off by up to 1.1 log units on two crops with a long dark tail (NN_14_L3, L5).


# Round 2: cheap fine-scale programs (fable_blobs_v2.json, fable_vessels_v2.json)

Brief: level-0 FOV pass of ~500x300 voxels must fit the viewer buffer (demo/app.js: slab * H * W <= 0.6 * 128 MB / 4 / d,
slab = 2*halo_z + 1, halo = sum over layers of the largest window radius in voxels); more visible nuclei / somata;
vessels with higher precision. Both banks used: strip bank (.../fable/bank) and stitched bank (.../fable/slabbank),
levels 0 and 1. Generators: make_blobs_v2.py, make_vessels_v2.py; iteration script run_r2.sh (weights for each
iteration under /orcd/scratch/bcs/002/satra/hwseg-tmp/fable/out/r2_N/). `fable_check.py cost` prints the buffer numbers.

Design decisions that make it cheap: ONE layer with local attention (everything else is global heads or MLPs, which add
no halo); z window 5 um (blobs: 2 voxels at L0 -> 5-plane slab) or 1 voxel (vessels: 3-plane slab); in-plane windows
<= 30-36 um; d = 25 (blobs) / 37 (vessels) by recycling dead channels. Vessel filters are in-plane only (6 directions
x 2 scales), so through-plane vessels are not detected by design.

## Iteration log (round 2)

### r2-1 (blobs d=23, vessels d=35): v1 logic squeezed into one local layer
Blobs: sigma 3/6/12, threshold 4x / 3x FOV mean |band|. Slab montage (YO_117, YO_90, new stitched data with dense crisp
nuclei): only ~10% of the visible nuclei marked; in a dense field the mean |band| is the nuclei themselves, so a
response-relative threshold defeats itself. Phantom base 0.56 (halo of the wider band).
Vessels: 6 in-plane directions x sigma_t 3/6, same min(strength, anisotropy) rule, kappa 18. LEC slab montage: clean
elongated marks on vessel segments at L0 and L1, perhaps a third of the visible ones. Phantom P 0.23 R 0.54.

### r2-2: noise-relative threshold
Added a 1-voxel box head (flat kernel, minimum window) and HP = |u' - box|: the FOV mean of HP measures voxel-scale
roughness (noise) and is not raised by nuclei, which are several voxels wide. Threshold = C * mean HP (C = 0.75 = 4 sd of
band 0 for white noise at L0; vessels 0.7 / 0.5), floors as before.
Blobs: phantom base 0.94 / c0.5 0.94 / b-0.15 0.94 / g2 0.69 / c2 0.65 / g0.5 0.55 / b+0.15 0.72 (recall drops where the
saturated cells cannot grow with the noise). Slab montages: every nucleus / soma is covered, but at L0 the marks merge
into clumps (20-25% of the slice, labelled soma); NN noise-only crop 0.2%; at NN_86 L1 the nucleus band (sigma 3) was
not gated and speckles a dark texture.
Vessels: phantom P 0.57 R 0.50, background positives 1%; LEC slab: most visible segments marked at L0 and L1, wall of
the big vessel traced, marks fatter than the vessels. But c0.5 IoU 0.32-0.41 and detection grows 2-4x at c >= 2.

### r2-3 (blobs d=25)
Blobs: additive signal term (threshold = 0.6 * mean HP + 1.25 / 1.5 * FOV mean |band|) so only blob cores pass and
neighbours separate; gate ratio 0.85 -> 0.93. Slab YO: nuclei marked individually at L0 (8-12% of the slice), fewer at
L1 (3-7%). Phantom unchanged (0.94 base).

### r2-4
Blobs: kappa band 1 1.5 -> 1.2 (L1 recall), flat-view floor made steeper (0.05 - 2*MAD) so it only acts when MAD < 0.025.
Strip bank NN_40_L0/L1: all visible somata marked; noise crop NN_14_L0 0.1%. robust mean IoU strip / slab bank:
c0.5 0.85 / 0.93, c2 0.54 / 0.67, c4 0.40 / 0.57, g0.5 0.52 / 0.64, g2 0.58 / 0.73, b+0.15 0.95 / 0.97, b-0.15 0.75 / 0.85;
flat 0. YO_122_L0 detected fraction 4.76 / 4.90 / 4.81 / 4.93 % at c = 0.5 / 1 / 2 / 4 (plateau).
Vessels (r2-3): floor fix + ZETA 0.5: c0.5 IoU 0.54-0.57 (absolute 0.01 floor still dominates dim crops).
Vessels (r2-4): absolute floor 0.002, additive anisotropy term kappa 6, ZETA 0.3: c0.5 IoU 0.90-0.93, phantom P 0.84
R 0.40 F1 0.53 but c2 0.17 / g0.5 0.22 (too strict once the noise doubles).

### r2-5
Blobs: floor 0.025 -> 0.02, kappa band 0 1.25 -> 1.1: c0.5 IoU 0.92 / 0.95, phantom base 0.94, c2 0.73, g0.5 0.60, g2 0.67,
b+ 0.77, b- 0.93; classes: balls <= 8 um nucleus, 10 um mixed, >= 12 um soma.
Vessels: C 0.4 / 0.3, class weights 1 : 1.5: phantom precision fell to 0.27 (too permissive), tubes <= 8 um now capillary,
>= 12 um vessel. -> go back up with C (next).

### r2-6
Vessels: C 0.55 / 0.4, kappa 4: phantom P 0.46 R 0.51 (F1 0.46-0.53 in every view except gamma 2, 0.16). Slab LEC
montage: essentially every visible segment marked at L0 and L1, but the marks are much fatter than the vessels and merge
(15% of the slice in dense fields).
Blobs: looked at adjusted views of r2-5 on the stitched bank. c=2: sensible (visible somata marked, empty crop 0.7%).
Dark view (b=-0.5, c=0.5): the image is almost black, yet 13% of YO_90_L0_w0 was marked with fat marks: clipping
removes the noise, both relative terms vanish and the floor 0.02 is all that is left. Change: darkness floor
0.10 - 1.25 * FOV mean of u' (acts only when the FOV mean is below 0.08). After: b=-0.5 gives 0 up to c=0.5 on all four
strip crops; the dark montage shows marks only on the brightest cores of the one crop where nuclei are still visible.

### r2-7
Vessels: footprint trim. Third term in the min: response <= 3 * (box3(u') - E_d), i.e. the voxel itself must be at
least as bright as the oriented mean through it. "Outside" marks around clean tubes fell from 2-10% to < 1%; phantom
P 0.50 R 0.48; contrast growth milder (LEC_165_L0 3.9% -> 5.6% at c=4, was 6.6 -> 12.4).
Blobs: class weight 3.5 -> 2.8: balls <= 10 um nucleus, >= 12 um soma. Strip YO montage: most bright nuclei marked at
L0 and L1 (was about a third in v1); at YO_122_L1 a few marks sit on the bright rim of a crack (no local-statistics
head in the cheap version).

### r2-8 / r2-9 (vessels)
r2-8: same darkness floor as blobs. Slab LEC montage: slim marks that follow the segments (8% instead of 15% in the
densest crop); bright round cross-sections are still marked; in the dim LEC_56_L1_w0 crop small ring-shaped marks
appear on faint dots. The floor hurt dim LEC crops (slab IoU c2 0.31, b-0.15 0.47).
r2-9: darkness floor lowered to 0.06 - 1.25 * mean (acts below 0.048): slab IoU c2 0.36, g2 0.41, b-0.15 0.63.

### r2-10 (vessels, rejected)
Seen in the final check: run on NeuN crops (stitched bank, L0) the vessels program marks 4-10% of the slice, mostly on
somata (many are elongated or touch a neighbour). Tried blob penalties ETA 0.3 -> 0.55, ETA2 0.5 -> 0.8: somata marks
only fell to 2.5-7%, while phantom vessel recall halved (F1 0.49 -> 0.30) and clean 6-10 um tubes lost the capillary
class. Not adopted; the final file keeps the r2-9 numbers.

## Round 2 final (files fable_blobs_v2.json, fable_vessels_v2.json; run_r2final.sh reproduces the tables)

| file | d | layers | heads (global) | ReLU units | nonzero | local layers | largest window (z, y, x) um |
|---|---|---|---|---|---|---|---|
| fable_blobs_v2.json | 25 | 6 | 11 (5) | 29 | 180 | 1 | 5, 30, 30 |
| fable_vessels_v2.json | 37 | 8 | 21 (5) | 81 | 565 | 1 | 2.5, 36, 36 |

Buffer cost for a 500 x 300 voxel FOV pass (rule of demo/app.js: slab * H * W <= 0.6 * 128 MB / 4 / d, slab = 2*halo_z+1):

| program | level | halo z,y,x (voxels) | slab | tokens | tokens*d*4 B | incl. in-plane halo | viewer budget (tokens) |
|---|---|---|---|---|---|---|---|
| blobs_v2 | 0 | 2, 9, 12 | 5 | 750,000 | 72 MB | 79 MB | 805,306: fits |
| blobs_v2 | 1 | 1, 5, 6 | 3 | 450,000 | 43 MB | 45 MB | fits |
| vessels_v2 | 0 | 1, 10, 15 | 3 | 450,000 | 64 MB | 72 MB | 544,125: fits |
| vessels_v2 | 1 | 1, 5, 8 | 3 | 450,000 | 64 MB | 68 MB | fits |
(v1 for comparison: blobs d=51 with a 240 um window, vessels d=95 with 285 um: ~100-voxel halo at level 0.)

Phantom F1 of seg (scripts/fable_eval.py --phantom)

| view | blobs_v2 (cells) | blobs v1 | vessels_v2 (vessels) | vessels v1 |
|---|---|---|---|---|
| base | 0.94 | 0.76 | 0.49 | 0.41 |
| c0.5 | 0.94 | 0.79 | 0.49 | 0.54 |
| c2 | 0.73 | 0.75 | 0.40 | 0.70 |
| g0.5 | 0.60 | 0.89 | 0.45 | 0.73 |
| g2 | 0.67 | 0.33 | 0.28 | 0.15 |
| b+0.15 | 0.77 | 0.86 | 0.48 | 0.65 |
| b-0.15 | 0.93 | 0.69 | 0.49 | 0.39 |
| c0.1 | nothing | nothing | nothing | nothing |
blobs_v2 base: precision 0.93, recall 0.95. vessels_v2 base: precision 0.50 (v1 0.26), recall 0.48; 89% of positives
within a voxel of a vessel, 10% on the round cells, 1% background. Recall is limited because the 6 directions are
in-plane: phantom vessels that cross the 3-plane slab are not seen (clean tubes: 0% for two of four test orientations).
On the cells-only phantom the vessels program overlaps cells with F1 0.07 (v1 0.13).

IoU of seg vs the default view, mean over crops (levels 0 and 1)

| view | blobs_v2 strip bank (8) | blobs_v2 stitched bank (24)* | vessels_v2 strip (4) | vessels_v2 stitched (12) |
|---|---|---|---|---|
| c0.5 | 0.92 | 0.95 | 0.85 | 0.82 |
| c2 | 0.55 | 0.66 | 0.51 | 0.36 |
| c4 | 0.39 | 0.57 | 0.31 | 0.22 |
| g0.5 | 0.52 | 0.65 | 0.66 | 0.71 |
| g2 | 0.55 | 0.70 | 0.63 | 0.41 |
| b+0.15 | 0.95 | 0.97 | 0.99 | 0.99 |
| b-0.15 | 0.76 | 0.81 | 0.81 | 0.63 |
| c0.1 | 0 detected | 0 detected | 0 detected | 0 detected |
* the stitched-bank blobs column is from iteration r2-6, whose seg is identical to the final file (only the class
weight changed afterwards); the final-file job printed 14 of 24 crops before the 15 min limit and those rows agree
(YO crops: c0.5 0.91-0.99, c2 0.53-0.83).

Detected fraction at brightness 0, contrast 0.1 / 0.25 / 0.5 / 1 / 2 / 4 (blobs_v2, final file):
phantom 0 / 0.15 / 0.15 / 0.15 / 0.09 / 0.07; YO_122_L0 0 / 2.10 / 5.02 / 5.18 / 5.13 / 5.24; YO_123_L0 0 / 1.01 / 3.09 / 3.25 /
5.76 / 7.10; NN_40_L0 0 / 3.59 / 6.99 / 7.14 / 9.06 / 7.22. Brightness -0.5: 0 up to c = 0.5 in all four.
Brightness +0.5 is not "washed out" for these crops (somata stay visible) and they stay marked (NN_40_L0 7.5%).
vessels_v2 (same constants, measured at r2-9): LEC_165_L0 0 / 2.40 / 3.30 / 3.89 / 5.08 / 5.57; LEC_97_L0 0 / 2.36 / 2.88 /
3.13 / 5.18 / 1.57; brightness -0.5: 0 up to c = 0.5.

Classes: balls <= 10 um nucleus, >= 12 um soma (blobs_v2 has no cluster output). Tubes: 4 um 26% capillary / 29% vessel,
6-8 um mostly vessel (55-64%), >= 10 um vessel; clean balls 8 um 0%, 12 um 20%, 16 um 24%, 24 um 46% marked.

What the montages show (figs/r2_*)
* YO, both banks: most bright nuclei carry their own mark at L0 and L1 (v1: about a third); detected fraction 3-5% on
  the strip crops, 5-15% on the dense stitched crops. NN: nearly every visible soma marked at L0 and L1; the noise-only
  crops stay at 0.2-0.5%.
* LEC stitched bank: slim marks along most visible vessel segments at L0 and L1, the wall of the large vessel is traced.
* Dark view (b -0.5, c 0.5): nothing, except thin cores in the one crop where gray nuclei are still visible.

Limitations (round 2)
* Levels 0 and 1 only: at level 2 and coarser every scale is gated off and both programs return nothing; the v1 files
  remain the coarse-level option. No cluster / large classes.
* blobs_v2 has no local-statistics head (it would need a second local layer): at YO_122_L1 a few marks sit on the bright
  rim of a crack. Marks at L1 are only 1-3 voxels wide.
* Contrast >= 2 and gamma still change the mask (IoU 0.4-0.7); a crop that is mostly noise can gain or lose nearly
  all of its few marks (NN_14_L1, NN_86_L1_w0).
* vessels_v2 does NOT reject real somata: on NeuN crops it marks 4-10% of the slice. It must only be routed to LEC.
  Bright round vessel cross-sections in LEC are marked too.
* vessels_v2 is in-plane: vessels crossing the slab are missed; capillary class is weak (most real marks are "vessel").
* Dim LEC crops (LEC_56): at c = 2 or brightness -0.15 the clipped view falls under the darkness floor and nearly all
  marks disappear (IoU 0.0-0.2 for those crops).
* Verified only through the Python FFT path; the JS / WebGPU engines were not run on the v2 files. The 1-voxel box head
  relies on the minimum-window rule (radius >= 1 voxel), which both engines implement.
