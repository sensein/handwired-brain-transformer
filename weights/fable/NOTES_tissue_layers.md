# Hand-written weights: fable_tissue / fable_cortex_wm / fable_layers

All weights are numbers chosen by reasoning and by looking at images (whole-slab class maps) and printed mask
statistics. No optimizer, no fitting loop, no labels. Generators (`make_tissue.py`, `make_cortex_wm.py`,
`make_layers.py`) only place the chosen numbers into sparse JSON using `fable_common.py`; they do not import
`hwseg.programs` or `hwseg.wiring`. `tl_look.py` only evaluates weight files (class maps, robustness tables).

## What the data looks like (slab bank, centre plane, default percentile display window)

* L6 whole slabs (146-164 um x 207-232 um in-plane): the stitched slab has ~5 % exact-zero padding (left/right bars),
  so the window is [0, hi] and the *background is not black*: air sits at u' ~ 0.30-0.45, tissue at 0.75-0.97, and a
  smooth haze (out-of-plane light around the tissue, u' 0.45-0.65) lies between. L5 windows (73-82 um) have no padding:
  air at 0.0-0.2, tissue 0.6-0.97.
* Inside tissue the cortex / white-matter contrast in the log display is small (a few 0.01 display units; strip
  banding from the stitched illumination is of similar size) and it is **stain specific**:
  NN: cortex slightly brighter than white matter, with bright laminar lines;
  YO: a thin bright pial rim, then a *darker* cortical band, then brighter white matter.
  The cortical ribbon is ~2.5-3.5 mm wide (30-40 voxels at L5, 12-20 at L6).
* NN_86_L5_w0 is a deep FOV: almost all white matter with a few holes / sulcal ends (the "no cortex" test FOV).

## Iteration log

### tissue v1 (t1) - 6 layers, d=52, 411 nonzero
Design: smooth u' (sigma 120 um); 16 display-level ramps and global heads give a soft CDF of the FOV; bright level H =
level exceeded by ~3.5 % of the FOV; dark set = voxels below 0.8 H; dark level L = 15 %-quantile of the dark set (so the
5 % zero padding does not become "the background level"); threshold T = L + 0.5 (H - L); view gates H >= 0.55 and
H - L >= 0.2 push T out of reach (nothing detected, bg also 0).
Seen (L6 full x4, L5 x12, default view): tissue outlines follow the slab on all NN and YO views, sulci and the ventricle
of YO_117 are bg, haze is bg. H ~ 0.94, L ~ 0.31 (L6) / 0.0-0.18 (L5), T 0.47-0.67. NN_161_L6: the dim left lobe
(u' ~ 0.6, below T = 0.67) is called bg; in the default view it looks like dim tissue. NN_86_L5_w0 (all tissue):
only the holes are bg (5 %), no hallucinated background.
robust (IoU vs default view; 16 FOVs): tissue mean 0.92-0.99 over c0.5/c2/c4/g0.5/g2/b+-0.15, min 0.56; bg min 0.26.
Failures: (i) FLAT c=0.05 gave tissue 100 %: with an empty dark set every level had cond_j = 0 and the half-ramp gave
L ~ 0.25, so the range gate passed. (ii) g=0.5 on the two L5 windows with zero padding (YO_90_L5_w2 0.56, NN_161_L5_w2
0.64): padding was > 15 % of the dark set so L = 0. (iii) NN_86_L5_w0 (all tissue) bg IoU 0.26 at g=2: the dim tail of a
unimodal FOV is split by T.

### tissue v2 (t2) - 7 layers, d=56, 440 nonzero
Changes: L ramp clamp(cond/WL + 1) (empty dark set -> L = FOV minimum); RHO 0.8 -> 0.75 and QD 0.15 -> 0.30 (padding up
to 30 % of the dark set tolerated); new bimodality switch: indicators at L + 0.25 R and L + 0.75 R, global heads give
dark mass DM and mid-band mass MB; T = L + 0.5 R when MB < DM, falling to L + 0.25 R otherwise (C_BI = 5).
Seen: same outlines as v1 on bimodal FOVs; NN_86_L5_w0 switches to the low threshold (T 0.51, bg 5 % = holes only);
YO_90_L5_w0 also takes the low threshold (T 0.33) and still shows its sulci as bg.

### cortex_wm v1 (c1) - tissue v2 + 1 layer (Gaussian proximity of bg, sigma 1.3 mm, isotropic, z +-700 um)
Seen: L6 cortex = 100 % of the FOV. Two errors: the tissue gate M = 50 was weaker than K (PB - THETA) so background
became "cortex"; and PB never fell below 0.12 inside tissue because the slab is only ~1 mm thick in z (6-8 planes at L6,
~12 at L5 hold tissue) - z neighbours are the cut faces of the slab, not pia.

### cortex_wm v2 (c2) - in-plane proximity head (sigma_z 10 um), M = 1.5 K
Seen (c2): an orange band follows the pial outline on every FOV and white matter fills the cores; nothing in bg. The band
is ~2 mm wide at straight edges (the visible ribbon is ~3 mm), thinner inside narrow sulci, and absent along *closed*
sulci (thin dark lines, YO_117_L5_w1/w2) so whole folded regions are called wm. Holes in white matter (NN_86_L5_w0,
the ventricle of YO_117_L6) get a cortex ring.
tissue v2 robust: mean IoU tissue 0.95-0.99, bg 0.88-0.98; DARK b-0.5, FLAT c0.1, FLAT c0.05, SAT b+1 all 0 % for both
outputs. Worst: NN_161_L5_w2 g0.5 0.60 (10 % padding + gray bg), NN_86_L5_w0 bg at g2 0.21 (bg is only 4 % there).
SAT b+0.6 still shows the silhouette (tissue 1.0 on gray 0.6-0.8) and is detected.

### cortex_wm v3 (c3) - d=61, 9 layers, 495 nonzero
Changes: SIG_D 1.3 -> 1.6 mm, THETA 0.03 -> 0.015 (straight edge 3.5 mm, 0.2 mm slit 2.5 mm); new dark-line detector
VAL = step(G_300um(u') - u' - 0.06 (H - L)) inside tissue, added to the proximity value with weight 3 (closed sulci);
output steps hardened (K 2000, tissue SHARP 400) for contour drawing.
Seen: ribbons now run along the closed sulci of YO_117_L5_w1/w2 and YO_90_L5_w0 and enclose white-matter lobes, which
matches the ribbons visible in the contrast-stretched images. Band ~3.3 mm at open edges, slightly thick. New false
cortex: blobs around short dark vessel segments in white matter (YO_117_L5_w0: 3 blobs; NN_86_L5_w0: 44 % cortex in a
FOV that is almost all white matter - ring around the hole plus lines along streaks).
L6 whole slabs (c3): ribbon around every slab and along closed sulci, white-matter cores inside; the lateral ventricle of
YO_117 and the central hole of NN_86 get a (wrong) cortex ring.

### cortex_wm v4 (c4) - test of an intensity cue (not kept)
Added: background filled to T + 0.35 R, local brightness DoG = G_300um - G_1500um of the filled image (channels DP/DN).
Seen (NN_161_L5_w0, NN_86_L5_w0, YO_117_L5_w2): DP is positive along NN ribbons (0.03-0.09 display units) but equally
strong on the bright rim around the hole in white matter (NN_86_L5_w0) and DN lines up with tissue edges (fill
mismatch) rather than with white matter; in YO the sign pattern follows edges and sulcal lines, not the ribbon. The cue
would move the boundary by edge artefacts, so it is not used. cortex vs wm stays geometric (stain-agnostic).

### cortex_wm v5 (c5)
Changes: DoG heads removed; W_VAL 3 -> 2 and THETA 0.015 -> 0.02 so that a dark segment must be > ~2.6 mm long to seed
cortex (straight edge 2.05 sigma = 3.3 mm; closed sulcus line ~1.9 mm each side).
Seen (c5, default): L6 and L5 maps like c3 with fewer blobs in white matter (YO_117_L5_w0 cortex 41 -> 31 %,
NN_86_L5_w0 44 -> 34 %). robust (9 FOVs): DARK / FLAT / SAT b+1 = 0 % for all classes. Mean IoU cortex 0.92 (c0.5)
0.80 (c2) 0.70 (c4) 0.86 (g0.5) 0.74 (g2) 0.89 / 0.94 (b+-0.15); wm 0.62 at g2 with two FOVs at 0.00.
Looked at g2 on YO_90_L5_w0 and NN_86_L5_w0 (tissue-dominated FOVs): L jumped to 0.43-0.47 (the 30 % quantile of the
dark set lands in dim tissue once gamma widens the dark set), R = H - L collapsed, the line detector threshold
DELTA * R went to ~0.01 and VAL fired on 4-6 % of the FOV -> cortex everywhere, wm 0 %.

### tissue v3 / cortex_wm v6 (c6) - d=57 / 62
Changes: new FOV-minimum level LMIN (1 % quantile from the same soft CDF). In the non-bimodal regime the threshold is
now LMIN + 0.25 (H - LMIN) instead of L + 0.25 (H - L); the line detector is scaled by H - LMIN.
Seen (c6): g2 on the two tissue-dominated FOVs now keeps wm (YO_90_L5_w0 27 %, NN_86_L5_w0 47 %). robust (9 FOVs):
mean IoU cortex 0.94 / 0.81 / 0.72 (c0.5 / c2 / c4), 0.84 / 0.77 (g0.5 / g2), 0.89 / 0.93 (b+-0.15); wm 0.96 / 0.86 /
0.78, 0.87 / 0.76, 0.92 / 0.95; bg >= 0.89 mean; DARK, FLAT, SAT b+1: 0 % everywhere. Minimum 0.43-0.47
(NN_161_L6 at c4, YO_117_L5_w2 wm at g2).
Looked at c4 on L6 (YO_117, NN_161): in the c=4 image the closed sulcal lines are saturated away and the haze lobes of
NN_161 turn white, so the maps lose the ribbons along closed sulci and gain haze "tissue" - the output follows what
the picture shows. Looked at an out-of-range stretched view (c2.5, b-0.75) at L5: on NN the orange band coincides with
the visibly brighter ribbon (NN_161_L5_w0/w1/w2, NN_86_L5_w2: width and course agree); on YO the stretched view breaks
(dark cortex falls under T, line detector fires on texture) - outside the required range, but it shows the line
detector is the fragile part.

### layers v1 (f1) - cortex_wm v6 + 2 heads + 1 layer, d=69, 10 layers
Design: two more in-plane proximity heads (sigma 0.5 mm with line weight 6, sigma 1.0 mm with line weight 3), threshold
0.03 -> bands sup (0-~1 mm from pia), mid (~1-1.9 mm), deep (rest of the cortex mask).
Seen (L5 x12, L6 x4): three concentric bands parallel to the pial outline and to closed sulci on NN and YO. Isolated
1-2 voxel line detections inside cortex make round yellow "sup" blobs (weight 6 too high).
### layers v2
Change: W_SUP 6 -> 4 (a line still reaches 1.04 mm; a 2-voxel speck falls from 0.07 to 0.05).

### FOV / level checks on cortex_wm v6 (fin)
* pseudo level 4 (NN_161_L5_w0 upsampled 2x, spacing halved; the bank has no level-4 entries): cortex 34.5 % / wm 21.0 %
  vs 35 % / 20 % at L5 - the same map, so the physical-unit windows behave across voxel sizes on open-edge geometry.
* zoomed crop of YO_117_L5_w2 (128 x 192 voxels, 13 x 16 mm, own display window): cortex 75 % / wm 22 %; the same crop
  upsampled 2x: 60 % / 37 % - the faint closed-sulcus line in the middle is lost after interpolation, so the ribbon
  along it disappears. The thin-line detector depends on the level (a line must be ~1 voxel wide and >= 6 % of the FOV
  range darker than its 300 um surround).
* all-white-matter crop of NN_86_L5_w0 (100 x 150 voxels, no background): cortex 0 %, wm 0 %, tissue 0 %. The
  percentile window of such a FOV is 1 log unit wide (hi = lo + 1), the picture is a dim gray field with bright
  vessels, H = 0.475 < H_MIN = 0.55, so the view is declared unreadable. No hallucinated cortex, but also no wm label.

### tissue v4 / cortex_wm v7 (final)
Change: the range gate uses H - LMIN instead of H - L (in an all-tissue FOV L is a dim-tissue level, not a background
level; the gate is meant to catch flat views only).

### tissue v5 / cortex_wm v8 / layers v3 (final files)
Change: third view gate, LMIN > 0.55 -> unreadable (washed-out view: b = +0.6 lifts even the zero padding to 0.6). In
the required range LMIN stays <= ~0.45 (c0.5 with b+0.15 gives 0.40).

## Final design (what each file computes)

Shared front end (7 layers, `make_tissue.tissue_layers`): smoothed display image S (sigma 120 um); a 16-level soft CDF of
the FOV from global heads; bright level H (exceeded by ~3.5 % of the FOV), FOV minimum LMIN (1 % quantile), dark level L
(30 % quantile of the voxels darker than 0.75 H); bimodality test from two more global heads (mass between the modes
vs mass of the dark mode); threshold T = L + 0.5 (H - L) if bimodal, else LMIN + 0.25 (H - LMIN); three view gates
(H >= 0.55, H - LMIN >= 0.2, LMIN <= 0.55) that switch every output off. tissue = step(S - T), bg = step(T - S).

cortex / wm (2 more layers): thin dark lines inside tissue (closed sulci) VAL = step(G_300um(u') - u' - 0.06 (H - LMIN));
in-plane proximity PB = G_1.6mm(bg + 2 VAL); cortex = tissue and PB > 0.02, wm = tissue and PB <= 0.02. In words: cortex
is tissue within ~3.3 mm of an open tissue edge (2.2 mm for a 0.2 mm wide sulcus, ~1.9 mm either side of a closed
sulcus line); white matter is tissue deeper than that. A FOV with no background and no sulcal line gets no cortex; a
FOV with no deep tissue gets no white matter. No intensity cue separates gray from white (tested in v4, rejected).

layers (1 more layer, experimental): sup / mid / deep = cortex cut at ~1 mm and ~1.9 mm from the pial surface.

Attention windows (half-extent; all heads except the 120 um smoother are in-plane, z window = 1 voxel with zero weight):
| head | sigma | half-extent | L6 (164 x 232 um) | L5 (82 x 116 um) | L4 (41 x 58 um) |
|---|---|---|---|---|---|
| S smoother (isotropic) | 120 um | 240 um | 2 x 2 x 2 vox | 3 x 3 x 3 | 6 x 5 x 6 |
| S2 local level | 300 um | 600 um | 4 x 3 | 8 x 6 | 15 x 11 |
| PB proximity | 1.6 mm | 3.84 mm | 24 x 17 | 47 x 34 | 94 x 67 |
| layers: PS / PM | 0.5 / 1.0 mm | 1.2 / 2.4 mm | 8 x 6 / 15 x 11 | 15 x 11 / 30 x 21 | 30 x 21 / 59 x 42 |
(x by y voxels each side.) 19 of the heads are global (FOV means), free of any window.

## Final files

| file (generator) | outputs | d | layers | heads (global) | ReLU units | nonzero params | largest window half-extent |
|---|---|---|---|---|---|---|---|
| `fable_tissue.json` (`make_tissue.py`) | tissue, bg | 57 | 7 | 20 (19) | 147 | 543 | 240 um |
| `fable_cortex_wm.json` (`make_cortex_wm.py`) - stain-agnostic, NN and YO | cortex, wm, bg | 62 | 9 | 22 (19) | 153 | 598 | 3.84 mm in-plane |
| `fable_layers.json` (`make_layers.py`) - experimental | sup, mid, deep, wm, bg | 69 | 10 | 24 (19) | 160 | 649 | 3.84 mm in-plane |

(The counts are printed by the generators; run `make_*.py` to reproduce the files.) Outputs are hard steps (ramp width
0.0025 display units for tissue/bg, 0.0005 in PB for cortex/wm), so a contour at 0.5 is the class boundary: tissue
outline = boundary of bg; gray/white boundary = boundary between cortex and wm. No per-stain variants were needed
because no intensity cue is used; that also means the gray/white boundary is a *distance* surface, not a measured one.

Intermediate weight files of every iteration: /orcd/scratch/bcs/002/satra/hwseg-tmp/fable/out/tl/{t1,t2,c1..c6,f1,f2}.
`figs_tissue/`: default-view class maps at L6 (4 whole slabs) and L5 (12 windows) for the three programs; cortex_wm at
c0.5 / c2 / c4 / g0.5 / g2 on L6; dark (b-0.5) and washed-out (b+0.6) views; zoomed crop; pseudo-L4; the repo harness
(`scripts/fable_eval.py`) montages for tissue at L5; robustness tables `robust_*.txt`.

## Final observations (centre plane +-1 plane, percentile display window)

Robustness, final files (IoU of each class mask vs the default view; 9 FOVs = 4 L6 slabs + 5 L5 windows):
| class | c0.5 | c2 | c4 | g0.5 | g2 | b+0.15 | b-0.15 | min |
|---|---|---|---|---|---|---|---|---|
| tissue | 0.98 | 0.95 | 0.93 | 0.97 | 0.95 | 0.97 | 0.98 | 0.70 |
| bg | 0.97 | 0.90 | 0.89 | 0.94 | 0.93 | 0.95 | 0.97 | 0.72 |
| cortex | 0.94 | 0.81 | 0.72 | 0.84 | 0.77 | 0.89 | 0.93 | 0.47 |
| wm | 0.96 | 0.86 | 0.78 | 0.87 | 0.76 | 0.92 | 0.95 | 0.43 |
layers on 3 FOVs: sup 0.66-0.92, mid 0.58-0.89, deep 0.56-0.88 (means), min 0.34.
DARK (b-0.5), FLAT (c0.1, c0.05), SAT (b+1, b+0.6): 0 % detected for every output including bg, on all FOVs tested.
Cells below 0.7 for cortex / wm: 12 of 126, at c2 / c4 / g2 on YO_117_L6, NN_86_L6, NN_161_L6 (3 settings),
YO_90_L5_w0, YO_117_L5_w2, NN_86_L5_w0. Four more L5 windows run on the final file: all >= 0.66 except NN_161_L5_w2 at
g0.5 (cortex 0.40, wm 0.47, bg 0.30).

What looks right: the tissue outline on all 16 FOVs; a ribbon of ~3 mm along every open pial edge and along the
closed sulci that show as thin dark lines; white-matter cores; NN ribbon position agrees with the visibly brighter
NeuN band in a contrast-stretched view (NN_161_L5_w0/w1/w2, NN_86_L5_w2); a deep FOV (NN_86_L5_w0) is 68 % wm.

What looks wrong / limits:
* The gray/white boundary is a fixed distance from the pial surface (in-plane). It does not follow real thickness
  changes, is thinner inside narrow sulci (~2.2 mm) and along closed sulci (~1.9 mm each side), and is invisible to the
  true boundary in the image. No intensity cue is used (the v4 test showed it follows edge artefacts).
* Any hole or tear in white matter larger than ~0.3 mm^2, the lateral ventricle (YO_117_L6) and cut faces of the block
  through white matter get a cortex ring; dark vessel segments longer than ~2.6 mm seed a cortex blob (YO_117_L5_w0,
  NN_86_L5_w0: 28 % cortex in a FOV that is mostly white matter).
* Closed sulci with no visible dark line (most at L6, some at L5) leave their cortex labelled wm. The line detector is
  level- and contrast-dependent; at c=4 the lines saturate and those ribbons vanish.
* Cortex whose pial surface lies outside the FOV is labelled wm (the model sees only the FOV).
* The slab is ~1 mm thick, so proximity is in-plane only; obliquely cut cortex appears thicker than 3.3 mm in-plane and
  its inner part is labelled wm.
* Haze around tissue at L6 (out-of-plane light, u' 0.45-0.65) is bg in the default view but becomes tissue at c >= 2 on
  NN_161_L6 (the dim left lobe there is ambiguous to the eye as well).
* An all-tissue, low-contrast FOV (crop of NN_86_L5_w0 with no background: H = 0.475) is declared unreadable: no
  hallucination, but also no tissue / wm label.
* Zero padding larger than ~30 % of the dark part of a FOV with gray air breaks the dark-level estimate (NN_161_L5_w2 at
  g0.5).
* Level 4 is not in the bank; only a 2x-upsampled L5 window was run (same map as L5 on open-edge geometry; thin
  sulcal lines are lost by the interpolation). Levels 0-1 were not run (the programs are meant for >= 40 um voxels).
* Layers: three distance bands, never compared with any label; isolated line specks make round "sup" blobs; thin
  bands are the least stable under contrast (IoU often < 0.7 at c2 / c4 / g2).
