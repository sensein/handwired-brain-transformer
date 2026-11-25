# Hand-written programs vs Neurodesk reference models (label-free)

DANDI 000108 ships no labels, so **nothing here is accuracy**. The reference models (Cellpose 3 `nuclei`, StarDist
`2D_versatile_fluo`, cellfinder candidate detection, VesselBoost) are established tools trained or tuned on *other* data
(cultured cells and DSB nuclei, mouse whole-brain light-sheet, human MR angiography); they are a second opinion, not ground
truth. The numbers measure how much the methods agree with each other on the same real windows, next to how much the
references agree among themselves and how much agreement chance alone gives.

## Investigation: what the Neurodesk apps provide

| candidate | Neurodesk app | weights host (browser-fetchable?) | size | licence | input spec | post-processing | 2-D / 3-D | used? |
|---|---|---|---|---|---|---|---|---|
| Cellpose 3 `nuclei` | `apps/cellpose` | HF dataset `neurodeskorg/webapps`, revision-pinned; CORS `*` (checked with an `Origin` header) | 26.5 MB | BSD-3 code and weights (HHMI); training data include CC-BY-NC sets | 2 channels (nuclear + zeros), `normalize99` 1-99th pct, rescale to diam_mean 17 px, 224-px tiles, 10% overlap, pad to /16 | flow dynamics 200 steps, mask clustering, flow-error QC 0.4, fill holes, min 15 px | 2-D (3-D = ortho-plane averaging, not used) | yes |
| Cellpose 3 `cyto3` | same | same | 26.5 MB | same | cytoplasm (+ nuclear) channel | same | 2-D | no (we have a nuclear stain) |
| StarDist `2D_versatile_fluo` | `apps/stardist` | HF `neurodeskorg/webapps`; CORS `*` | 5.7 MB | BSD-3 | 1 channel NHWC, 1-99.8th pct, /16, grid 2 | prob 0.479, polygon NMS 0.3 (clipper-lib, Boost) | 2-D | yes |
| StarDist `3D_demo` | same | same | - | BSD-3 | anisotropy 2:1:1, 96 rays | polyhedron NMS | 3-D | no (demo model) |
| cellfinder detection | `apps/lightsheet-atlas` (JS port of cellfinder 1.10.1) | none (classical) | - | BSD-3 | uint16 raw, voxel sizes in um, soma 16 um default | LoG + mean+10 SD threshold, 3-D ball filter, splitting -> points | 3-D | yes (points drawn as spheres) |
| micro-SAM `vit_t_lm` / MobileSAM | `apps/microsam` | HF `neurodeskorg/webapps` | 28 + 16.5 MB | CC BY 4.0 (micro-sam), Apache-2.0 (MobileSAM) | 1024 x 1024 encoder | prompt decoder only; AMG / AIS automatic segmentation not in the port | 2-D (+ z propagation) | no: needs prompts |
| VesselBoost | `apps/vesselboost`, `packages/vesselboost` | HF `sbollmann/neurodesk-webapps-assets`; CORS `*` | 26.3 MB | code MIT; weights NOASSERTION | 64^3 patches, z-score | sigmoid, threshold 0.1, components >= 10 voxels | 3-D | yes, experimental (MRA-trained) |
| Big-FISH spots | `apps/spot-detection` | none | - | BSD-3 | LoG at spot radius | local maxima, elbow threshold | 2-D/3-D | no: diffraction-limited spots |
| `gpu-unet` kernels | `packages/runtime-support/src/gpu-unet` | - | - | - | WebGPU FP32 3-D U-Net kernels used by MRI apps | - | 3-D | no: onnxruntime-web covers our models |
| SynthSeg, MindGrab, Brainchop | MRI apps | - | - | - | MRI contrasts | - | 3-D | no: MRI-specific |

## How the comparison was run

`demo/test/nd_compare.mjs` (headless Chromium on a Slurm node: our programs on the WebGPU engine via SwiftShader, the
references with onnxruntime-web 1.29.0 on 4 WebAssembly threads) streams real level-0 windows from DANDI 000108 (MITU01;
the middle strip of each slab; per strip the 80th and 95th percentile blocks of a level-4 intensity ranking, i.e. tissue-rich
spots), runs every model on the same 208 x 208 x 9 voxel window (160 x 160 scored core + 24 px context margin; the central 5
planes are scored, the outer 2 on each side are context), exactly as the hook does, and dumps the label volumes;
`scripts/nd_compare.py` computes the numbers and the overlays. Objects are scored when their 3-D centroid lies in the core.
Matching: Hungarian assignment of centroids, tolerance half the declared diameter (YO 10 um -> 5 um; NN 16 um -> 8 um).
Chance: the same counts placed uniformly at random in the scored volume (20 draws). Our programs get the exposure window
from our `exposure` program (as in the viewer); the references get their own percentile normalization over the window's
central plane.

RESULTS
