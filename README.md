# handwired-brain-transformer

3D volumetric transformers whose weights are **written by hand (weight-space programming), not trained**.
They segment structures in the multiscale OME-Zarr light-sheet volumes of
[DANDI:000108](https://dandiarchive.org/dandiset/000108) (human brain slabs, stains YO / NN / LEC and others),
streaming chunks from the pyramid, and are made as small as the task allows.

## How a transformer is programmed

Token = voxel. Fixed embedding channels: `log I`, physical position `(z,y,x)` in µm, a constant 1, and the
position products `z², y², x², zy, zx, yx`. Everything else is a scratch channel that a program allocates.

**Gaussian attention is exactly representable.** Softmax over keys `j` of `-½ (pⱼ-pᵢ-δ)ᵀ P (pⱼ-pᵢ-δ)` equals a
normalized Gaussian kernel with precision `P`. The terms depending only on `i` cancel in the softmax, leaving
`q_i·k_j` with `q_i = [P(p_i+δ); 1]` and `k_j = [p_j; -½ p_jᵀ P p_j]`: pure linear maps of the embedding. So a head
is a Gaussian smoother, any anisotropy or orientation is a choice of `P`, and `δ` shifts it. (`tests/test_wiring.py`
checks this against an explicit kernel to 1e-9.) Other building blocks:

| Operation | How it is wired |
|---|---|
| Difference of Gaussians | two heads with opposite `Wo`, or `Wo` summed in an MLP |
| Oriented line filter | head with elongated `P` along direction `d` |
| max(a, b) | `b + relu(a − b)`, as a tournament over layers |
| Clamp / soft step | `relu(a) − relu(a − 1)` |
| In-context calibration | a *global* head (uniform attention = all-zero scores) gives tile statistics |
| Illumination invariance | the embedding uses `log(I+1)`, so DoG is a log-ratio |

## The input convention: the model sees what you see

Every program takes the **display image** `u'` in [0, 1] as its input channel, not raw intensity:
`u = clamp((log(I+1) - lo) / (hi - lo))`, then `u' = clamp((u - .5) * contrast + .5 + brightness) ** gamma`
(`hwseg.exposure.display`, identical to `demo/seg.js`). The viewer's brightness / contrast / gamma sliders therefore
change what the model sees, across the whole slab in view, and the window `[lo, hi]` is a property of the
field of view, not of a tile or a level. The default exposure comes from a program (`exposure`, below).

The models are **view-dependent**: the model sees exactly the fetched field of view and the depth slab in view, and its
in-context statistics (global heads) and the exposure window come from that region alone. Zooming in shrinks the FOV and
moves to a finer pyramid level (fewer, finer voxels); zooming out grows it and moves to coarser levels, so cost follows
the screen, not the data. Every program runs as **one forward pass over the FOV**. Only when the FOV does not fit in the
device's storage-buffer limit do cells / vessels fall back to tiles with per-tile statistics (the status line says so);
tissue and layers coarsen the level instead. Measured with Playwright (`demo/test/fov_zoom_test.mjs`), zooming
6 → 3 → 1.5 → 0.75 µm/px fetched 570×268 → 570×268 (level 1 → 0) → 286×134 → 144×68 voxels, and zooming out to the whole slab
fetched 597×64 voxels at level 5 in one pass.

## Programs

| Program | Zoom band (µm per screen px) | What it computes | Nonzero params |
|---|---|---|---|
| `exposure` | always | FOV mean and mean +/- deviations via global heads → recommended display window `[lo, hi]` and mean; viewer derives gamma. Coefficients are two scalars fitted in closed form against the 2nd–99.7th percentile window on real crops. | 15 |
| `fable_tissue` (default, YO/NN) / `tissue` (builder) | ≥ 8.5 | smoothed display image vs. FOV mean, thresholds relative to the FOV's own spread (global head) plus an absolute darkness floor → bg / gray / white | 53 |
| `layers` (**experimental**, never auto-selected) | explicit choice only, run at L5 | density indicators → pia and white-matter proximity (Gaussian attention at 150/400/900 µm) → signed depth → ReLU bands I–VI at depths declared in µm | ~320 |
| `fable_blobs_v2` (default) / `cells` (builder) | < 8.5 (YO/NN/other) | hand-written v2 at levels 0-1 / builder DoG on the display image; `seg = step(dog - kappa*mean|dog| - tau0)`, kappa 2, tau0 0.03 | ~50 |
| `vessels_13dir`, `vessels_26dir` | < 8.5 (LEC/CD31) | max over oriented elongated Gaussians minus wide background; kappa 1, tau0 0.12 | ~330 / ~650 |

**Scale families.** Gaussian attention is analytic in physical lengths, so every program exists at scales x1, x2, x4, x8
(all sigmas multiplied). Within a band the viewer picks the scale that keeps features about 1.5 screen pixels wide and
the coarsest pyramid level that still resolves it (nuclei → nuclear clusters, thin → thick vessels), which also lowers
cost as you zoom out. Equivariance is tested exactly: `program(scale=s)` on a volume with spacing `s·dx` equals
`program(scale=1)` on the original.

**Contrast robustness and the darkness-to-saturation progression.** Thresholds are `kappa * (FOV response spread) + tau0`:
the first term makes detection stable once an object is clearly visible, the absolute floor `tau0` (display units) makes
detection vanish when nothing is visible. Measured on phantoms with tau0 0.04 (cells F1 / vessels F1; the shipped cells floor is 0.03, chosen on real crops where
object counts stay within about 25% across gamma 0.6–2): default exposure 0.88 / 0.77;
contrast x2 0.81 / 0.67; gamma 0.5 0.62 / 0.23, gamma 2 0.69 / 0.55; brightness -0.3 0.88 / 0.79; contrast 0.5 0.36 / 0.02
(too dim to see: fewer detections); contrast 0.15 (flat) 0 / 0 and brightness +0.4 (saturated) 0.02 / 0: nothing.
Without the floor (relative threshold alone) the detectors hallucinate structure from noise in flat views.

Sizes are *nonzero* weights; dense tensors are larger (`scripts/export_models.py` prints both). Model size is a design
axis, not a trained outcome: for vessels, direction count trades accuracy for size (phantom oracle-threshold F1:
13 dirs 0.79, 26 dirs 0.86, 49 dirs 0.90).

## Evidence, and what it does not show

* **Programmed attention = its specification** (unit test, exact); **fast FFT path = dense attention** (1e-7);
  **JS engine and WebGPU engine = Python** on golden volumes for tissue, layers, cells and vessels (soft channels
  <= 3e-4, hard segmentation outputs identical; `demo/test/golden.json`).
* **Synthetic phantoms with exact ground truth** (`hwseg.phantom`): cells, vessels, and a cortical ribbon with six layers.
* **What each program picks up per pyramid level** (`scripts/zoom_sensitivity.py`, real streamed windows, 2 YO slabs):
  fixed-sigma cells/vessels have signal only at levels 0–2; at level >= 3 the response is exactly zero because sigma is
  below the voxel size, and levels 1–2 give speckle (object counts balloon 37 → 152 → 237 with no structure). Tissue
  is stable across levels 4–6 (about 25% gray, 52% white, 23% background). This is why the scale families exist.
* **Real streamed crops** (`scripts/stream_smoke*.py`, Playwright runs in `demo/test/`): cells and vessels land on
  visible nuclei and vessel walls. **No ground-truth labels ship with 000108, so there are no real-data accuracy numbers.**
* **`layers` is the weakest program.** On the phantom ribbon it is depth-ordered and within one layer for >85% of
  cortex voxels (exact accuracy about 0.5). On real crops it is **not trustworthy**: with FOV-relative thresholds,
  "not gray" is white matter by construction (any view calls about half of itself WM), the depth priors assume a
  perpendicular section through a flat ribbon, and real slabs are oblique cuts. It needs absolute density references
  or layer labels (annotation or atlas registration), which DANDI 000108 does not ship.
* **Not validated:** cross-slab robustness beyond a handful of slabs, and everything on real data (no labels).
  Thresholds (`kappa`, `tau0`, layer depth priors) are declared design constants calibrated on phantoms or anatomy
  priors, not fitted per slab. Vessel programs belong to LEC/CD31 stains; on nuclear stains they respond to cell bodies.

## Python in the browser (Pyodide) and Neuroglancer

* **Python on the main page**: the sidebar's "Python on this view" panel runs Python (Pyodide, Python 3.14 in WebAssembly, numpy;
  scipy loads on import) inside the Neuroglancer page. The `ng` module (`demo/py/ng_prelude.py`) reads the image under the current
  view (`v = await ng.read(depth=9)` -> numpy `log(I+1)`, with level, voxel spacing and origin) and writes results back
  (`await ng.segmentation(labels, view=v)`, `await ng.annotations(points, view=v)`). The default example smooths, thresholds
  with a local contrast and labels blobs with `scipy.ndimage`, then adds the segmentation and the centroids. A Python segmentation
  layer keeps its code and view parameters in the layer's source URL, so a shared link can regenerate it, but a link carrying code is
  never run silently: the page shows the code and asks first. (The numpy-only `hwseg` modules are still synced into `demo/py/` by
  `scripts/sync_browser_py.sh` and checked by a test, so `from hwseg.programs import cells` works in that Python.)
  What Pyodide cannot do: its package set has numpy, scipy, scikit-image, zarr, numcodecs, fsspec and aiohttp but no
  torch, onnxruntime, tornado or neuroglancer, so the Python Neuroglancer server and the torch executor do not run there;
  inference stays on WebGPU.
* **Neuroglancer**: the hosted standalone viewer exposes `window.viewer`; on a probe against it, the view state was
  readable (position, zoom, layers) and observable (about 118 change events), `restoreState` changed the view, and a local
  annotation layer with ellipsoids persisted when it declared its own `(z, y, x)` output space (the viewer space is
  `(t, z, y, x)`; a 3-D annotation in the wrong space is silently dropped). Hooking it from our own page needs our own
  build (same-origin); the Python wheel's bundled client is the Python-integration build and cannot be hosted standalone.
  Our own build is hosted on Pages under `/ng/` (`scripts/build_neuroglancer.sh` builds the pinned commit, Node >= 22.18;
  `scripts/assemble_pages.sh` adds it with `demo/ng-hook.js` injected; the build is not committed to `main`). The hook
  reads the view (the FOV is recovered from the data coordinates Neuroglancer reports under the panel corners), runs the
  programmed model on exactly that FOV with the WebGPU engine, and writes the detections back as an annotation layer; the
  image layer uses a shader that reproduces our display pipeline (log window from the exposure program, contrast,
  brightness, gamma sliders). Hard-won specifics: the annotation layer must be declared in the viewer's own (t, z, y, x)
  space (any other space reorders the global dimensions and resets the view); `restoreState` alone leaves a broken display
  plane, so the hook calls `state.reset()` first. Tissue results are drawn as contour lines (Neuroglancer annotations have no region type): a
  tissue outline and a gray/white boundary, each its own annotation layer styled with a thin-line annotation shader. `demo/test/ng_bridge_test.mjs` is the end-to-end test.

## Hand-written weights (`weights/fable/`)

Subagents wrote these weight files by hand (no optimizer, no builder imports, every number chosen and then adjusted by
looking at images): `fable_blobs_v2.json` and `fable_vessels_v2.json` (round 2, the **defaults for zoomed-in views**, levels
0-1), the heavier round-1 `fable_blobs.json` / `fable_vessels.json` (octave scales up to 80 um, selectable as experimental)
and `fable_exposure.json`. Generators, the iteration log and montages are in `weights/fable/` (`NOTES.md`, `figs/`).
The cell / vessel / exposure files are numerically identical in the Python, JavaScript and WebGPU engines (<= 1.4e-4 on soft channels, hard outputs exact).

| file | d | layers | nonzero params | what it does |
|---|---|---|---|---|
| `fable_blobs_v2` | 25 | 6 | 180 | nuclei / somata (`seg`, `nucleus`, `soma`), sigmas 3 / 6 / 12 um, noise-relative threshold |
| `fable_vessels_v2` | 37 | 8 | 565 | in-plane oriented tubes (`seg`, `capillary`, `vessel`), 6 directions x 2 scales |

Buffer cost of a 500 x 300-voxel field of view at level 0: blobs v2 72 MB, vessels v2 64 MB (device limit assumed 128 MB), so
one forward pass fits (round 1 did not and forced the viewer to coarsen the level to 3, where it detected nothing).

Compared on the same real stitched windows (I looked at the montages): on YO the builder cells program marks 1-4% of the
area (clean but conservative; it misses many nuclei in dense windows), blobs v2 marks 5-15% and covers nearly every bright
nucleus in YO_117, but merges neighbours and marks diffuse texture and a crack rim in YO_90; on NN, v2 covers somata more fully
(including a large soma the builder program reduces to a dot) and both stay empty on a noise-only window. On LEC the builder
vessels program marks almost nothing (0-1.5%; its floors were calibrated on phantoms), while vessels v2 traces the long and
the ring vessel and finds the small vessels in dim windows. Agent-reported phantom F1 at default exposure: blobs v2 0.94
(v1 0.76), vessels v2 0.49 (precision 0.50, recall 0.48; v1 0.41).

Known limitations (from the agent's report and my checks): vessels v2 does not reject real NeuN somata (marks 4-10% of NN
windows), so it is routed to LEC only; round bright vessel cross-sections in LEC are marked; vessels are in-plane only
(vessels crossing the depth slab are missed) and `capillary` is weak; at level 2 and coarser both programs return nothing
by design; contrast >= 2 and gamma still change the masks (IoU 0.2-0.7); blobs v2 can mark a crack rim; no real-data accuracy
numbers exist (no labels). `demo/test/fable_viewer.mjs` runs a hand-written model in the viewer.

### Tissue and cortex vs white matter (hand-written, YO/NN)

Engine check on a real level-6 window of a stitched NN slab (a sub-mm phantom is useless here: these coarse programs correctly abstain on it, which makes every comparison 0 = 0): WebGPU vs Python <= 4.8e-4 (tissue exact to 8e-6); the JS reference engine differs from Python by up to 1.3e-2 on soft channels of cortex / layers, whose very steep thresholds flip a few voxels under float32 rounding.

`fable_tissue.json` (d 57, 543 nonzero params), `fable_cortex_wm.json` (d 62, 598) and the experimental `fable_layers.json`
(d 69, 649) are hard class maps for pyramid levels 4-6 (whole-slab and multi-mm views). Tissue builds a soft CDF of the
field of view's display intensities with global heads and thresholds between the dark (background) and bright mode, with
gates that switch every output off in dark, flat or washed-out views. Cortex vs white matter is **geometric, not measured**:
cortex is tissue within about 3.3 mm of an open tissue edge (about 2.2 mm inside a narrow sulcus, about 1.9 mm either side of
a closed sulcus that shows as a thin dark line), white matter is tissue deeper than that; the layers program cuts the cortex
mask at about 1 mm and 1.9 mm from the pia. Both agents' reasoning and iteration logs are in `weights/fable/NOTES_tissue_layers.md`.
I ran them on the four stitched whole slabs next to my builder tissue program (`scripts/compare_regions.py`): the builder
program now labels all tissue "gray" and finds no white matter at all, while the hand-written maps give a clean tissue mask,
a cortex ribbon along the pial surface and white-matter cores with closed sulci and the ventricle handled
(`demo/test/ng_link_test.mjs` with `REGIONS=1` runs them on the whole stitched slab inside Neuroglancer: tissue outline and
cortex/white-matter boundary as contour annotations, one pass over a 779 x 290-voxel level-6 view).

Agent-reported robustness (mean IoU vs the default view over 9 fields of view): tissue 0.93-0.98 and background 0.89-0.97 for
contrast 0.5-4, gamma 0.5-2 and brightness +-0.15; cortex 0.72-0.94 and white matter 0.76-0.96 (13 of 126 cells below 0.7, all at
contrast >= 2 or gamma 2); layer bands are least stable (21 of 63 cells below 0.7). Dark, flat and saturated views detect nothing.

Known limitations: the cortex/WM boundary is a fixed distance from the pia and ignores real thickness changes; false cortex
rings around dark structures inside white matter (lateral ventricle, holes, tears, long dark vessels); closed sulci with no
visible dark line, and cortex whose pia is outside the field of view, are missed; the thin-line detector is the fragile part
(depends on level and contrast); out-of-plane haze at level 6 turns into tissue at contrast >= 2 on one slab. These programs
are only meaningful for the NN and YO stains (on LEC they return nothing or a full rectangle), so the router skips them for LEC.
Levels: tested at 5 and 6 by the agent; I read the same 12 x 8 mm field of view at levels 3, 4 and 5 (`scripts/regions_levels.py`): tissue is consistent (NN 100%, YO 95% at all three), but cortex vs white matter is **not scale-invariant** (white matter 36-41% at level 5, 12-16% at level 4, 0% at level 3 on NN and YO), so cortex/WM and the layer bands are restricted to level 5 and above. Levels 0-2 would need tens of millions of tokens for the 10 mm+ field of view these programs require, so they were not run; LEC is not applicable; no real-data accuracy numbers exist (no labels).

## Streaming, no bulk download

`hwseg.data.StreamedZarr` and the demo read only the zarr chunks a crop intersects (HTTP GET per chunk from the
DANDI S3 bucket) and decode them; an optional disk cache is hard-capped. Nothing downloads whole volumes.

## Demo: Neuroglancer with weight-programmed models

Live: https://sensein.github.io/handwired-brain-transformer/ (GitHub Pages, branch `gh-pages`; redeploy with
`NG_DIST=<built neuroglancer dist> scripts/deploy_pages.sh`).

The front page (`index.html`) is a Neuroglancer view embedded in a thin shell: a sidebar selects stain, sample, acquisition,
whole slab vs one strip, and the model; Neuroglancer's own shader panel gives lo / hi / contrast / brightness / gamma.
"Run on this view" streams the chunks of the current field of view (tiled into overlapping 384-voxel GPU passes, up to 100 tiles, centre first, so the whole view is covered), runs the model on your GPU (WebGPU) and adds the result as a Neuroglancer *segmentation* layer (voxel labels served from memory by the service worker as a `precomputed` volume:
one id per cell, class labels for vessels, gray / white matter for the region programs; the region programs label only the displayed plane)
plus hidden annotation layers (cell centres / extents, tissue outline and cortex/white-matter contours). The whole slab is a *virtual
precomputed volume* generated in the browser by our stitcher and served to Neuroglancer by a service worker (`sw.js`), so
adjacent strips are cross-faded into one continuous volume. The earlier stand-alone 2D viewer, its 3D renderer and the slab gallery were removed; Neuroglancer's own panels, depth keys and volume rendering replace them.
Cell and vessel detection in the Neuroglancer view is 3-D: the "Depth" selector (default 9 planes; 1 = the old single-plane mode) runs the
model on a z slab (plus margin planes so objects cut by the slab edge keep their z extent), splits every plane into 2-D blobs and links blobs in adjacent planes into objects (a 3-D connected-component pass merged
touching cells into tall ellipsoids and was dropped; a track also ends where its area dips below 60% of its peak and rises again),
and the linked blobs are the cell labels; each object also gets an (optional, hidden) ellipsoid with a measured in-plane radius (largest cross-section) and z radius (area-weighted spread, r = sqrt(5 var)). A detection belongs
to the slab if its centre lies in it; tile cores own detections in x-y. The attention itself already spans z, but with a thin
radius (cells ~5 um, vessels 2.5-6 um) so z context is a few planes; tissue and cortex/white-matter stay 2-D contours.
Selection is by (subject, stain, sample): slab numbers repeat across subjects (e.g. sample 28 is one MITU01 acquisition and two SChmi53 ones),
so the sidebar has a Subject selector (MITU01 first, as in the BICCN dashboard). "Copy link to this view" produces a link to this page with the
sidebar choices and the full Neuroglancer state (layers, view, shader, selected panel, segmentation opacity). Segmentation layers hold their run
parameters (model, level, box, z slab, exposure, image source) in the layer's source URL, so whoever opens the link regenerates them on demand in
their own browser; "Regenerate segmentations" recomputes them in the current page. A run covers the visible field of view plus a 15% margin,
and the "Segmentation opacity" slider sets Neuroglancer's own selected-segment opacity.
Cortex / white-matter and layer programs are only routed at pyramid level >= 5 (coarse): the cortex boundary is a fixed
distance from the pia and is not scale-invariant. Levels 0-2 need a field of view above ~10 mm, larger than a GPU buffer, so
region programs are not tested there.

* **Stitching**: whole stitched slabs, not just strips: a slab acquisition is 8–11 strips stacked along y (each ~7.4 mm tall, up to
    127 mm long, ~0.8 mm overlap, z and x origins drifting by up to ~0.5 mm). The viewer stitches them from the physical
    translation stored in each zarr (what Neuroglancer does with a multi-source layer), blending overlaps with a feathered
    average; overlaps are cross-faded with a smoothstep; a strip that covers most of an earlier strip's region is a re-acquisition and is left out of the stitch
    (listed as "strip N only"). YO sample 90 opens as one 2259 × 18542 × 49819-voxel slab at overview zoom. You can also
    **paste a dashboard / Neuroglancer link**: every DANDI zarr source in it is mapped to the stitched slab.
*Seams between strips*: visible horizontal banding remains in YO and NN. Measured (`scripts/seam_metric.py`, vertical-gradient
    energy inside strip overlaps over elsewhere; 1.0 = invisible): a local correction of the smooth log-difference field inside each
    overlap (tapered to zero at the overlap's end, so strip interiors are untouched) moves LEC 1.59 -> 1.35 (1.22 when estimated
    at a costlier level), is neutral on NN (1.07 -> 1.09) and slightly worse on YO (1.67 -> 1.72), so it is an opt-in
    "correct strip seams" toggle. Tried and rejected: chained per-strip gain offsets (accumulate a ramp; detrended they vanish) and a
    common flat-field profile along the strip (worse on NN). The banding therefore is not a simple per-strip gain or vignette; its
    cause is not established. Neuroglancer's native multi-source layer has no seam handling at all.

Zoom routing (µm per screen pixel `p`): `p ≥ 8.5` tissue (`fable_tissue`); `p < 8.5` cells (YO/NN/other stains) or vessels
(LEC/CD31); layers only by explicit choice; the scale family and pyramid level are then chosen from `p` (`demo/route.js`).

```bash
python -m http.server -d demo 8000     # then open http://localhost:8000 in a WebGPU browser
```
(or use the Pages link above)

* Streams OME-Zarr chunks straight from DANDI S3 in the browser (own blosc/zstd decoder, `demo/blosc.js`).
* Runs real softmax attention in WGSL compute shaders (online softmax over each head's window). Layer output ranges
  are planned backwards from the displayed plane, so attention is only evaluated where it is needed. Falls back to a
  slow JS reference if WebGPU is absent.
* Tested headless on SwiftShader (software WebGPU): correct but slow (seconds per tile). Speed on a real GPU is not
  measured here.

Browser tests (Playwright, headless Chromium; set `DEMO_ROOT`/`SHOT_DIR` if needed):
`node demo/test/gputest.mjs` compares the WebGPU engine to the Python golden outputs; `node demo/test/viewer_test.mjs`
drives the live viewer through zoom levels and screenshots each.

## Reference models from the Neurodesk web apps (`demo/nd/`)

Next to the hand-written programs, the model menu offers trained / classical **reference models** taken from
[neurodesk/webapps](https://github.com/neurodesk/webapps), so our programs can be compared with established tools on the
same view. They run in the same hook, on tiles of the field of view, and write through the same segmentation path (new
`P.kind` values `nd_cellpose`, `nd_stardist`, `nd_cellfinder`, `nd_vesselboost`; a shared link regenerates them).

| model | what | input / scale | post-processing | weights (fetched at run time) | licence |
|---|---|---|---|---|---|
| `nd_cellpose` | Cellpose 3 `nuclei` (2-D U-Net, flows) | per plane, 1-99th percentile normalization over the FOV's displayed plane, plane resampled so the declared diameter = 17 px (diam_mean), 224-px tiles, 10% overlap | flow dynamics (200 steps), mask clustering, flow-error QC 0.4, hole filling, min 15 px (Neurodesk port of cellpose 3.1.1.1) | 26.5 MB ONNX, Neurodesk HF dataset `neurodeskorg/webapps` @bc1de48 | BSD-3 code + weights; **training data partly CC-BY-NC** (check before commercial use) |
| `nd_stardist` | StarDist `2D_versatile_fluo` | per plane, 1-99.8th percentile normalization, resampled so the diameter = 16 px, NHWC | star-convex polygons, prob 0.479, polygon NMS 0.3 (clipper-lib 6.4.2 from CDN) | 5.7 MB ONNX, `neurodeskorg/webapps` @2cc5042 | BSD-3 (clipper-lib: Boost) |
| `nd_cellfinder` | cellfinder 1.10.1 candidate detection (classical, 3-D) | raw intensities (uint16), voxel sizes in um, soma diameter = declared diameter, +8 context planes each side | per-plane LoG + threshold (mean + 3 SD here; cellfinder's 10 SD default finds nothing in dense pan-nuclear YO), 3-D ball filter, structure splitting; candidate *points* drawn as spheres of the declared diameter | none | BSD-3 |
| `nd_vesselboost` (experimental) | VesselBoost 3-D U-Net (trained on MR angiography) | 64-plane slab, z-score over the tile, 64^3 patches | sigmoid, Gaussian-weighted (sigma 8) patch blending, threshold 0.1, drop components < 10 voxels | 26.3 MB ONNX, `sbollmann/neurodesk-webapps-assets` @a8cdbf8 | code MIT; **weights carry no licence statement** (NOASSERTION): not redistributed |

All weights are downloaded by the browser from Hugging Face (CORS `*`, revision-pinned URLs, size and sha256 checked
before use) and executed with onnxruntime-web 1.29.0 from jsDelivr (WebGPU execution provider, WebAssembly fallback; WebAssembly on a
software WebGPU adapter, where the WebGPU provider was measured more than 25x slower: StarDist on 8 tiles did not finish in
14 min vs 30 s; 3-D convolutions on WebAssembly, as in the Neurodesk app; `?ndep=wasm|webgpu` forces one). Nothing is vendored except the BSD-3 JavaScript pre/post-processing
ports (`cellpose-core.js`, `stardist-core.js`, `cellfinder-detect.js`, with their licence and notice files). The reference
models therefore also run without WebGPU.

**Scale.** The networks expect objects of a fixed size in pixels, so the hook runs them at the coarsest pyramid level where
the declared object spans at least 4 voxels (level 0 for 10-16 um nuclei: DANDI 000108 voxels are 2.56 x 3.63 um in x / y,
anisotropic in plane) and resamples each plane to square pixels at the network's size. The diameter is a sidebar setting
(default YO 10 um, NN 16 um: YO nuclei measured about 10 um by a LoG scale search on real windows; NeuN somata are larger);
it is stored in the layer's parameters. The planes' instances are linked across z with the same rule as our programs
(`demo/instances.js`), and tiles keep their cores (160-192 px), up to 36 tiles centre first. Cost: per 208 x 208 x 9 window on
4 WASM threads (headless, SwiftShader node) Cellpose took minutes, StarDist and VesselBoost seconds to tens of seconds; on a
real GPU with the WebGPU provider it should be much faster (not measured). Label-free comparison with our programs:
[`docs/neurodesk/COMPARE.md`](docs/neurodesk/COMPARE.md) (`demo/test/nd_compare.mjs` + `scripts/nd_compare.py`);
`demo/test/nd_bridge_test.mjs` runs a reference model inside Neuroglancer and checks labels across the field of view and
their regeneration from the link (SwiftShader node, 1 mm x 0.3 mm view at 1.3 um/px, 8 tiles: StarDist 964 objects in 28 s,
Cellpose 306 in 203 s, cellfinder over 5 planes 196 in 4 s; labels in 16/16, 16/16 and 13/16 grid blocks, identical after
regeneration).

Not integrated: micro-SAM / MobileSAM (the Neurodesk port is prompt-driven; automatic instance segmentation, AMG/AIS, is
not part of it, and a 1024 x 1024 encoder pass per tile is heavy), the 3-D StarDist `3D_demo` model (labelled a demo
model upstream; 2-D per plane plus linking keeps one code path), Big-FISH spot detection (diffraction-limited spots, not nuclei), and the MRI models (SynthSeg, MindGrab, Brainchop).

## Development

```bash
uv sync --extra dev
uv run pytest -q
uv run python scripts/export_models.py            # weights -> demo/models/*.json
uv run python scripts/make_golden.py demo/test/golden.json
uv run python scripts/build_asset_index.py demo/assets.json
```

`src/hwseg/`: `wiring.py` (attention/MLP programming primitives), `programs.py`, `model.py` (executor, dense and
FFT paths), `data.py` (streaming), `phantom.py`. Scripts under `scripts/` include the design sweeps used to pick
vessel parameters.
