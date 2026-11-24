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

The models are **view-dependent**: the in-context statistics (global heads) are computed over the field of view and the
depth slab in view, nothing outside it. Tissue and layers run as one forward pass over the visible region (no tile grid,
no seams); cells and vessels are purely local, so tiling them is only a memory optimization.

## Programs

| Program | Zoom band (µm per screen px) | What it computes | Nonzero params |
|---|---|---|---|
| `exposure` | always | FOV mean and mean +/- deviations via global heads → recommended display window `[lo, hi]` and mean; viewer derives gamma. Coefficients are two scalars fitted in closed form against the 2nd–99.7th percentile window on real crops. | 15 |
| `tissue` | ≥ 8.5 | smoothed display image vs. FOV mean, thresholds relative to the FOV's own spread (global head) plus an absolute darkness floor → bg / gray / white | 53 |
| `layers` (**experimental**, never auto-selected) | explicit choice only, run at L5 | density indicators → pia and white-matter proximity (Gaussian attention at 150/400/900 µm) → signed depth → ReLU bands I–VI at depths declared in µm | ~320 |
| `cells` | < 8.5 (YO/NN/other) | DoG on the display image; `seg = step(dog - kappa*mean|dog| - tau0)`, kappa 2, tau0 0.03 | ~50 |
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

## Streaming, no bulk download

`hwseg.data.StreamedZarr` and the demo read only the zarr chunks a crop intersects (HTTP GET per chunk from the
DANDI S3 bucket) and decode them; an optional disk cache is hard-capped. Nothing downloads whole volumes.

## Demo: WebGPU viewer and slab gallery

Live: https://sensein.group/handwired-brain-transformer/ (GitHub Pages, branch `gh-pages`; redeploy with
`scripts/deploy_pages.sh`). If the custom domain is unavailable use
https://sensein.github.io/handwired-brain-transformer/.

* **Viewer** (`index.html`): pick a slab, pan / zoom; models, scale and pyramid level follow the zoom.
  * *Depth like Neuroglancer*: Shift+wheel, PageUp/PageDown or arrow keys step through z (Shift: x10); a slab-thickness
    option shows a max projection or mean over 1–31 planes.
  * *Exposure*: brightness / contrast / gamma sliders on top of the model-recommended window and gamma (toggle "auto
    exposure"); the models always see the adjusted display image.
  * *3D*: a WebGL2 ray-marcher (MIP, composite with early termination, isosurface; rotate / zoom; threshold, gain,
    quality) renders the current field of view and depth slab from an 8-bit 3D texture, optionally overlaying the
    segmentation computed for the whole volume (WebGPU engine returns every plane of the slab).
* **Slab gallery** (`slabs.html`): one column per slab, one row per stain (YO / NN / LEC), a pyramid-level selector
  that drives the model choice, the same exposure controls, per-cell stats (class fractions, cells/mm² in the plane,
  vessel area) and a CSV export. Each cell streams a window at the brightest tissue location of that slab and links
  into the viewer (`index.html#id=...&p=...`).

Zoom routing (µm per screen pixel `p`): `p ≥ 8.5` tissue; `p < 8.5` cells (YO/NN/other stains) or vessels
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
