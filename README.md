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

## Programs

| Program | Scale | What it computes | Nonzero params |
|---|---|---|---|
| `tissue` | ≥ 30 µm/px (pyramid L4–L5) | smoothed log-intensity (nuclear-density proxy) vs. tile-global mean, via a global attention head → bg / gray / white scores | 32 |
| `layers` | 8.5–30 µm/px (run at L5, ~82 µm voxels, 64-voxel tiles) | density indicators → pia and white-matter proximity (bg / WM indicators smoothed by Gaussian attention at 150/400/900 µm) → signed depth `s` → soft ReLU bands for layers I–VI, edges computed from declared depths in µm, gated by tissue-and-not-WM | 313 |
| `cells` | < 8.5 µm/px, run at 2.6 µm (L0) | DoG blob detector on `log I` (σ 3.5 µm, ratio 1.6) | 33 |
| `vessels_13dir`, `vessels_26dir` | 2.6 µm (L0) | max over oriented elongated Gaussians (σ_l 8 µm, σ_t 2 µm) minus a wide background | 330 / 642 |

Sizes are *nonzero* weights; dense tensors are larger (`scripts/export_models.py` prints both). Model size is a
design axis, not a trained outcome: for vessels, direction count trades accuracy for size (phantom oracle-threshold
F1: 13 dirs 0.79, 26 dirs 0.86, 49 dirs 0.90).

## Evidence, and what it does not show

* **Programmed attention = its specification** (unit test, exact), **fast FFT path = dense attention** (1e-7), and
  **JS engine and WebGPU engine = Python** on golden volumes (≤ 1e-4; tissue, layers, cells, vessels; `demo/test/golden.json`).
* **Synthetic phantoms with exact ground truth** (`hwseg.phantom`): cells and vessels with Poisson + multiplicative
  noise. `tests/` requires F1 > 0.5 for both programs at the calibrated contrast.
* **Real streamed crops** (`scripts/stream_smoke.py`): cells and vessels land on visible nuclei and vessel walls in
  YO/NN and LEC slabs. **No ground-truth labels ship with 000108, so there are no real-data accuracy numbers.**
* **`layers` is the weakest program.** On a synthetic cortical ribbon (`make_cortex_phantom`) it is depth-ordered and
  within one layer for >85% of cortex voxels (exact-layer accuracy is about 0.5; edges blur by design). On real crops
  it is **not trustworthy**: the declared depth priors assume a perpendicular section through a flat ribbon, the
  real slabs are oblique cuts, and large-scale white-matter signal dominates the depth proxy, so most tissue lands in
  L5/L6/WM and layers I-IV are nearly empty. Calibrating it needs layer labels (manual annotation or atlas
  registration), which DANDI 000108 does not ship.
* **Tissue v2** (density relative to the tile mean) separates dark white matter from brighter gray on real crops far
  better than v1 (which labeled almost everything white), but it calibrates against each tile's own statistics, so
  tile seams are visible.
* **Not validated:** cross-slab robustness beyond a handful of slabs, and everything on real data (no labels).
  Thresholds (`tau`, `k_bg`, `k_gm`, layer depth priors) are declared design constants calibrated on phantoms or
  anatomy priors, not fitted per slab. Vessel programs belong to LEC/CD31 stains; on nuclear stains they respond to
  cell bodies.

## Streaming, no bulk download

`hwseg.data.StreamedZarr` and the demo read only the zarr chunks a crop intersects (HTTP GET per chunk from the
DANDI S3 bucket) and decode them; an optional disk cache is hard-capped. Nothing downloads whole volumes.

## Demo: WebGPU viewer and slab gallery

Live: https://sensein.group/handwired-brain-transformer/ (GitHub Pages, branch `gh-pages`; redeploy with
`scripts/deploy_pages.sh`). If the custom domain is unavailable use
https://sensein.github.io/handwired-brain-transformer/.

* **Viewer** (`index.html`): pick a slab and pan/zoom; models follow the zoom.
* **Slab gallery** (`slabs.html`): one column per slab, one row per stain (YO / NN / LEC), a pyramid-level selector
  that drives the model choice, per-cell stats (class fractions, cells/mm² in the plane, vessel area) and a CSV
  export. Each cell streams a window at the brightest tissue location of that slab and links into the viewer.

Zoom routing (µm per screen pixel `p`): `p ≥ 30` tissue; `8.5 ≤ p < 30` layers; `p < 8.5` cells (YO/NN/other stains) or vessels (LEC/CD31).

```bash
python -m http.server -d demo 8000     # then open http://localhost:8000 in a WebGPU browser
```
(or use the Pages link above)

* Streams OME-Zarr chunks straight from DANDI S3 in the browser (own blosc/zstd decoder, `demo/blosc.js`).
* Pick stain / slab / chunk / plane; pan and zoom.
* **Auto** routes by zoom (µm per screen pixel): tissue when zoomed out, cells (YO/NN etc.) or vessels (LEC/CD31)
  when zoomed in; the pyramid level follows. Any model can be forced from the dropdown.
* Runs real softmax attention in WGSL compute shaders (online softmax over each head's window), tiled over the
  viewport, nearest the center first. Layer output ranges are planned backwards from the displayed plane, so
  attention is only evaluated where it is needed. Falls back to a slow JS reference if WebGPU is absent.
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
