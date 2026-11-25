// Zoom routing with scale families. Gaussian attention is analytic in physical lengths, so each program exists at
// scales x1, x2, x4, ... (every sigma multiplied). Within a program's zoom band the scale grows with the pixel size so
// features stay a constant number of screen pixels wide (nuclei -> nuclear clusters, vessels -> thicker vessels,
// finer tissue -> coarser tissue), and the coarsest pyramid level that still resolves that scale is used.
export const FAMILIES = {
  cells: { kind: 'cells', sigma0: 3.5, ref: 2.56 },
  vessels_26dir: { kind: 'vessels', sigma0: 2.0, ref: 2.56 },
  vessels_13dir: { kind: 'vessels', sigma0: 2.0, ref: 2.56 },
  tissue: { kind: 'tissue', sigma0: 60, ref: 60 },
  layers: { kind: 'layers', fixed: true },
  // hand-written, multi-scale and scale-selective: one weight file works at every pyramid level (small scales collapse harmlessly)
  fable_blobs: { kind: 'blobs', invariant: true },
  fable_vessels: { kind: 'vessels_ms', invariant: true },
  // round 2: cheap fine-scale programs (levels 0-1 only; every scale is gated off from level 2 up), default for zoomed-in views
  // hand-written region maps (hard classes), validated at pyramid levels 4-6; planar windows up to 3.84 mm
  fable_tissue: { kind: 'tissue2', invariant: true, minLevel: 4 },
  fable_cortex_wm: { kind: 'cortex_wm', invariant: true, minLevel: 5 },   // NOT scale-invariant: same 12x8 mm FOV gives wm 36-41% at L5, 12-16% at L4, 0% at L3 (scripts/regions_levels.py)
  fable_layers: { kind: 'layers2', invariant: true, minLevel: 5 },
  fable_blobs_v2: { kind: 'blobs', invariant: true, fineOnly: true, maxLevel: 1 },
  fable_vessels_v2: { kind: 'vessels_ms', invariant: true, fineOnly: true, maxLevel: 1 },
  // hierarchical hand-programmed transformers (docs/hierarchical/SPEC.md): ONE weight file for every zoom. The program reads the
  // voxel size as an input, normalises contrast itself (rank front end) and selects the feature scale by response, so there is no
  // level routing, no _xN variant and no exposure program. The level is only a data-cost choice: the coarsest level whose voxel is
  // no larger than a screen pixel (what is drawn is what is analysed).
  hier_cells: { kind: 'blobs', invariant: true, hier: true },
  hier_vessels: { kind: 'vessels_ms', invariant: true, hier: true },
  hier_regions: { kind: 'regions', invariant: true, hier: true },
};
export const MAX_SCALE = { cells: 8, vessels_26dir: 8, vessels_13dir: 8, tissue: 4 };

// Which family the zoom calls for (p = um per screen pixel): tissue >= 8.5 (finest tissue scale first), cells/vessels below.
// 'layers' is never auto-selected: Playwright runs on real slabs showed it hallucinating (e.g. 91-98% layer I on a LEC
// view, ~60% L6 + 30% WM on YO), so it is available only by explicit choice and labelled experimental.
export function familyForZoom(p, stain, opts = {}) {
  if (opts.hier) return ['LEC', 'CD31'].includes(stain) ? (p >= 8.5 ? 'none' : 'hier_vessels') : (p >= 8.5 ? 'hier_regions' : 'hier_cells');
  if (p >= 8.5) return ['LEC', 'CD31'].includes(stain) ? 'none' : 'fable_tissue';   // tissue models are for the nuclear / NeuN stains; LEC is a vessel stain
  return ['LEC', 'CD31'].includes(stain) ? 'fable_vessels_v2' : 'fable_blobs_v2';   // hand-written v2 programs beat the builder ones on real windows
}

// scaleSel: 'auto' or a number. Returns {model (key in models/index.json), family, scale, level, sigmaUm}.
export function scaledModel(family, p, zarr, scaleSel = 'auto') {
  const F = FAMILIES[family], top = zarr.top;
  if (F.fixed) return { model: family, family, scale: 1, level: Math.min(5, top), sigmaUm: null };
  if (F.invariant) {   // coarsest level whose voxel is no larger than a screen pixel: cost follows the screen, not the data
    let lv = 0; for (let l = 0; l <= top; l++) if (zarr.levels[l].sp[2] <= Math.max(p, zarr.levels[0].sp[2])) lv = l;
    return { model: family, family, scale: 1, level: Math.max(F.minLevel ?? 0, Math.min(lv, F.maxLevel ?? lv, zarr.top)), sigmaUm: null };
  }
  let s = scaleSel === 'auto' ? 2 ** Math.round(Math.log2(Math.max(p, F.ref) / F.ref)) : +scaleSel;
  s = Math.min(Math.max(s, 1), MAX_SCALE[family]);
  const sigma = F.sigma0 * s;
  let level = 0;
  for (let l = 0; l <= top; l++) if (zarr.levels[l].sp[2] <= sigma / 1.2) level = l;   // coarsest level with voxel <= sigma/1.2
  return { model: s === 1 ? family : `${family}_x${s}`, family, scale: s, level, sigmaUm: sigma };
}

export function scaleLabel(kind, scale, sigmaUm) {
  if (kind === 'regions') return 'tissue / cortical depth bands / white matter (hierarchical, one program for all zooms)';
  if (kind === 'tissue2') return 'tissue vs background (hand-written)';
  if (kind === 'cortex_wm') return 'cortex vs white matter (geometric, hand-written)';
  if (kind === 'layers2') return 'distance-from-pia bands (experimental)';
  if (kind === 'blobs') return 'nucleus / soma (hand-written)';
  if (kind === 'vessels_ms') return 'capillary / vessel (hand-written)';
  if (kind === 'cells') return scale === 1 ? 'nuclei / somata' : `nuclear clusters (sigma ${sigmaUm.toFixed(0)} um)`;
  if (kind === 'vessels') return scale === 1 ? 'vessels' : `vessels, thick (sigma_t ${sigmaUm.toFixed(0)} um)`;
  if (kind === 'tissue') return scale === 1 ? 'tissue' : `tissue, coarse (sigma ${sigmaUm.toFixed(0)} um)`;
  return kind;
}
