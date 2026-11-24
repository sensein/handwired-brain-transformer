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
  fable_blobs_v2: { kind: 'blobs', invariant: true, fineOnly: true, maxLevel: 1 },
  fable_vessels_v2: { kind: 'vessels_ms', invariant: true, fineOnly: true, maxLevel: 1 },
};
export const MAX_SCALE = { cells: 8, vessels_26dir: 8, vessels_13dir: 8, tissue: 4 };

// Which family the zoom calls for (p = um per screen pixel): tissue >= 8.5 (finest tissue scale first), cells/vessels below.
// 'layers' is never auto-selected: Playwright runs on real slabs showed it hallucinating (e.g. 91-98% layer I on a LEC
// view, ~60% L6 + 30% WM on YO), so it is available only by explicit choice and labelled experimental.
export function familyForZoom(p, stain) {
  if (p >= 8.5) return 'tissue';
  return ['LEC', 'CD31'].includes(stain) ? 'fable_vessels_v2' : 'fable_blobs_v2';   // hand-written v2 programs beat the builder ones on real windows
}

// scaleSel: 'auto' or a number. Returns {model (key in models/index.json), family, scale, level, sigmaUm}.
export function scaledModel(family, p, zarr, scaleSel = 'auto') {
  const F = FAMILIES[family], top = zarr.top;
  if (F.fixed) return { model: family, family, scale: 1, level: Math.min(5, top), sigmaUm: null };
  if (F.invariant) {   // coarsest level whose voxel is no larger than a screen pixel: cost follows the screen, not the data
    let lv = 0; for (let l = 0; l <= top; l++) if (zarr.levels[l].sp[2] <= Math.max(p, zarr.levels[0].sp[2])) lv = l;
    return { model: family, family, scale: 1, level: Math.min(lv, F.maxLevel ?? lv), sigmaUm: null };
  }
  let s = scaleSel === 'auto' ? 2 ** Math.round(Math.log2(Math.max(p, F.ref) / F.ref)) : +scaleSel;
  s = Math.min(Math.max(s, 1), MAX_SCALE[family]);
  const sigma = F.sigma0 * s;
  let level = 0;
  for (let l = 0; l <= top; l++) if (zarr.levels[l].sp[2] <= sigma / 1.2) level = l;   // coarsest level with voxel <= sigma/1.2
  return { model: s === 1 ? family : `${family}_x${s}`, family, scale: s, level, sigmaUm: sigma };
}

export function scaleLabel(kind, scale, sigmaUm) {
  if (kind === 'blobs') return 'nucleus / soma (hand-written)';
  if (kind === 'vessels_ms') return 'capillary / vessel (hand-written)';
  if (kind === 'cells') return scale === 1 ? 'nuclei / somata' : `nuclear clusters (sigma ${sigmaUm.toFixed(0)} um)`;
  if (kind === 'vessels') return scale === 1 ? 'vessels' : `vessels, thick (sigma_t ${sigmaUm.toFixed(0)} um)`;
  if (kind === 'tissue') return scale === 1 ? 'tissue' : `tissue, coarse (sigma ${sigmaUm.toFixed(0)} um)`;
  return kind;
}
