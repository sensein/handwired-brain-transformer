// Neuroglancer state for a DANDI 000108 slab / strip: our display pipeline as a shader (log window lo/hi from the exposure
// program, contrast, brightness, gamma as uicontrol sliders), explicit dimensions and displayDimensions.
export const NG_SHADER = `#uicontrol float lo slider(min=0, max=12, step=0.05, default=4)
#uicontrol float hi slider(min=0, max=12, step=0.05, default=9)
#uicontrol float contrast slider(min=0.2, max=4, step=0.05, default=1)
#uicontrol float brightness slider(min=-0.5, max=0.5, step=0.01, default=0)
#uicontrol float gamma slider(min=0.2, max=3, step=0.05, default=1)
void main() {
  float l = log(toNormalized(getDataValue()) * 65535.0 + 1.0);   // uint16 counts
  float u = clamp((l - lo) / max(hi - lo, 0.01), 0.0, 1.0);
  u = clamp((u - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0);
  emitGrayscale(pow(u, gamma));
}`;
export const ZARR_URL = (id) => `zarr://https://dandiarchive.s3.amazonaws.com/zarr/${id}/`;
// ids: zarr ids of the strips (one id = one strip; several = the stitched slab as a multi-source layer, like the dashboard)
// L0: level-0 geometry {sp:[z,y,x] um, off:[z,y,x] um translation of the volume origin}; view: {cx, cy (um, local), z (level-0 index), p (um/px)}
export function ngState(ids, L0, view, exposure = null) {
  const sp = L0.sp, off = L0.off, um = (v) => v * 1e-6;
  return {
    dimensions: { t: [1, 's'], z: [um(sp[0]), 'm'], y: [um(sp[1]), 'm'], x: [um(sp[2]), 'm'] },
    position: [0, view.z + off[0] / sp[0], view.cy / sp[1] + off[1] / sp[1], view.cx / sp[2] + off[2] / sp[2]],
    crossSectionScale: view.p / sp[2], layout: 'xy', displayDimensions: ['x', 'y', 'z'], showDefaultAnnotations: false,
    selectedLayer: { visible: true, layer: ids.length > 1 ? `slab (${ids.length} strips)` : 'strip', },   // opens the layer panel with the shader sliders (lo / hi / contrast / brightness / gamma)
    layers: [{ type: 'image', name: ids.length > 1 ? `slab (${ids.length} strips)` : 'strip', source: ids.length > 1 ? ids.map(ZARR_URL) : ZARR_URL(ids[0]), tab: 'rendering', shader: NG_SHADER,
      shaderControls: exposure ? { lo: +exposure.lo.toFixed(2), hi: +exposure.hi.toFixed(2), gamma: +exposure.gamma.toFixed(2) } : {} }],
  };
}

// The whole stitched slab as ONE virtual Neuroglancer `precomputed` volume, computed on demand by our stitcher (smoothstep
// cross-fade, optional seam correction) and served by the service worker (sw.js): no native overlap bands. Its coordinate
// space is the slab's own level-0 voxel grid (x, y, z; origin 0), so there are no translations to account for.
export const virtualUrl = (ids, seams = false) => new URL(`virtual/slab/${ids.join('+')}${seams ? '~s' : ''}`, location.href).href;
export function ngStateVirtual(ids, L0, view, seams = false) {
  const sp = L0.sp, um = (v) => v * 1e-6, name = `slab (${ids.length} strips, stitched)`;
  return {
    dimensions: { x: [um(sp[2]), 'm'], y: [um(sp[1]), 'm'], z: [um(sp[0]), 'm'] },
    position: [view.cx / sp[2], view.cy / sp[1], view.z], crossSectionScale: view.p / sp[2], layout: 'xy', displayDimensions: ['x', 'y', 'z'], showDefaultAnnotations: false,
    selectedLayer: { visible: true, layer: name, },
    layers: [{ type: 'image', name, source: 'precomputed://' + virtualUrl(ids, seams), tab: 'rendering', shader: NG_SHADER, shaderControls: {} }],
  };
}
