"""`ng`: Python <-> Neuroglancer bridge, loaded into Pyodide inside the Neuroglancer page.
   v = await ng.read(depth=9)                      # log(I+1) around the current view (numpy, shape (z, y, x))
   await ng.segmentation(labels_uint32, view=v)     # show as a segmentation layer (zyx, same shape as v.data)
   await ng.annotations(points_zyx, view=v)         # ellipsoid annotations at voxel coordinates of v
"""
import sys, types
import numpy as np
import js
from pyodide.ffi import to_js


class View:
    def __init__(self, r):
        self.data = np.frombuffer(r.data.to_py(), dtype=np.float32).reshape(tuple(r.shape.to_py()))   # log(I+1), (z, y, x)
        self.level = int(r.level)                                  # pyramid level the data comes from
        self.spacing_um = tuple(float(s) for s in r.spacing.to_py())   # voxel size (z, y, x) in micrometres at this level
        self.origin = tuple(int(o) for o in r.origin.to_py())      # level voxel index of data[0, 0, 0] (z, y, x)
        self.plane = int(r.plane)                                  # index along z of the displayed plane
        self.um_per_px = float(r.umPerPx)                          # size of one screen pixel in micrometres

    def __repr__(self):
        return f"View(shape={self.data.shape}, level={self.level}, spacing_um={self.spacing_um}, um_per_px={self.um_per_px:.2f})"


async def read(level=None, depth=9, margin=0.15):
    """Crop of the image under the current Neuroglancer view (plus `margin`), `depth` planes around the displayed plane.
    level=None picks the pyramid level whose voxels are about one screen pixel."""
    return View(await js.__hwPy.read(level, depth, margin))


async def segmentation(labels, view, name="python segmentation"):
    """Add `labels` (integer array, shape == view.data.shape, 0 = background) as a Neuroglancer segmentation layer."""
    lab = np.ascontiguousarray(labels, dtype=np.uint32)
    if lab.shape != view.data.shape:
        raise ValueError(f"labels shape {lab.shape} != view shape {view.data.shape}")
    await js.__hwPy.segmentation(to_js(lab.ravel()), to_js(list(lab.shape)), name)


async def annotations(points, view, radii_um=None, name="python annotations"):
    """Ellipsoid annotations at `points` (N x 3, voxel indices (z, y, x) inside view.data); radii_um: scalar or length-N array."""
    p = np.ascontiguousarray(np.asarray(points, dtype=np.float32).reshape(-1, 3))
    r = np.full(len(p), 6.0, np.float32) if radii_um is None else np.broadcast_to(np.asarray(radii_um, np.float32), (len(p),)).copy()
    await js.__hwPy.annotations(to_js(p.ravel()), to_js(r), name)


ng = types.ModuleType("ng")
ng.read, ng.segmentation, ng.annotations, ng.View = read, segmentation, annotations, View
sys.modules["ng"] = ng
