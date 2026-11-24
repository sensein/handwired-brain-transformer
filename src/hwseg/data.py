"""Streaming access to DANDI 000108 OME-Zarr pyramids. Never downloads whole volumes:
only the chunks intersecting a requested crop are fetched (HTTP range/GET per chunk) and decoded.
An optional on-disk chunk cache is hard-capped (default 512 MB, LRU eviction)."""
from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass

import numpy as np

API = "https://api.dandiarchive.org/api"
DANDISET = "000108"


def list_zarrs(subject="sub-MITU01", stains=None):
    """Return [(asset_id, path)] of ome.zarr assets via the API (metadata only, a few hundred KB)."""
    out, url = [], f"{API}/dandisets/{DANDISET}/versions/draft/assets/?page_size=1000&path={subject}"
    while url:
        d = json.load(urllib.request.urlopen(url))
        out += [(r["asset_id"], r["path"]) for r in d["results"] if r["path"].endswith(".ome.zarr")]
        url = d["next"]
    if stains:
        out = [o for o in out if re.search(r"stain-(\w+?)_", o[1]) and re.search(r"stain-(\w+?)_", o[1]).group(1) in stains]
    return out


def s3_url(asset_id):
    d = json.load(urllib.request.urlopen(f"{API}/assets/{asset_id}/"))
    return next(u for u in d["contentUrl"] if "s3.amazonaws.com/zarr" in u).rstrip("/")


@dataclass
class Level:
    path: str
    shape: tuple
    spacing_um: tuple  # (z,y,x)
    tr: tuple = (0.0, 0.0, 0.0)  # OME translation (um) of the volume origin


class StreamedZarr:
    def __init__(self, url: str, cache_dir: str | None = None, cache_mb: int = 512):
        import fsspec
        import zarr

        fs = fsspec.filesystem("http")
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            fs = fsspec.filesystem("filecache", target_protocol="http", cache_storage=cache_dir,
                                   same_names=True, expiry_time=0)
            self._cache_dir, self._cache_mb = cache_dir, cache_mb
        self.url = url
        self.grp = zarr.open_group(fs.get_mapper(url), mode="r")
        ms = self.grp.attrs["multiscales"][0]
        self.levels = []
        for ds in ms["datasets"]:
            sc = next(t["scale"] for t in ds["coordinateTransformations"] if t["type"] == "scale")
            tr = next((t["translation"] for t in ds["coordinateTransformations"] if t["type"] == "translation"), [0, 0, 0, 0, 0])
            self.levels.append(Level(ds["path"], tuple(self.grp[ds["path"]].shape), tuple(sc[-3:]), tuple(tr[-3:])))

    def level_for_spacing(self, target_um: float) -> int:
        """Coarsest level whose finest axis spacing <= target (or finest available)."""
        best = 0
        for i, L in enumerate(self.levels):
            if min(L.spacing_um) <= target_um:
                best = i
        return best

    def crop(self, level: int, z0z1, y0y1, x0x1) -> np.ndarray:
        arr = self.grp[self.levels[level].path]
        v = arr[0, 0, z0z1[0]:z0z1[1], y0y1[0]:y0y1[1], x0x1[0]:x0x1[1]]
        self._evict()
        return np.asarray(v)

    def _evict(self):
        d = getattr(self, "_cache_dir", None)
        if not d:
            return
        files = [(os.path.getmtime(os.path.join(r, f)), os.path.join(r, f), os.path.getsize(os.path.join(r, f)))
                 for r, _, fs in os.walk(d) for f in fs]
        tot = sum(s for *_, s in files)
        for _, p, s in sorted(files):
            if tot <= self._cache_mb * 2**20:
                break
            os.remove(p)
            tot -= s


def to_logI(a: np.ndarray, eps: float = 1.0) -> np.ndarray:
    return np.log(a.astype(np.float32) + eps)


class StreamedSlab:
    """Several strips (chunks) of one slab acquisition stitched by their OME translations, like Neuroglancer / demo/zarr.js:
    re-acquired strips (covering most of an earlier strip's y-range) are dropped, each strip's log-intensity is shifted so
    neighbours agree in their overlap (chained, mean-zero), and overlaps are cross-faded. crop() returns log(I+1) float32."""

    def __init__(self, zarr_ids, equalize=True, cache_dir=None):
        base = "https://dandiarchive.s3.amazonaws.com/zarr/"
        members = [StreamedZarr(base + i, cache_dir=cache_dir) for i in zarr_ids]
        ids = list(zarr_ids)
        top = len(members[0].levels) - 1
        yr = lambda m: (m.levels[top].tr[1], m.levels[top].tr[1] + m.levels[top].shape[-3 + 1] * m.levels[top].spacing_um[1])
        keep = []
        self.dropped = []
        for i, m in enumerate(members):
            a0, a1 = yr(m)
            dup = any(min(a1, yr(members[k])[1]) - max(a0, yr(members[k])[0]) > 0.5 * min(a1 - a0, yr(members[k])[1] - yr(members[k])[0]) for k in keep)
            (self.dropped if dup else keep).append(i)
        self.dropped = [ids[i] for i in self.dropped]
        keep.sort(key=lambda i: members[i].levels[top].tr[1])
        self.members = [members[i] for i in keep]
        self.ids = [ids[i] for i in keep]
        nl = min(len(m.levels) for m in self.members)
        self.levels = []
        for l in range(nl):
            sp = self.members[0].levels[l].spacing_um
            trs = np.array([m.levels[l].tr for m in self.members]); shp = np.array([m.levels[l].shape[-3:] for m in self.members])
            org = trs.min(0); end = (trs + shp * np.array(sp)).max(0)
            offs = np.round((trs - org) / np.array(sp)).astype(int)
            self.levels.append(Level("slab", tuple(int(np.ceil(v)) for v in (end - org) / np.array(sp)), tuple(sp), tuple(org)))
            self.levels[-1].offs = offs
        self.top = nl - 1
        self.seam = {}
        if equalize and len(self.members) > 1:
            self._equalize()

    def _member_crop(self, i, level, box):
        L = self.members[i].levels[level]; sh = L.shape[-3:]
        b = [max(0, box[0] - self.levels[level].offs[i][0]), min(sh[0], box[1] - self.levels[level].offs[i][0]), max(0, box[2] - self.levels[level].offs[i][1]), min(sh[1], box[3] - self.levels[level].offs[i][1]),
             max(0, box[4] - self.levels[level].offs[i][2]), min(sh[2], box[5] - self.levels[level].offs[i][2])]
        if b[1] <= b[0] or b[3] <= b[2] or b[5] <= b[4]:
            return None, b
        return np.log(self.members[i].crop(level, (b[0], b[1]), (b[2], b[3]), (b[4], b[5])).astype(np.float32) + 1.0), b

    def _equalize(self):
        """Local seam correction. For every neighbouring pair (A above B) estimate the smooth log-intensity difference field
        D(y, x) = log B - log A inside their overlap (z-averaged, tissue voxels only, normalised Gaussian smoothing) at a
        coarse level. crop() later removes D from B inside the overlap with a taper that is 1 where the overlap starts and
        0 where it ends, so strip interiors are never touched and nothing can accumulate along the chain."""
        from scipy.ndimage import gaussian_filter
        l = self.top; L = self.levels[l]; self.seam = {}
        for i in range(1, len(self.members)):
            A, B = i - 1, i; shA, shB = self.members[A].levels[l].shape[-3:], self.members[B].levels[l].shape[-3:]
            oA, oB = L.offs[A], L.offs[B]
            ya0, ya1 = oB[1], oA[1] + shA[1]
            x0, x1 = max(oA[2], oB[2]), min(oA[2] + shA[2], oB[2] + shB[2])
            z0, z1 = max(oA[0], oB[0]), min(oA[0] + shA[0], oB[0] + shB[0])
            if ya1 - ya0 < 2 or x1 - x0 < 4 or z1 - z0 < 1:
                continue
            zc = (z0 + z1) // 2; z0, z1 = max(z0, zc - 16), min(z1, zc + 17)
            box = (z0, z1, ya0, ya1, x0, x1)
            a, ba = self._member_crop(A, l, box); b, bb = self._member_crop(B, l, box)
            if a is None or b is None:
                continue
            def place(v, bx, k):
                out = np.full((z1 - z0, ya1 - ya0, x1 - x0), np.nan, np.float32); o = L.offs[k]
                out[bx[0] + o[0] - z0: bx[1] + o[0] - z0, bx[2] + o[1] - ya0: bx[3] + o[1] - ya0, bx[4] + o[2] - x0: bx[5] + o[2] - x0] = v; return out
            Aa, Bb = place(a, ba, A), place(b, bb, B)
            m = (np.isfinite(Aa) & np.isfinite(Bb) & (Aa > np.log(51)) & (Bb > np.log(51))).astype(np.float32)
            d = np.where(m > 0, np.nan_to_num(Bb - Aa), 0.0) * m
            dm, mm = d.sum(0), m.sum(0)                        # z-sum of the difference and of the tissue mask
            sg = (max(1.0, (ya1 - ya0) / 6), 12.0)
            num, den = gaussian_filter(dm, sg, mode="nearest"), gaussian_filter(mm, sg, mode="nearest")
            D = np.where(den > 2.0, num / np.maximum(den, 1e-6), 0.0).astype(np.float32)
            self.seam[B] = dict(lvl=l, y0=ya0, y1=ya1, x0=x0, D=D)

    def _seam_correction(self, i, level, b, shape):
        """Smooth difference field to subtract from member i (the lower strip of a seam) inside its top overlap."""
        sm = getattr(self, "seam", {}).get(i)
        if sm is None:
            return 0.0
        from scipy.ndimage import map_coordinates
        o = self.levels[level].offs[i]; f = 2.0 ** (sm["lvl"] - level)               # level-`level` voxel -> seam-level voxel
        gy = (np.arange(b[2], b[3]) + o[1] + 0.5) / f - sm["y0"]; gx = (np.arange(b[4], b[5]) + o[2] + 0.5) / f - sm["x0"]
        rows, cols = sm["D"].shape
        tt = np.clip(gy / max(rows, 1), 0.0, 1.0); taper = 1.0 - tt * tt * (3 - 2 * tt)
        inside = ((gy >= 0) & (gy < rows))[:, None] & ((gx >= 0) & (gx < cols))[None, :]
        yy, xx = np.meshgrid(np.clip(gy - 0.5, 0, rows - 1), np.clip(gx - 0.5, 0, cols - 1), indexing="ij")
        D = map_coordinates(sm["D"], [yy, xx], order=1, mode="nearest")
        corr = (D * taper[:, None] * inside).astype(np.float32)
        return corr[None, :, :]

    def crop(self, level, z, y, x):
        z0, z1 = z; y0, y1 = y; x0, x1 = x
        out = np.zeros((z1 - z0, y1 - y0, x1 - x0), np.float32); wsum = np.zeros_like(out)
        feather = max(2, round(780 / self.levels[level].spacing_um[1]))
        for i, m in enumerate(self.members):
            v, b = self._member_crop(i, level, (z0, z1, y0, y1, x0, x1))
            if v is None:
                continue
            o = self.levels[level].offs[i]; sh = m.levels[level].shape[-3:]
            ly = np.arange(b[2], b[3]); t = np.clip(np.minimum(ly + 1, sh[1] - ly) / feather, 1e-3, 1.0); w = (t * t * (3 - 2 * t))[None, :, None].astype(np.float32) + 1e-3
            sl = (slice(b[0] + o[0] - z0, b[1] + o[0] - z0), slice(b[2] + o[1] - y0, b[3] + o[1] - y0), slice(b[4] + o[2] - x0, b[5] + o[2] - x0))
            v = v - self._seam_correction(i, level, b, v.shape)
            out[sl] += w * v; wsum[sl] += w
        return np.where(wsum > 0, out / np.maximum(wsum, 1e-9), 0.0).astype(np.float32)
