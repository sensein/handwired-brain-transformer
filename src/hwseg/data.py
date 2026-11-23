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
            self.levels.append(Level(ds["path"], tuple(self.grp[ds["path"]].shape), tuple(sc[-3:])))

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
