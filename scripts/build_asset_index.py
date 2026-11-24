"""Index the OME-Zarr assets of DANDI 000108 for the demo (metadata only, no image data)."""
import json, re, sys, urllib.request
out, url = [], "https://api.dandiarchive.org/api/dandisets/000108/versions/draft/assets/?page_size=1000&glob=*.ome.zarr"
while url:
    d = json.load(urllib.request.urlopen(url))
    for r in d["results"]:
        if not r["path"].endswith(".ome.zarr") or not r.get("zarr"):
            continue
        m = re.search(r"sub-(\w+?)/.*sample-(\w+?)_stain-(\w+?)_run-\d+_chunk-(\d+)", r["path"])
        if m:
            out.append(dict(zarr=r["zarr"], sub=m[1], sample=m[2], stain=m[3], chunk=int(m[4])))
    url = d["next"]
out.sort(key=lambda r: (r["stain"], r["sample"], r["chunk"]))
json.dump(out, open(sys.argv[1], "w"), separators=(",", ":"))
print(len(out), "zarrs", {s: sum(r["stain"] == s for r in out) for s in {r["stain"] for r in out}})
