"""Neurodesk reference models (demo/nd/): weights are fetched at run time from pinned, checksummed URLs, never vendored."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = (ROOT / "demo/nd/nd-seg.js").read_text()


def test_no_network_weights_in_the_repository():
    tracked = [p for p in ROOT.rglob("*") if p.suffix in (".onnx", ".pt", ".pth", ".h5", ".safetensors") and ".venv" not in p.parts and "build" not in p.parts]
    assert not tracked, f"model weights must not be vendored: {tracked}"


def test_weights_urls_are_pinned_and_checksummed():
    urls = re.findall(r"url: HF \+ '([^']+)'", SRC)
    assert len(urls) == 3, urls                                   # Cellpose, StarDist, VesselBoost (cellfinder has no weights)
    for u in urls:
        assert re.search(r"/resolve/[0-9a-f]{40}/", u), f"not pinned to a revision: {u}"
    sums = re.findall(r"sha256: '([0-9a-f]{64})'", SRC)
    assert len(sums) == 3, sums
    assert re.search(r"ORT_VERSION = '\d+\.\d+\.\d+'", SRC), "onnxruntime-web must be a pinned version"
    assert "clipper-lib@6.4.2" in SRC


def test_reference_models_are_selectable_and_licences_noted():
    hook = (ROOT / "demo/ng-hook.js").read_text(); shell = (ROOT / "demo/shell.js").read_text()
    for k in ("nd_cellpose", "nd_stardist", "nd_cellfinder", "nd_vesselboost"):
        assert f'value="{k}"' in hook and f"key: '{k}'" in shell and f"{k}: {{" in SRC
    for f in ("LICENSE-cellpose-port", "NOTICE-cellpose", "LICENSE-stardist-port", "NOTICE-stardist", "LICENSE-cellfinder"):
        assert (ROOT / "demo/nd" / f).exists()
    assert "CC-BY-NC" in SRC and "NOASSERTION" in SRC
