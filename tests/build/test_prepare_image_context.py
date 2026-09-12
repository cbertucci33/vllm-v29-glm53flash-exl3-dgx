from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_module():
    root = Path(__file__).parents[2]
    path = root / "build" / "prepare_image_context.py"
    spec = importlib.util.spec_from_file_location("prepare_image_context", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verify_manifest_normalizes_sha256sum_relative_path(tmp_path: Path) -> None:
    module = _load_module()
    artifact = tmp_path / "artifact.whl"
    artifact.write_bytes(b"artifact")
    digest = module.sha256(artifact)
    (tmp_path / "artifact-sha256.txt").write_text(
        f"{digest}  ./artifact.whl\n", encoding="utf-8"
    )

    assert module.verify_manifest(tmp_path, "artifact-sha256.txt") == {
        "artifact.whl": digest
    }


def test_verify_manifest_rejects_parent_path(tmp_path: Path) -> None:
    module = _load_module()
    outside = tmp_path.parent / "outside.whl"
    outside.write_bytes(b"outside")
    digest = module.sha256(outside)
    (tmp_path / "artifact-sha256.txt").write_text(
        f"{digest}  ../outside.whl\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="unsafe or missing"):
        module.verify_manifest(tmp_path, "artifact-sha256.txt")
