from __future__ import annotations

import errno
import json
import os
from pathlib import Path

import pytest

from resin.kernel_cache import ArtifactCache


class TestArtifactCache:
    def test_same_size_corruption_is_a_cache_miss(self, tmp_path: Path) -> None:
        cache = ArtifactCache(tmp_path, "sm90-cuda-test")
        artifact = cache.root / "flashinfer" / "module.so"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"abc")

        record = cache.publish(artifact, identity="flashinfer:module")
        assert record.schema == 2
        assert record.size == 3
        assert (
            record.blake3
            == "6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85"
        )
        assert cache.validate(artifact, identity="flashinfer:module")

        artifact.write_bytes(b"xyz")
        assert not cache.validate(artifact, identity="flashinfer:module")

    def test_missing_and_invalid_records_fall_back_to_build(
        self, tmp_path: Path
    ) -> None:
        cache = ArtifactCache(tmp_path, "test-abi")
        artifact = cache.root / "module.so"
        artifact.parent.mkdir(parents=True)
        calls: list[str] = []

        def build() -> str:
            calls.append("build")
            artifact.write_bytes(b"compiled")
            return "built"

        def load(path: Path) -> str:
            calls.append(f"load:{path.name}")
            return "loaded"

        assert (
            cache.load_or_build(artifact, identity="kernel", load=load, build=build)
            == "built"
        )
        assert (
            cache.load_or_build(artifact, identity="kernel", load=load, build=build)
            == "loaded"
        )

        cache.record_path(artifact).write_text("not json", encoding="utf-8")
        assert (
            cache.load_or_build(artifact, identity="kernel", load=load, build=build)
            == "built"
        )
        assert calls == ["build", "load:module.so", "build"]
        assert cache.hits == 1
        assert cache.misses == 2
        assert cache.validate(artifact, identity="kernel")

    def test_load_or_build_keeps_relative_artifact_behavior(
        self, tmp_path: Path
    ) -> None:
        cache = ArtifactCache(tmp_path, "test-abi")
        artifact = cache.root / "nested" / "module.so"
        artifact.parent.mkdir(parents=True)

        assert (
            cache.load_or_build(
                "nested/module.so",
                identity="kernel",
                load=lambda _path: "loaded",
                build=lambda: (artifact.write_bytes(b"compiled"), "built")[1],
            )
            == "built"
        )
        assert cache.validate(artifact, identity="kernel")

    def test_record_rejects_wrong_abi_identity_and_artifact_path(
        self, tmp_path: Path
    ) -> None:
        cache = ArtifactCache(tmp_path, "test-abi")
        artifact = cache.root / "module.so"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"compiled")
        cache.publish(artifact, identity="kernel")
        record_path = cache.record_path(artifact)
        original = json.loads(record_path.read_text(encoding="utf-8"))

        for key, value in (
            ("abi", "other-abi"),
            ("identity", "other-kernel"),
            ("artifact", "elsewhere/module.so"),
        ):
            changed = {**original, key: value}
            record_path.write_text(json.dumps(changed), encoding="utf-8")
            assert not cache.validate(artifact, identity="kernel")

    def test_atomic_publish_preserves_old_record_if_replace_fails(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cache = ArtifactCache(tmp_path, "test-abi")
        artifact = cache.root / "module.so"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"old")
        cache.publish(artifact, identity="kernel")
        record_path = cache.record_path(artifact)
        old_record = record_path.read_bytes()
        artifact.write_bytes(b"new")

        def fail_replace(*_args, **_kwargs):
            raise OSError("failed")

        monkeypatch.setattr("resin.kernel_cache.os.replace", fail_replace)
        with pytest.raises(OSError, match="failed"):
            cache.publish(artifact, identity="kernel")

        assert record_path.read_bytes() == old_record
        assert list(record_path.parent.glob(".*.tmp")) == []

    def test_unsupported_directory_fsync_does_not_turn_publish_into_a_miss(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cache = ArtifactCache(tmp_path, "test-abi")
        artifact = cache.root / "module.so"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"compiled")

        real_fsync = os.fsync
        fsync_calls = 0

        def reject_directory(descriptor: int) -> None:
            nonlocal fsync_calls
            fsync_calls += 1
            if fsync_calls == 2:
                raise OSError(errno.EINVAL, "directory fsync unsupported")
            real_fsync(descriptor)

        monkeypatch.setattr("resin.kernel_cache.os.fsync", reject_directory)
        cache.publish(artifact, identity="kernel")

        assert cache.validate(artifact, identity="kernel")

    def test_artifact_must_stay_inside_abi_namespace(self, tmp_path: Path) -> None:
        cache = ArtifactCache(tmp_path, "test-abi")
        outside = tmp_path / "outside.so"
        outside.write_bytes(b"compiled")
        with pytest.raises(ValueError, match="escaped"):
            cache.publish(outside, identity="kernel")

    def test_publish_from_copies_local_artifact_and_records_copied_bytes(
        self,
        tmp_path: Path,
    ) -> None:
        root = tmp_path
        cache = ArtifactCache(root / "durable", "test-abi")
        source = root / "scratch" / "module.so"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"locally compiled")

        record = cache.publish_from(
            source,
            "flashinfer/module.so",
            identity="flashinfer:module",
        )
        artifact = cache.root / "flashinfer" / "module.so"

        assert artifact.read_bytes() == b"locally compiled"
        assert record.artifact == "flashinfer/module.so"
        assert record.size == len(b"locally compiled")
        assert cache.validate(artifact, identity="flashinfer:module")
        assert list(artifact.parent.glob(".*.tmp")) == []

    def test_publish_from_copy_replace_failure_preserves_previous_cache(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        root = tmp_path
        cache = ArtifactCache(root / "durable", "test-abi")
        artifact = cache.root / "module.so"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"old compiled")
        cache.publish(artifact, identity="kernel")
        old_record = cache.record_path(artifact).read_bytes()
        source = root / "scratch.so"
        source.write_bytes(b"new compiled")

        def fail_replace(*_args, **_kwargs):
            raise OSError("failed")

        monkeypatch.setattr("resin.kernel_cache.os.replace", fail_replace)
        with pytest.raises(OSError, match="failed"):
            cache.publish_from(source, artifact, identity="kernel")

        assert artifact.read_bytes() == b"old compiled"
        assert cache.record_path(artifact).read_bytes() == old_record
        assert cache.validate(artifact, identity="kernel")
        assert list(artifact.parent.glob(".*.tmp")) == []

    def test_publish_from_same_path_does_not_copy_artifact(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cache = ArtifactCache(tmp_path, "test-abi")
        artifact = cache.root / "module.so"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"compiled")

        def unexpected_copy(*_args, **_kwargs):
            raise AssertionError("same-path publication must not copy")

        monkeypatch.setattr("resin.kernel_cache._atomic_copy_artifact", unexpected_copy)
        cache.publish_from(artifact, artifact, identity="kernel")
        assert cache.validate(artifact, identity="kernel")

    def test_load_or_build_from_propagates_failures_without_readiness(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path
        cache = ArtifactCache(root / "durable", "test-abi")
        artifact = cache.root / "module.so"
        source = root / "scratch.so"

        with pytest.raises(RuntimeError, match="compiler failed"):
            cache.load_or_build_from(
                artifact,
                source=source,
                identity="kernel",
                load=lambda _path: "loaded",
                build=lambda: (_ for _ in ()).throw(RuntimeError("compiler failed")),
            )

        assert not artifact.exists()
        assert not cache.record_path(artifact).exists()
        assert cache.misses == 1

        with pytest.raises(FileNotFoundError):
            cache.load_or_build_from(
                artifact,
                source=source,
                identity="kernel",
                load=lambda _path: "loaded",
                build=lambda: "built without output",
            )

        assert not artifact.exists()
        assert not cache.record_path(artifact).exists()
        assert cache.hits == 0
        assert cache.misses == 2
