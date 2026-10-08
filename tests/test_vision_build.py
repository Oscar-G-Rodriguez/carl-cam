"""The replacement builder must retain the real source target and privacy hold."""

import importlib.util
import re
from pathlib import Path

import pytest


def load_builder():
    path = Path(__file__).resolve().parents[1] / "tools/build_vision_runtime.py"
    spec = importlib.util.spec_from_file_location("vision_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def builder_setup(tmp_path, monkeypatch):
    builder = load_builder()
    source = tmp_path / "build/vendor/mediapipe-f8ef212d"
    source.mkdir(parents=True)
    (source / ".bazelversion").write_text("7.4.1\n")
    monkeypatch.setattr(builder, "checksum", lambda path: builder.BAZEL_SHA256)
    monkeypatch.setattr(builder.subprocess, "check_output", lambda args, **kwargs:
                        builder.SOURCE_COMMIT if args[1] == "rev-parse" else "")
    monkeypatch.setattr(builder.Path, "exists", lambda path: True)
    return builder, tmp_path, source


def test_build_uses_real_source_not_prebuilt_alias(builder_setup):
    builder, root, _ = builder_setup
    source, command, env, inventory, patch = builder.prepare(root, root / "cache", "build")
    assert source == root / "build/vendor/mediapipe-f8ef212d"
    assert builder.SOURCE_TARGET in command
    assert "//mediapipe/tasks/c:mediapipe_windows" not in command
    assert "--define=BUILD_FOR_3P=1" in command
    assert "--define=MEDIAPIPE_DISABLE_GPU=1" in command
    assert "--define=OPENCV=source" in command
    assert "--jobs=4" in command
    assert "logging_factory.cc" in " ".join(inventory)
    assert "tasks_dummy_logger.h" in " ".join(inventory)
    assert "mediapipe/gpu/gpu_service.cc" in inventory
    assert "mediapipe/gpu/gpu_service.h" in inventory
    assert "mediapipe/framework/graph_service.h" in inventory
    assert "mediapipe/framework/legacy_calculator_support.cc" in inventory
    assert "mediapipe/framework/legacy_calculator_support.h" in inventory
    assert "mediapipe/framework/api3/calculator_context.h" in inventory
    assert "mediapipe/framework/api3/graph.h" in inventory
    assert "mediapipe/framework/api3/subgraph_context.h" in inventory
    assert env["HERMETIC_PYTHON_VERSION"] == "3.12"
    assert patch == ""


def test_query_retains_whole_dependency_graph(builder_setup):
    builder, root, _ = builder_setup
    _, command, _, _, _ = builder.prepare(root, root / "cache", "query")
    assert f"deps({builder.SOURCE_TARGET})" in command
    assert "cquery" in command
    assert "--define=BUILD_FOR_3P=1" in command
    assert "--output=label" in command


def test_short_output_base_is_startup_option_and_preserves_cache(builder_setup):
    builder, root, _ = builder_setup
    short_base = root / "short"
    for action in ("query", "build"):
        _, command, _, _, _ = builder.prepare(
            root, root / "cache", action, output_base=short_base)
        assert f"--output_user_root={root / 'cache'}" in command
        assert f"--output_base={short_base}" in command
        operation = "cquery" if action == "query" else "build"
        assert command.index(f"--output_base={short_base}") < command.index(operation)
        assert "clean" not in command


def test_relative_output_base_rejected(builder_setup):
    builder, root, _ = builder_setup
    with pytest.raises(ValueError, match="absolute ordinary directory"):
        builder.prepare(root, root / "cache", "build", output_base=Path("relative"))


def test_msvc_c11_options_are_scoped_to_threadpool_in_both_actions(builder_setup):
    builder, root, _ = builder_setup
    for action in ("query", "build"):
        command = builder.prepare(root, root / "cache", action)[1]
        assert "--per_file_copt=external/pthreadpool/.*@/std:c11,/experimental:c11atomics" in command
        assert not any(option.startswith(("--copt=", "--conlyopt=")) for option in command)


def test_conforming_preprocessor_covers_target_and_build_tools(builder_setup):
    builder, root, _ = builder_setup
    for action in ("query", "build"):
        command = builder.prepare(root, root / "cache", action)[1]
        assert "--cxxopt=/Zc:preprocessor" in command
        assert "--host_cxxopt=/Zc:preprocessor" in command


def test_wrong_tool_hash_stops_before_subprocess(builder_setup, monkeypatch):
    builder, root, _ = builder_setup
    monkeypatch.setattr(builder, "checksum", lambda path: "wrong")
    with pytest.raises(ValueError, match="identity"):
        builder.prepare(root, root / "cache", "build")


def test_wrong_source_commit_rejected(builder_setup, monkeypatch):
    builder, root, _ = builder_setup
    monkeypatch.setattr(builder.subprocess, "check_output", lambda *args, **kwargs: "wrong")
    with pytest.raises(ValueError, match="source version"):
        builder.prepare(root, root / "cache", "build")


def test_wrong_bazel_version_rejected(builder_setup):
    builder, root, source = builder_setup
    (source / ".bazelversion").write_text("8.0.0\n")
    with pytest.raises(ValueError, match="source version"):
        builder.prepare(root, root / "cache", "build")


def test_source_patch_preserved_in_provenance(builder_setup, monkeypatch):
    builder, root, _ = builder_setup
    patch = "diff --git a/WORKSPACE b/WORKSPACE\ncompatibility patch\n"
    monkeypatch.setattr(builder.subprocess, "check_output", lambda args, **kwargs:
                        builder.SOURCE_COMMIT if args[1] == "rev-parse" else patch)
    assert builder.prepare(root, root / "cache", "build")[-1] == patch


def test_opencv_build_recipe_and_hash_patch_retained(builder_setup):
    builder, root, source = builder_setup
    third_party = source / "third_party"
    third_party.mkdir()
    (third_party / "carl_cam_opencv_msvc_hash.diff").write_text("fixture patch\n")
    inventory = builder.prepare(root, root / "cache", "build")[3]
    assert "third_party/BUILD" in inventory
    assert "third_party/carl_cam_opencv_msvc_hash.diff" in inventory


def test_retained_compatibility_patch_has_complete_hunks():
    patch = Path(__file__).resolve().parents[1] / "packaging/mediapipe-windows-build.patch"
    lines = patch.read_text().splitlines()
    starts = [index for index, line in enumerate(lines) if line.startswith("@@ ")]
    assert starts
    for start in starts:
        header = re.fullmatch(r"@@ -\d+,(\d+) \+\d+,(\d+) @@.*", lines[start])
        assert header is not None
        old_count = new_count = 0
        for line in lines[start + 1:]:
            if line.startswith(("@@ ", "diff --git ")):
                break
            assert line and line[0] in " +-", "Malformed patch context"
            old_count += line[0] in " -"
            new_count += line[0] in " +"
        assert (old_count, new_count) == tuple(map(int, header.groups()))


def test_builder_cannot_promote_or_load_sdk():
    builder = load_builder()
    code = Path(builder.__file__).read_text()
    assert '"privacy_verified": False' in code
    assert '"abi_verified": False' in code
    assert '"promoted": False' in code
    assert "import mediapipe" not in code
    assert "APPROVED_VISION_RUNTIMES" not in code
