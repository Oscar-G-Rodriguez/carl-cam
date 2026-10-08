"""Static candidate evidence must never be mistaken for runtime approval."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def inspector(monkeypatch):
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("candidate_inspector", tools / "inspect_vision_candidate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exports_require_a_real_table(inspector):
    assert inspector.parse_exports(" 1 0 000ABC MpImageFree\n 2 A 123DEF MpErrorFree") == {
        "MpImageFree", "MpErrorFree"}
    with pytest.raises(ValueError, match="export table"):
        inspector.parse_exports("No exports found")


def test_imports_cover_api_sets_and_delay_names_without_paths(inspector):
    text = "  Image dependencies:\n    opencv_core3411.dll\n    KERNEL32.dll\n" + (
        "  Delay load dependencies:\n    dbghelp.dll\n    api-ms-win-crt-heap-l1-1-0.dll\n"
        "Dump of file C:/fake/not-an-import.dll\n")
    assert inspector.parse_dependencies(text) == [
        "api-ms-win-crt-heap-l1-1-0.dll", "dbghelp.dll", "KERNEL32.dll", "opencv_core3411.dll"]


def test_signature_collection_never_executes_wrapper(inspector):
    text = "raise RuntimeError('must not run')\n_CFunction('MpImageFree', [], None)\n" + (
        "utils.CStatusFunction('MpHandLandmarkerCreate', ())\n"
        "CFunction(func_name='MpErrorFree', argtypes=[], restype=None)\n")
    assert inspector.wrapper_symbols(text) == {
        "MpImageFree", "MpHandLandmarkerCreate", "MpErrorFree"}
    with pytest.raises(TypeError, match="literal"):
        inspector.wrapper_symbols("CFunction(dynamic(), [], None)")


@pytest.mark.parametrize("change", [
    {"compiler_returncode": None}, {"compiler_returncode": False}, {"compiler_returncode": 1},
    {"inputs_unchanged": False}, {"action": "query"}, {"source_commit": "wrong"},
    {"finished_utc": None}])
def test_nonterminal_or_unverified_build_rejected(inspector, change):
    record = {"action": "build", "compiler_returncode": 0, "inputs_unchanged": True,
              "finished_utc": "2026-10-02T13:28:39Z", "source_commit": inspector.SOURCE_COMMIT}
    record.update(change)
    with pytest.raises(ValueError, match="successful unchanged"):
        inspector.require_terminal_attempt(record)


@pytest.fixture
def static_fixture(inspector, tmp_path, monkeypatch):
    source = tmp_path / "build/vendor/mediapipe-f8ef212d"
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    (attempt / "source.diff").write_text("fixture patch")
    installed = tmp_path / ".venv/Lib/site-packages/mediapipe/tasks/python"
    for folder in (source / "mediapipe/tasks/python", installed):
        for name in inspector.WRAPPERS:
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"_CFunction('MpImageFree', [], None)\n")
    record = {"action": "build", "compiler_returncode": 0, "inputs_unchanged": True,
              "finished_utc": "2026-10-02T13:28:39Z", "source_commit": inspector.SOURCE_COMMIT,
              "input_hashes": {}}
    (attempt / "attempt.json").write_text(json.dumps(record))
    candidate = tmp_path / "binary/mediapipe_source.dll"
    candidate.parent.mkdir()
    candidate.write_bytes(b"fixture candidate")
    opencv, system = tmp_path / "opencv", tmp_path / "system"
    opencv.mkdir()
    system.mkdir()
    (opencv / "opencv_core3411.dll").write_bytes(b"fixture build dependency")
    (system / "KERNEL32.dll").write_bytes(b"fixture system dependency")
    tool = tmp_path / "dumpbin.exe"
    tool.write_bytes(b"fixture static reader")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    monkeypatch.setattr(inspector.subprocess, "check_output", lambda cmd, **kw:
                        inspector.SOURCE_COMMIT if cmd[1] == "rev-parse" else "fixture patch")

    def fake_run(cmd, **kwargs):
        if cmd[2] == "/exports":
            output = " 1 0 000ABC MpImageFree\n 2 1 000DEF MpErrorFree\n"
        elif Path(cmd[3]).name == "mediapipe_source.dll":
            output = "    opencv_core3411.dll\n    api-ms-win-crt-heap-l1-1-0.dll\n"
        else:
            output = "    KERNEL32.dll\n"
        return SimpleNamespace(stdout=output, stderr="", returncode=0)

    monkeypatch.setattr(inspector.subprocess, "run", fake_run)
    return (tmp_path, attempt, candidate, tool, opencv, system, evidence)


def test_static_success_cannot_approve_or_load_runtime(inspector, static_fixture):
    report = inspector.inspect(*static_fixture)
    assert report["static_export_check_passed"] and report["inspection_inputs_unchanged"]
    assert len(report["libraries"]) == 2
    assert all(report[name] is False for name in (
        "runtime_loaded", "abi_verified", "privacy_verified", "numerical_verified", "promoted"))
    assert report["libraries"]["mediapipe_source.dll"]["imports"][0]["kind"] == "windows_api_set"
    assert (static_fixture[-1] / "manifest.json").is_file()


def test_missing_dependency_blocks_inventory(inspector, static_fixture):
    (static_fixture[4] / "opencv_core3411.dll").unlink()
    with pytest.raises(ValueError, match="Unresolved import"):
        inspector.inspect(*static_fixture)


def test_only_newlines_may_differ_in_installed_wrappers(inspector, static_fixture):
    root = static_fixture[0]
    installed = root / ".venv/Lib/site-packages/mediapipe/tasks/python" / inspector.WRAPPERS[0]
    installed.write_bytes(b"_CFunction('MpImageFree', [], None)\r\n")
    report = inspector.inspect(*static_fixture)
    pair = report["wrappers"][inspector.WRAPPERS[0]]
    assert pair["installed"]["sha256"] != pair["source"]["sha256"]
    assert pair["text_matches_after_newline_normalization"]
    installed.write_text("_CFunction('MpDifferentFunction', [], None)\n")
    with pytest.raises(ValueError, match="wrapper differ"):
        inspector.inspect(*static_fixture)


def test_missing_export_is_retained_as_failed_evidence(inspector, static_fixture, monkeypatch):
    real_parse = inspector.parse_exports
    monkeypatch.setattr(inspector, "parse_exports", lambda text: real_parse(text) - {"MpErrorFree"})
    with pytest.raises(ValueError, match="static requirements"):
        inspector.inspect(*static_fixture)
    report = json.loads((static_fixture[-1] / "manifest.json").read_text())
    assert report["missing_exports"] == ["MpErrorFree"] and not report["static_export_check_passed"]


def test_changed_dependency_cannot_produce_passing_report(inspector, static_fixture, monkeypatch):
    previous = inspector.subprocess.run

    def change_during_exports(cmd, **kwargs):
        result = previous(cmd, **kwargs)
        if cmd[2] == "/exports":
            (static_fixture[4] / "opencv_core3411.dll").write_bytes(b"changed during inspection")
        return result

    monkeypatch.setattr(inspector.subprocess, "run", change_during_exports)
    with pytest.raises(ValueError, match="static requirements"):
        inspector.inspect(*static_fixture)
    report = json.loads((static_fixture[-1] / "manifest.json").read_text())
    assert not report["inspection_inputs_unchanged"] and not report["privacy_verified"]


def test_static_reader_failure_cannot_become_success(inspector, static_fixture, monkeypatch):
    monkeypatch.setattr(inspector.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(stdout="", stderr="fixture reader failed", returncode=1))
    with pytest.raises(ValueError, match="Static inspection failed"):
        inspector.inspect(*static_fixture)
    assert not (static_fixture[-1] / "manifest.json").exists()


def test_both_marker_encodings_are_evidence_only(inspector, tmp_path):
    path = tmp_path / "candidate.dll"
    path.write_bytes(b"portable_clearcut_uploader" + "play.googleapis.com".encode("utf-16le"))
    markers = inspector.marker_scan(path)
    assert markers["portable_clearcut_uploader"]["ascii"]
    assert markers["play.googleapis.com"]["utf16le"]
