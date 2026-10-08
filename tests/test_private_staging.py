"""Private packaging must preserve exact bytes and existing conflicting files."""

import importlib.util
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'


@pytest.fixture
def staging_tool(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location('carlcam_private_staging', TOOLS / 'stage_private_bundle.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop('collect_notices', None)


def test_private_staging_copies_exact_bytes_without_overwriting_conflicts(staging_tool, tmp_path):
    source = tmp_path / 'source.dll'
    source.write_bytes(b'fixture bytes; not a DLL')
    target = tmp_path / 'bundle/runtime.dll'
    first = staging_tool.copy_exact(source, target)
    assert target.read_bytes() == source.read_bytes() and len(first) == 64
    assert staging_tool.copy_exact(source, target) == first
    source.write_bytes(b'different fixture bytes')
    with pytest.raises(ValueError, match='Existing staged file differs'):
        staging_tool.copy_exact(source, target)
    assert target.read_bytes() == b'fixture bytes; not a DLL'


@pytest.mark.parametrize('scenario', ['exact', 'changed', 'outside', 'wrong_arch', 'wrong_destination', 'ignored', 'duplicate'])
def test_compiler_runtime_lookup_retains_only_exact_reviewed_crt_sources(staging_tool, tmp_path, scenario):
    redist = tmp_path / 'MSVC'
    directory = (tmp_path / 'unreviewed' if scenario == 'outside' else redist / '14.44')
    source = directory / ('arm64' if scenario == 'wrong_arch' else 'x64') / 'microsoft.vc143.crt/msvcp140_atomic_wait.dll'
    source.parent.mkdir(parents=True)
    source.write_bytes(b'fixture CRT bytes; not an executable library')
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    target = bundle / source.name
    target.write_bytes(b'conflicting bytes' if scenario == 'changed' else source.read_bytes())
    report = tmp_path / 'compiler.xml'
    tree = staging_tool.ElementTree.Element('nuitka-compilation-report')
    attributes = {'name': source.name, 'source_path': str(source),
                  'dest_path': 'other.dll' if scenario == 'wrong_destination' else source.name,
                  'ignored': 'yes' if scenario == 'ignored' else 'no'}
    staging_tool.ElementTree.SubElement(tree, 'included_dll', attributes)
    if scenario == 'duplicate':
        staging_tool.ElementTree.SubElement(tree, 'included_dll', attributes)
    staging_tool.ElementTree.ElementTree(tree).write(report, encoding='utf-8')
    before = target.read_bytes()
    if scenario == 'duplicate':
        with pytest.raises(ValueError, match='ambiguous'):
            staging_tool.compiler_runtime_sources(report, bundle, redist)
    else:
        expected = {source.name: str(source)} if scenario == 'exact' else {}
        assert staging_tool.compiler_runtime_sources(report, bundle, redist) == expected
    assert target.read_bytes() == before


def test_missing_compiler_report_does_not_invent_a_runtime_origin(staging_tool, tmp_path):
    assert staging_tool.compiler_runtime_sources(tmp_path / 'absent.xml', tmp_path / 'bundle', tmp_path / 'MSVC') == {}
