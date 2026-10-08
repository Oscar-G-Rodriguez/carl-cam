"""Compiler success must not certify an executable made from changing sources."""

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    'carlcam_native_builder', Path(__file__).resolve().parents[1] / 'tools/build_native.py')
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def test_source_inventory_detects_changes_additions_and_deletions(tmp_path):
    source = tmp_path / 'src/carl_cam'
    source.mkdir(parents=True)
    code = source / 'app.py'
    code.write_text('original source')
    asset = source / 'image.svg'
    asset.write_text('original asset')
    cache = source / '__pycache__'
    cache.mkdir()
    (cache / 'app.pyc').write_bytes(b'not a build input')
    before = builder.input_inventory(tmp_path)
    assert len(before) == 2
    assert builder.changed_inputs(before, builder.input_inventory(tmp_path)) == []
    code.write_text('changed source')
    asset.unlink()
    (source / 'new.task').write_bytes(b'new model')
    assert builder.changed_inputs(before, builder.input_inventory(tmp_path)) == [
        'src/carl_cam/app.py', 'src/carl_cam/image.svg', 'src/carl_cam/new.task']
