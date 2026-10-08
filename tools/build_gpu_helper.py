"""Compile an isolated GPU helper with the same pinned Nuitka freezer.

The GPU interpreter supplies its own NumPy/ORT ABI. No driver, global toolkit,
app environment, existing installer, or original model is changed here.
Staging original runtime inventories and native verification are separate gates.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_inventory(root):
    paths = [root / 'Carl-Cam-GPU.py', root / 'tools/build_gpu_helper.py']
    paths.extend(sorted((root / 'src/carl_cam').rglob('*.py')))
    result = []
    for path in paths:
        if '__pycache__' in path.parts:
            continue
        if (not path.is_file() or any(part.is_symlink() or part.is_junction()
                                    for part in (path, *path.parents))):
            raise ValueError('GPU build inputs must be unlinked files')
        result.append({'path': path.relative_to(root).as_posix(),
                       'sha256': digest(path), 'bytes': path.stat().st_size})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    interpreter = root / 'build/gpu-conversion-env/Scripts/python.exe'
    from carl_cam.gpu_process import DEVELOPMENT_PYTHON_SHA256
    if digest(interpreter) != DEVELOPMENT_PYTHON_SHA256:
        raise ValueError('GPU build interpreter identity differs')
    output = root / 'deployment/gpu-helper'
    if any(part.is_symlink() or part.is_junction() for part in (output, *output.parents)):
        raise ValueError('GPU build output must not use links')
    command = [str(interpreter), '-m', 'nuitka', '--standalone', '--follow-imports',
               '--windows-console-mode=force', '--output-dir=' + str(output),
               '--output-filename=Carl-Cam-GPU.exe', '--include-module=carl_cam.gpu_service',
               '--include-module=carl_cam.gpu_regions',
               '--nofollow-import-to=mediapipe,tensorflow,torch,PySide6,matplotlib,scipy,onnx',
               '--assume-yes-for-downloads', '--mingw64', '--jobs=2', '--low-memory',
               '--report=' + str(root / 'build/gpu-nuitka-report.xml'),
               str(root / 'Carl-Cam-GPU.py')]
    if args.inspect_only:
        print(json.dumps({'command': command, 'separate_gpu_interpreter': True,
                          'app_environment_unchanged': True}))
        return 0
    compiler_version = subprocess.run(
        [str(interpreter), '-I', '-c',
         "import importlib.metadata as m; print(m.version('Nuitka'))"],
        check=True, capture_output=True, text=True).stdout.strip()
    if compiler_version != '4.1.1':
        raise ValueError('The GPU helper requires the pinned Carl-Cam freezer')
    evidence = root / 'build' / ('gpu-native-input-' + uuid4().hex)
    evidence.mkdir(exist_ok=False)
    inventory = source_inventory(root)
    for item in inventory:
        target = evidence / 'snapshot' / item['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / item['path'], target)
    report = {'schema_version': 1, 'started_utc': datetime.now(UTC).isoformat(),
              'compiler_version': compiler_version, 'interpreter_sha256': digest(interpreter),
              'command': command, 'inputs': inventory, 'compiler_returncode': None,
              'native_loading_verified': False, 'installed_app_verified': False}
    record = evidence / 'inputs.json'
    record.write_text(json.dumps(report, indent=2), encoding='utf-8')
    environment = dict(os.environ)
    environment['NUITKA_CACHE_DIR'] = str(root / 'build/nuitka-cache')
    environment['PYTHONPATH'] = str(root / 'src')
    print('GPU helper build evidence: ' + str(evidence), flush=True)
    with (evidence / 'compiler.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=root, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
        report['compiler_returncode'] = process.wait()
    report['finished_utc'] = datetime.now(UTC).isoformat()
    report['source_matches_initial_inventory'] = source_inventory(root) == inventory
    executable = output / 'Carl-Cam-GPU.dist/Carl-Cam-GPU.exe'
    if report['compiler_returncode'] == 0 and executable.is_file():
        report['executable_sha256'] = digest(executable)
    record.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if report['compiler_returncode'] != 0:
        return report['compiler_returncode']
    if not report['source_matches_initial_inventory'] or not executable.is_file():
        raise ValueError('GPU helper output is incomplete or source changed during compilation')
    print('GPU helper compiled. Stage exact runtime inventories and verify native execution next.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
