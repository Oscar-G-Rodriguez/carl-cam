"""Stage offline help/notices and audited runtimes for same-machine checks only.

The system runtime option never grants redistribution permission. It retains
publisher/hash evidence and labels this copy private; the public licensing and
clean-machine gates remain open. No system file or user dataset is modified.
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
from xml.etree import ElementTree

import PySide6
from collect_notices import collect

RUNTIMES = ('msvcp140.dll', 'vcomp140.dll', 'msvcp140_atomic_wait.dll',
            'vcruntime140_threads.dll')


def checksum(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def copy_exact(source, target):
    """Preserve a conflicting existing file rather than silently replacing it."""
    if any(path.is_symlink() or path.is_junction() for path in (target, *target.parents)):
        raise ValueError('Staging paths must not use links or junctions')
    if not source.is_file() or source.is_symlink():
        raise ValueError('Required staging source is missing or linked')
    digest = checksum(source)
    if target.exists():
        if not target.is_file() or checksum(target) != digest:
            raise ValueError(f'Existing staged file differs: {target.name}')
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    if checksum(target) != digest:
        raise ValueError('Staged copy does not match its original bytes')
    return digest


def compiler_runtime_sources(report_path, bundle, redist):
    """Retain exact compiler-copied CRT bytes from the reviewed MSVC root.

    Signatures are checked separately by the Windows audit. A report entry
    alone never authorizes replacing a conflicting file or redistributing it.
    """
    if not report_path.is_file():
        return {}
    sources = {}
    for node in ElementTree.parse(report_path).getroot().iter('included_dll'):
        name = node.get('name')
        if name not in RUNTIMES or node.get('dest_path') != name or node.get('ignored') != 'no':
            continue
        source = Path(node.get('source_path', ''))
        target = bundle / name
        if (not source.is_file() or not source.resolve().is_relative_to(redist.resolve())
                or source.name != name or source.parent.name.lower() != 'microsoft.vc143.crt'
                or source.parent.parent.name.lower() != 'x64'):
            continue
        if any(path.is_symlink() or path.is_junction() for path in (source, *source.parents)):
            raise ValueError('Compiler runtime source must not use links')
        if not target.is_file() or checksum(source) != checksum(target):
            continue
        if name in sources:
            raise ValueError('Compiler runtime source is ambiguous')
        sources[name] = str(source)
    return sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-system-runtime-for-private-check', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    bundle = root / 'deployment/Carl-Cam.dist'
    if not (bundle / 'Carl-Cam.exe').is_file():
        raise ValueError('Finish actual native compilation before staging a bundle')
    copy_exact(root / 'docs/Quick Start.html', bundle / 'help/Quick Start.html')
    licenses = bundle / 'licenses'
    if any(path.is_symlink() or path.is_junction() for path in (licenses, *licenses.parents)):
        raise ValueError('License destination must not use links or junctions')
    collect(licenses)
    if args.include_system_runtime_for_private_check:
        environment = dict(os.environ)
        environment.pop('CARLCAM_QT_RUNTIME_SOURCE', None)
        compiler_report = root / 'build/nuitka-report.xml'
        redist = Path(os.environ['ProgramFiles(x86)']) / 'Microsoft Visual Studio/2022/BuildTools/VC/Redist/MSVC'
        compiler_sources = compiler_runtime_sources(compiler_report, bundle, redist)
        environment['CARLCAM_COMPILER_RUNTIME_SOURCES'] = json.dumps(compiler_sources)
        qt_runtime = Path(PySide6.__file__).parent / 'msvcp140.dll'
        existing = bundle / 'msvcp140.dll'
        # Nuitka may already retain Qt's own publisher-signed runtime. Keep
        # that exact wheel byte sequence rather than overwrite it with a newer
        # system copy. Other conflicting files still stop staging.
        if existing.is_file() and qt_runtime.is_file() and checksum(existing) == checksum(qt_runtime):
            environment['CARLCAM_QT_RUNTIME_SOURCE'] = str(qt_runtime)
        # The read-only probe resolves Windows' actual system directory and
        # validates Microsoft signatures. No computed deletion/move is used.
        script = """
# A Python child of PowerShell 7 can inherit its module search path. Load the
# Windows PowerShell security module explicitly rather than hiding an import
# error behind a successful final ConvertTo-Json command.
Import-Module -Name (Join-Path $PSHOME 'Modules\\Microsoft.PowerShell.Security\\Microsoft.PowerShell.Security.psd1') -ErrorAction Stop
$carlRuntimeNames = @('msvcp140.dll','vcomp140.dll','msvcp140_atomic_wait.dll','vcruntime140_threads.dll')
$carlSystemPath = [Environment]::GetFolderPath('System')
$carlCompilerSources = $env:CARLCAM_COMPILER_RUNTIME_SOURCES | ConvertFrom-Json
$carlRuntimeEvidence = foreach ($carlRuntimeName in $carlRuntimeNames) {
    $carlRuntimePath = Join-Path $carlSystemPath $carlRuntimeName
    if ($carlRuntimeName -eq 'msvcp140.dll' -and $env:CARLCAM_QT_RUNTIME_SOURCE) {
        $carlRuntimePath = $env:CARLCAM_QT_RUNTIME_SOURCE
    } elseif ($carlCompilerSources.PSObject.Properties[$carlRuntimeName]) {
        $carlRuntimePath = $carlCompilerSources.PSObject.Properties[$carlRuntimeName].Value
    }
    $carlSignature = Get-AuthenticodeSignature -LiteralPath $carlRuntimePath -ErrorAction Stop
    $carlInfo = Get-Item -LiteralPath $carlRuntimePath -ErrorAction Stop
    [pscustomobject]@{name=$carlRuntimeName; path=$carlRuntimePath; status=[string]$carlSignature.Status;
        signer=$carlSignature.SignerCertificate.Subject; version=$carlInfo.VersionInfo.FileVersion}
}
ConvertTo-Json -InputObject @($carlRuntimeEvidence) -Compress
"""
        result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script],
                                capture_output=True, text=True, check=True, timeout=60, env=environment)
        records = json.loads(result.stdout)
        print(json.dumps({'runtime_signature_audit': records}), flush=True)
        if not isinstance(records, list) or {item['name'] for item in records} != set(RUNTIMES):
            raise ValueError('Runtime audit has incompatible evidence')
        for item in records:
            source = Path(item['path'])
            from_qt = source == qt_runtime and item['name'] == 'msvcp140.dll'
            from_compiler = compiler_sources.get(item['name']) == str(source)
            publisher = 'O=The QT Company Oy' if from_qt else 'O=Microsoft Corporation'
            if item['status'] != 'Valid' or publisher not in (item['signer'] or ''):
                raise ValueError('Runtime publisher signature could not be verified')
            if source.name != item['name'] or not (from_qt or from_compiler or source.parent.name.lower() == 'system32'):
                raise ValueError('Runtime probe did not resolve an exact system DLL')
            item['origin_kind'] = ('verified PySide6 wheel bytes' if from_qt else
                                   'compiler-copied MSVC redistributable bytes' if from_compiler else 'developer System32')
            item['sha256'] = copy_exact(source, bundle / item['name'])
            item['bytes'] = source.stat().st_size
        report = {'schema_version': 1, 'captured_utc': datetime.now(UTC).isoformat(),
                  'purpose': 'private same-machine native loading check only',
                  'redistribution_authority': 'not verified', 'clean_machine_verified': False,
                  'public_release_approved': False, 'files': records,
                  'compiler_report_sha256': checksum(compiler_report) if compiler_sources else None}
        (licenses / 'private-system-runtime-provenance.json').write_text(
            json.dumps(report, indent=2), encoding='utf-8')
        # Nuitka recreates its distribution directory on the next build. Keep
        # this exact audit outside it too, rather than losing runtime origin
        # evidence during an otherwise safe retained-cache rebuild.
        evidence = root / 'build' / ('private-runtime-audit-' + uuid4().hex)
        copy_exact(licenses / 'private-system-runtime-provenance.json', evidence / 'provenance.json')
        print(f'Exact private runtime provenance retained at {evidence}')
    print('Offline help and wheel notices staged; public licensing remains unapproved.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
