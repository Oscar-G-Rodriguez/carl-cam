"""Retain installed dependency notices for the private bundle's audit.

This inventories exact wheel evidence; it does not choose Carl-Cam's license
or certify that a public redistribution satisfies each upstream obligation.
"""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path


def collect(destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    inventory = []
    for distribution in sorted(importlib.metadata.distributions(), key=lambda item: item.metadata['Name'].lower()):
        name = distribution.metadata['Name']
        if name.lower().replace('_', '-') in ('carl-cam', 'pytest', 'ruff', 'nuitka', 'pluggy', 'iniconfig', 'colorama', 'setuptools'):
            continue
        files = []
        for relative in distribution.files or ():
            normalized = str(relative).replace('\\', '/')
            filename = Path(normalized).name.lower()
            if '.dist-info/' not in normalized or not filename.startswith(('license', 'copying', 'notice', 'authors')):
                continue
            source = Path(distribution.locate_file(relative))
            if not source.is_file() or source.is_symlink():
                continue
            # Only exact notice bytes, never arbitrary dependency source/data.
            target = destination / name / Path(normalized.split('.dist-info/', 1)[1])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            files.append({'path': target.relative_to(destination).as_posix(),
                          'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
        metadata = distribution.metadata
        inventory.append({'name': name, 'version': distribution.version,
                          'declared_license': metadata.get('License-Expression') or metadata.get('License'),
                          'license_classifiers': [item for item in metadata.get_all('Classifier', []) if item.startswith('License ::')],
                          'project_urls': metadata.get_all('Project-URL', []), 'notices': files})
    report = {'schema_version': 1, 'public_release_approved': False,
              'wheel_inventory_not_compliance_certificate': True, 'dependencies': inventory,
              'open_items': ['Choose and confirm the application source license before publishing.',
                             'Resolve pyvirtualcam GPL v2 terms alongside Qt LGPL v3 before public distribution.',
                             'Retain exact Qt/PySide corresponding source and applicable LGPL/GPL notices.',
                             'Verify all native-library and model attribution/source obligations against the actual bundle.']}
    (destination / 'inventory.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, default=Path('build/third-party-notices'))
    args = parser.parse_args()
    report = collect(args.destination)
    print(f"Retained wheel notices for {len(report['dependencies'])} runtime dependencies; public licensing gate remains open.")
