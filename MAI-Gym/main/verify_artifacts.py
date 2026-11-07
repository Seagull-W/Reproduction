"""Verify a copied run against its offline artifact-manifest.json; no files removed."""
import argparse
import json
from pathlib import Path
from artifacts import sha256


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'artifact-manifest.json').read_text())
    failures = []
    for item in manifest['files']:
        path = (root / item['path']).resolve()
        if not path.is_relative_to(root):
            failures.append({'path': item['path'], 'reason': 'outside run directory'})
        elif not path.is_file():
            failures.append({'path': item['path'], 'reason': 'missing'})
        elif path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
            failures.append({'path': item['path'], 'reason': 'size/hash mismatch'})
    return {'files_checked': len(manifest['files']), 'failures': failures, 'verified': not failures}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    args = parser.parse_args()
    result = verify(args.run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['verified'] else 1)
