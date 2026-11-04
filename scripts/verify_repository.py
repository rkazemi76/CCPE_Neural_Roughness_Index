#!/usr/bin/env python3
"""Verify all hashes in MANIFEST.sha256 and repository release exclusions."""
from pathlib import Path
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'MANIFEST.sha256'

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main() -> int:
    if not MANIFEST.is_file():
        print('ERROR: MANIFEST.sha256 is missing', file=sys.stderr)
        return 2
    failures=[]; total=0
    for line in MANIFEST.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        try:
            expected, rel = line.split('  ', 1)
        except ValueError:
            failures.append(f'Malformed manifest line: {line!r}'); continue
        path=(ROOT/rel).resolve()
        if ROOT.resolve() not in path.parents:
            failures.append(f'Path escapes repository: {rel}'); continue
        if not path.is_file():
            failures.append(f'Missing: {rel}'); continue
        actual=digest(path); total += 1
        if actual != expected:
            failures.append(f'Hash mismatch: {rel}')
    forbidden=list(ROOT.rglob('features.npz'))
    if forbidden:
        failures.append('features.npz must remain outside the GitHub archive')
    if failures:
        for item in failures: print('FAIL:', item)
        print(f'{len(failures)} issue(s); {total} file(s) checked.')
        return 1
    print(f'OK: {total} file(s) verified; no features.npz is present.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
