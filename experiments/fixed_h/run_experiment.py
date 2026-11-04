#!/usr/bin/env python3
"""Portable CPU handoff: integrity -> neural smoke -> full -> compact return ZIP."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
import zipfile

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    if sys.flags.optimize:
        parser.error('Do not run with -O or PYTHONOPTIMIZE; assertions are required.')
    expected = json.loads((ROOT/'RELEASE_MANIFEST.json').read_text())
    for name, digest in expected.items():
        if sha(ROOT/name) != digest:
            raise RuntimeError('Package integrity failure: '+name)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run = ROOT/'runs'/stamp
    run.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[key] = str(args.threads)
    status = {'success': False, 'neural_smoke_passed': False, 'full_passed': False,
              'smoke_only': args.smoke_only, 'python': sys.version, 'platform': platform.platform(),
              'release_manifest_sha256': sha(ROOT/'RELEASE_MANIFEST.json')}
    exit_code = 1
    try:
        with (run/'pip_freeze.txt').open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=log, stderr=subprocess.STDOUT, check=True)
        for profile in (['smoke'] if args.smoke_only else ['smoke', 'full']):
            command = [sys.executable, '-u', str(ROOT/'code/run_fixed_h.py'), '--neural',
                       '--profile', profile, '--threads', str(args.threads), '--out', str(run/profile)]
            print('Starting '+profile+'; log: '+str(run/(profile+'.log')), flush=True)
            with (run/(profile+'.log')).open('w', encoding='utf-8') as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads((run/profile/'status.json').read_text())
            tests = json.loads((run/profile/'tests.json').read_text())
            if result['status'] != 'PASS' or not result['neural_executed'] or not all(tests.values()):
                raise RuntimeError('Incomplete neural '+profile)
            status['neural_smoke_passed' if profile == 'smoke' else 'full_passed'] = True
            print(profile+' PASS', flush=True)
        status['success'] = True
        exit_code = 0
    except BaseException as exc:
        status['error'] = type(exc).__name__+': '+str(exc)
        (run/'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        print('Run incomplete. Send the return ZIP for diagnosis.', flush=True)
    finally:
        status['return_archive_omits'] = 'features.npz only; retained locally with hashes, regenerate from frozen code/seed if needed'
        (run/'handoff_status.json').write_text(json.dumps(status, indent=2)+'\n', encoding='utf-8')
        archive = ROOT/('CCPE_fixed_H_RETURN_'+stamp+'.zip')
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for p in sorted(run.rglob('*')):
                if p.is_file() and p.name != 'features.npz':
                    z.write(p, 'run/'+str(p.relative_to(run)))
            for name in expected:
                if name.startswith(('code/','reference/')) or name in ('PROTOCOL.md','README.md','README_FA.md','SOURCE_PROVENANCE.json','requirements.txt','run_experiment.py'):
                    z.write(ROOT/name, 'package/'+name)
            z.write(ROOT/'RELEASE_MANIFEST.json', 'package/RELEASE_MANIFEST.json')
        archive.with_suffix('.zip.sha256').write_text(sha(archive)+'  '+archive.name+'\n', encoding='utf-8')
        print('SEND: '+str(archive), flush=True)
        print('Retain runs/ locally, including features.npz.', flush=True)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
