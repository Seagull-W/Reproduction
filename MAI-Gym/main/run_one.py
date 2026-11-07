"""Single-task convenience entry; uses the same supervised runner and telemetry."""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-path', required=True)
    parser.add_argument('--prompt-path', required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--path-to-vm', required=True)
    parser.add_argument('--provider', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-steps', type=int, required=True)
    parser.add_argument('--max-tokens', type=int, required=True)
    parser.add_argument('--time-limit', type=float, required=True)
    parser.add_argument('--model-revision', required=True)
    parser.add_argument('--model-dir')
    parser.add_argument('--dcgm-url')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='mai-single-') as tmp:
        manifest = Path(tmp) / 'tasks.json'
        workers = Path(tmp) / 'workers.json'
        manifest.write_text(json.dumps([str(Path(args.task_path).resolve())]))
        workers.write_text(json.dumps([{'endpoint': args.endpoint, 'path_to_vm': args.path_to_vm}]))
        command = [sys.executable, str(Path(__file__).with_name('run_batch.py')),
                   '--manifest-path', str(manifest), '--workers-path', str(workers)]
        for key in ('prompt_path', 'provider', 'output_dir', 'max_steps', 'max_tokens', 'time_limit', 'dcgm_url', 'model_revision', 'model_dir'):
            if getattr(args, key) is not None:
                command += ['--' + key.replace('_', '-'), str(getattr(args, key))]
        raise SystemExit(subprocess.call(command))


if __name__ == '__main__':
    main()
