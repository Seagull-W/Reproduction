"""Dynamic task queue; a desktop and model endpoint stay fixed within each worker."""
import argparse
import contextlib
import importlib.metadata
import importlib.util
import json
import logging
import multiprocessing
import os
import platform
import queue
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from urllib.parse import urlsplit

from artifacts import append_jsonl, save_json, sha256, utc_now


def load_task_paths(manifest_path: str) -> list[Path]:
    manifest_path = Path(manifest_path).resolve()
    entries = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not isinstance(entries, list) or not all(isinstance(x, str) for x in entries):
        raise ValueError('Task manifest must be a JSON list of paths')
    paths = [(manifest_path.parent / entry).resolve() for entry in entries]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate task paths; use --trials for repeated evaluation')
    return paths


def validate_workers(workers, provider=None):
    if not isinstance(workers, list) or not workers:
        raise ValueError('workers must be a nonempty list')
    identities = []
    for worker in workers:
        parts = urlsplit(worker['endpoint'])
        if parts.scheme not in ('http', 'https') or not parts.netloc or parts.username or parts.query:
            raise ValueError('Use an HTTP endpoint without credentials or query parameters')
        vm = worker.get('path_to_vm')
        if not isinstance(vm, str) or not vm.strip() or '部署时' in vm or 'REPLACE' in vm:
            raise ValueError('Each worker needs an explicit, independently provisioned path_to_vm')
        identities.append(os.path.normpath(vm))
    # Docker creates a separate container for each environment; its qcow2 is mounted read-only.
    if provider != 'docker' and len(identities) != len(set(identities)):
        raise ValueError('Workers must not share a desktop instance/path_to_vm')


def interrupt_worker(signum, frame):
    raise InterruptedError(f'Worker received signal {signum}')


def run_worker(worker_id, jobs, worker, output_dir, system_prompt, options, stop_event):
    directory = Path(output_dir) / f'worker_{worker_id:02d}'
    directory.mkdir(exist_ok=False)
    state_path = directory / 'state.json'
    state = {'worker_id': worker_id, 'status': 'initializing', 'job_id': None,
             'stage_started_monotonic': time.monotonic(), 'timestamp': utc_now()}
    save_json(state_path, state)
    signal.signal(signal.SIGTERM, interrupt_worker)
    env = None
    with (directory / 'worker.log').open('a', encoding='utf-8', buffering=1) as log, \
            contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        logging.basicConfig(stream=log, level=logging.INFO, force=True)
        try:
            from desktop_env.desktop_env import DesktopEnv
            from evaluate import run_episode
            env = DesktopEnv(provider_name=options['provider'], path_to_vm=worker['path_to_vm'],
                             action_space='pyautogui', headless=True, require_a11y_tree=False,
                             require_terminal=False, cache_dir=str(directory / 'env_cache'))
            while not stop_event.is_set():
                try:
                    job = jobs.get_nowait()
                except queue.Empty:
                    break
                state.update(status='running', job_id=job['job_id'],
                             stage_started_monotonic=time.monotonic(), timestamp=utc_now())
                save_json(state_path, state)
                append_jsonl(directory / 'events.jsonl', {**state, 'event': 'task_claimed'})
                episode_dir = directory / job['job_id']
                record = {**job, 'worker_id': worker_id, 'endpoint': worker['endpoint'],
                          'output_dir': str(episode_dir.relative_to(output_dir))}
                task = json.loads((Path(output_dir) / job['snapshot']).read_text(encoding='utf-8'))
                try:
                    summary = run_episode(env, task, worker['endpoint'], str(episode_dir),
                                          system_prompt, options['max_steps'], options['max_tokens'],
                                          model=options['model'], read_timeout=options['read_timeout'],
                                          stop_event=stop_event)
                except BaseException as exc:
                    error_path = episode_dir / 'error.json'
                    error = json.loads(error_path.read_text()) if error_path.exists() else {
                        'status': 'error', 'error_type': type(exc).__name__, 'message': str(exc)}
                    append_jsonl(directory / 'results.jsonl', {**record, **error})
                    # No silent retry and no reuse of a potentially broken desktop.
                    raise
                append_jsonl(directory / 'results.jsonl', {**record, **summary})
                state.update(status='idle', job_id=None, timestamp=utc_now())
                save_json(state_path, state)
        except BaseException as exc:
            save_json(directory / 'worker_error.json', {
                **state, 'error_type': type(exc).__name__, 'message': str(exc),
                'traceback': traceback.format_exc(), 'timestamp': utc_now()})
            raise
        finally:
            existing_error = sys.exc_info()[0] is not None
            if env is not None:
                try:
                    env.close()
                    state['environment_closed'] = True
                except BaseException:
                    state['environment_closed'] = False
                    save_json(directory / 'close_error.json', {'traceback': traceback.format_exc()})
            state.update(status='exited', timestamp=utc_now())
            save_json(state_path, state)
            if state.get('environment_closed') is False and not existing_error:
                raise RuntimeError('Desktop close failed; inspect close_error.json')


def freeze_run(output_dir, task_paths, workers, prompt, args):
    snapshots = output_dir / 'tasks'
    snapshots.mkdir()
    jobs = []
    for index, path in enumerate(task_paths):
        task = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(task.get('instruction'), str):
            raise ValueError(f'Missing instruction: {path}')
        target = snapshots / f'task_{index:05d}.json'
        save_json(target, task)
        for trial in range(args.trials):
            jobs.append({'job_id': f'task_{index:05d}_trial_{trial:03d}',
                         'task_index': index, 'task_id': task.get('id', path.stem),
                         'domain': path.parent.name, 'trial': trial, 'task_path': str(path),
                         'snapshot': str(target.relative_to(output_dir)), 'sha256': sha256(target)})
    source_dir = output_dir / 'source'
    source_dir.mkdir()
    for path in Path(__file__).parent.iterdir():
        if path.suffix in ('.py', '.sh', '.txt'):
            shutil.copy2(path, source_dir / path.name)
    (output_dir / 'system_prompt.txt').write_text(prompt, encoding='utf-8')
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                         cwd=Path(__file__).parent, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = {'schema_version': 1, 'run_id': output_dir.name, 'created_at': utc_now(),
                'jobs': jobs, 'workers': workers, 'options': vars(args),
                'scheduling': 'shared dynamic queue; endpoint sticky within episode; no retries',
                'git_commit': commit, 'python': sys.version, 'platform': platform.platform(),
                'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
                'copy_status': 'local_only', 'replication': 'custom harness; official prompt not verified'}
    spec = importlib.util.find_spec('desktop_env')
    if spec and spec.origin:
        package_root = Path(spec.origin).parent
        manifest['osworld_package'] = str(package_root)
        try:
            manifest['osworld_commit'] = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=package_root, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            manifest['osworld_commit'] = None
        for name in ('desktop_env.py', 'controllers/python.py', 'controllers/setup.py'):
            path = package_root / name
            if path.exists():
                shutil.copy2(path, source_dir / ('osworld_' + name.replace('/', '_')))
    if args.model_dir:
        metadata = {}
        for name in ('config.json', 'processor_config.json', 'tokenizer_config.json',
                     'generation_config.json', 'model.safetensors.index.json'):
            path = Path(args.model_dir) / name
            if path.exists():
                metadata[name] = sha256(path)
                shutil.copy2(path, source_dir / ('model_' + name))
        manifest['model_metadata_sha256'] = metadata
    save_json(output_dir / 'run-manifest.json', manifest)
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-path', required=True)
    parser.add_argument('--workers-path', required=True)
    parser.add_argument('--prompt-path', required=True)
    parser.add_argument('--provider', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-steps', type=int, required=True)
    parser.add_argument('--max-tokens', type=int, required=True)
    parser.add_argument('--model', default='mai-gym')
    parser.add_argument('--model-revision', required=True, help='Exact deployed model revision; recorded, not inferred from alias')
    parser.add_argument('--model-dir', help='Optional local weights directory for small metadata hashes')
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--read-timeout', type=float, default=600)
    parser.add_argument('--episode-timeout', type=float, default=1800)
    parser.add_argument('--init-timeout', type=float, default=600)
    parser.add_argument('--time-limit', type=float, required=True, help='Total seconds; stops local jobs, not cloud billing')
    parser.add_argument('--min-free-gib', type=float, default=10)
    parser.add_argument('--telemetry-interval', type=float, default=2)
    parser.add_argument('--dcgm-url', help='Existing DCGM exporter metrics URL; no exporter is started')
    parser.add_argument('--no-telemetry', action='store_true', help='For CPU-only tests')
    parser.add_argument('--dry-run', action='store_true', help='Save frozen plan only; no HTTP, GPU, VM or process starts')
    args = parser.parse_args()
    for name in ('max_steps', 'max_tokens', 'trials', 'read_timeout', 'episode_timeout',
                 'init_timeout', 'time_limit', 'telemetry_interval'):
        if getattr(args, name) <= 0:
            parser.error(f'{name} must be positive')
    paths = load_task_paths(args.manifest_path)
    if not paths:
        parser.error('Empty task manifest')
    workers = json.loads(Path(args.workers_path).read_text())
    validate_workers(workers, args.provider)
    prompt = Path(args.prompt_path).read_text(encoding='utf-8')
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    jobs_list = freeze_run(output_dir, paths, workers, prompt, args)
    if args.dry_run:
        save_json(output_dir / 'status.json', {'status': 'planned', 'started': False})
        print(f'Plan saved: {output_dir}; {len(jobs_list)} episodes, {len(workers)} workers')
        return
    if shutil.disk_usage(output_dir).free < args.min_free_gib * 2**30:
        raise RuntimeError('Insufficient free disk space before run')
    # Model services are provisioned separately. Capture their actual advertised identity.
    import requests
    for index, endpoint in enumerate(dict.fromkeys(w['endpoint'] for w in workers)):
        response = requests.get(endpoint.rstrip('/') + '/models', timeout=(5, 10))
        response.raise_for_status()
        models = response.json()
        save_json(output_dir / f'endpoint_{index}.models.json', models)
        if args.model not in [m['id'] for m in models['data']]:
            raise ValueError(f'{endpoint} does not serve {args.model}')
    ctx = multiprocessing.get_context('spawn')
    stop = ctx.Event()
    telemetry_stop = ctx.Event()
    # Manager queue has no feeder-thread get_nowait race; tasks are tiny frozen references.
    manager = ctx.Manager()
    jobs = manager.Queue()
    for job in jobs_list:
        jobs.put(job)
    processes, launched, forced = [], {}, {}
    monitor = None
    started = time.monotonic()
    status = {'status': 'running', 'started_at': utc_now(), 'stop_reason': None}
    save_json(output_dir / 'status.json', status)
    try:
        if not args.no_telemetry:
            from telemetry import collect
            monitor = ctx.Process(target=collect, args=(str(output_dir),
                [w['endpoint'] for w in workers], telemetry_stop, args.telemetry_interval, args.dcgm_url))
            monitor.start()
        for worker_id, worker in enumerate(workers[:len(jobs_list)]):
            process = ctx.Process(target=run_worker, args=(worker_id, jobs, worker,
                str(output_dir), prompt, vars(args), stop))
            process.start()
            processes.append((worker_id, process))
            launched[worker_id] = time.monotonic()
        while any(p.is_alive() for _, p in processes):
            if time.monotonic() - started > args.time_limit:
                status['stop_reason'] = 'run_time_limit'
                break
            if shutil.disk_usage(output_dir).free < args.min_free_gib * 2**30:
                status['stop_reason'] = 'disk_space_limit'
                break
            if monitor is not None and monitor.exitcode is not None:
                status['stop_reason'] = 'telemetry_failed'
                break
            for worker_id, process in processes:
                if not process.is_alive():
                    continue
                state_path = output_dir / f'worker_{worker_id:02d}' / 'state.json'
                state = json.loads(state_path.read_text()) if state_path.exists() else {}
                timeout = args.episode_timeout if state.get('status') == 'running' else args.init_timeout
                since = state.get('stage_started_monotonic', launched[worker_id])
                if time.monotonic() - since > timeout:
                    forced[worker_id] = 'episode_timeout' if state.get('status') == 'running' else 'init_timeout'
                    process.terminate()
                    process.join(10)
                    if process.is_alive():
                        process.kill()
                        process.join(5)
            time.sleep(0.5)
    except BaseException:
        status['stop_reason'] = 'parent_exception_or_interrupt'
        save_json(output_dir / 'parent_error.json', {'traceback': traceback.format_exc()})
        raise
    finally:
        stop.set()
        for worker_id, process in processes:
            process.join(3)
            if process.is_alive():
                forced.setdefault(worker_id, status['stop_reason'] or 'shutdown_timeout')
                process.terminate()
                process.join(5)
                if process.is_alive():
                    process.kill()
                    process.join(5)
        telemetry_stop.set()
        if monitor is not None:
            monitor.join(15)
            if monitor.is_alive():
                monitor.terminate()
                monitor.join(5)
        exits = [{'worker_id': i, 'exitcode': p.exitcode, 'forced_reason': forced.get(i)} for i, p in processes]
        save_json(output_dir / 'worker_exitcodes.json', exits)
        status.update(finished_at=utc_now(), wall_seconds=time.monotonic() - started,
                      telemetry_exitcode=monitor.exitcode if monitor else None,
                      status='finished' if not status['stop_reason'] and all(p.exitcode == 0 for _, p in processes) else 'incomplete')
        save_json(output_dir / 'status.json', status)
        manager.shutdown()
        # Light summary now; plots and bulk hashing are explicit offline CPU steps.
        from analyze import summarize
        summary = summarize(output_dir)
    if status['status'] != 'finished' or summary['completed'] != summary['planned'] or summary['integrity_errors']:
        raise SystemExit('Run incomplete; see summary.json and worker logs')


if __name__ == '__main__':
    main()
