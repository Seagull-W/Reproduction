"""Opt-in inference-only concurrency sweep using saved representative requests.

This sends real inference requests when invoked; it does not start services/VMs.
Scores are NOT evaluation scores. Use varied early/middle/late episode snapshots.
"""
import argparse
import concurrent.futures
import json
import threading
import time
from pathlib import Path
from statistics import median

import requests
from agent import request_model
from artifacts import save_json, append_jsonl, utc_now
from replay_request import reconstruct
from telemetry import collect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', nargs='+', required=True, help='Saved step_*.request.json files')
    parser.add_argument('--endpoints', nargs='+', required=True)
    parser.add_argument('--concurrency', nargs='+', type=int, default=[2, 4, 8])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--read-timeout', type=float, default=120)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if min(args.concurrency) < 1 or args.repeats < 1:
        parser.error('concurrency and repeats must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    payloads = [reconstruct(path) for path in args.requests]
    for i, payload in enumerate(payloads):
        save_json(args.output_dir / f'input_{i}.json', payload)
    save_json(args.output_dir / 'settings.json', {**vars(args), 'output_dir': str(args.output_dir),
        'kind': 'inference_only_calibration_not_benchmark', 'timestamp': utc_now(),
        'limitations': 'Repeated samples can improve cache hits. Include long contexts and confirm with an end-to-end task sweep.'})
    stop = threading.Event()
    monitor = threading.Thread(target=collect, args=(str(args.output_dir), args.endpoints, stop, 2), daemon=True)
    monitor.start()
    results = []
    try:
        for concurrency in args.concurrency:
            directory = args.output_dir / f'concurrency_{concurrency}'
            directory.mkdir()
            started = time.perf_counter()
            slots = {endpoint: threading.Semaphore(concurrency) for endpoint in args.endpoints}
            def one(index):
                payload = payloads[index % len(payloads)]
                endpoint = args.endpoints[index % len(args.endpoints)]
                start = time.perf_counter()
                row = {'request': index, 'input': index % len(payloads), 'endpoint': endpoint,
                       'started_at': utc_now()}
                try:
                    with slots[endpoint], requests.Session() as session:
                        response = request_model(session, endpoint, payload['messages'], payload['max_tokens'],
                            model=payload['model'], artifact_prefix=directory / f'request_{index:04d}',
                            read_timeout=args.read_timeout)
                    save_json(directory / f'request_{index:04d}.response.json', response)
                    row.update(status='completed', usage=response.get('usage'),
                               finish_reason=response['choices'][0]['finish_reason'])
                except Exception as exc:
                    row.update(status='error', error_type=type(exc).__name__, message=str(exc))
                row['seconds'] = time.perf_counter() - start
                return row
            count = concurrency * len(args.endpoints) * args.repeats
            # Equal number of calls per endpoint; each pool has the tested concurrency.
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency * len(args.endpoints)) as pool:
                rows = list(pool.map(one, range(count)))
            for row in rows:
                append_jsonl(directory / 'results.jsonl', row)
            elapsed = time.perf_counter() - started
            valid = all(r['status'] == 'completed' and (r.get('usage') or {}).get('completion_tokens') is not None for r in rows)
            tokens = sum((r.get('usage') or {}).get('completion_tokens', 0) or 0 for r in rows)
            results.append({'concurrency_per_endpoint': concurrency, 'requests': count,
                'errors': sum(r['status'] != 'completed' for r in rows), 'wall_seconds': elapsed,
                'output_tokens_per_second': tokens / elapsed if valid else None,
                'median_request_seconds': median(r['seconds'] for r in rows)})
            save_json(args.output_dir / 'sweep.json', results)
    finally:
        stop.set()
        monitor.join(30)
    eligible = [r for r in results if r['output_tokens_per_second'] is not None and not r['errors']]
    best = max(eligible, key=lambda r: r['output_tokens_per_second']) if eligible else None
    save_json(args.output_dir / 'recommendation.json', {'best_measured': best,
        'scope': 'Only these requests, engine settings, hardware and cache conditions; validate desktop CPU/RAM and end-to-end episodes before increasing workers'})


if __name__ == '__main__':
    main()
