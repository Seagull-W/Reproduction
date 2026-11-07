"""Offline summaries/CSV/figures from saved evidence; never contacts a model or VM."""
import argparse
import csv
import gzip
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean

from artifacts import save_json, read_jsonl, sha256, utc_now


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def summarize(root):
    root = Path(root)
    manifest = json.loads((root / 'run-manifest.json').read_text())
    planned = {job['job_id']: job for job in manifest['jobs']}
    records, errors, claims = {}, [], {}
    for directory in sorted(root.glob('worker_*')):
        if not directory.is_dir():
            continue
        rows, problems = read_jsonl(directory / 'results.jsonl')
        errors.extend(problems)
        for row in rows:
            job_id = row['job_id']
            if job_id in records:
                errors.append({'duplicate_job_id': job_id, 'file': str(directory)})
                continue
            records[job_id] = row
        events, problems = read_jsonl(directory / 'events.jsonl')
        errors.extend(problems)
        for event in events:
            if event.get('event') == 'task_claimed':
                claims[event['job_id']] = event
    rows = []
    for job_id, job in planned.items():
        if job_id in records:
            rows.append(records[job_id])
        else:
            claim = claims.get(job_id)
            rows.append({**job, 'status': 'interrupted' if claim else 'not_started', 'score': None,
                         'worker_id': claim['worker_id'] if claim else None})
    for job_id in records.keys() - planned.keys():
        errors.append({'unplanned_result': job_id})
    counts = Counter(row['status'] for row in rows)
    completed = [r for r in rows if r['status'] == 'completed']
    scores = [float(r['score']) for r in completed if r.get('score') is not None]
    summary = {'planned': len(rows), 'completed': len(completed), 'status_counts': dict(counts),
               'mean_score_completed': mean(scores) if scores else None,
               'mean_score_all': mean(scores) if len(scores) == len(rows) and rows else None,
               'score_one_count': sum(score == 1 for score in scores),
               'score_one_rate_completed': sum(score == 1 for score in scores) / len(scores) if scores else None,
               'missing_scores': len(rows) - len(scores), 'integrity_errors': errors,
               'denominator_note': 'completed-only statistics exclude errors and unstarted tasks; all-run mean is null until every score exists',
               'generated_at': utc_now()}
    status_path = root / 'status.json'
    if status_path.exists():
        status = json.loads(status_path.read_text())
        elapsed = status.get('wall_seconds')
        summary['wall_seconds'] = elapsed
        summary['completed_episodes_per_hour'] = len(completed) * 3600 / elapsed if elapsed else None
    domains = {}
    for domain in sorted({r.get('domain', 'unknown') for r in rows}):
        subset = [r for r in rows if r.get('domain', 'unknown') == domain]
        scored = [float(r['score']) for r in subset if r['status'] == 'completed' and r.get('score') is not None]
        domains[domain] = {'planned': len(subset), 'scored': len(scored),
                           'mean_score_completed': mean(scored) if scored else None}
    summary['domains'] = domains
    save_json(root / 'summary.json', summary)
    save_json(root / 'results.json', rows)
    save_json(root / 'unfinished.json', [r for r in rows if r['status'] != 'completed'])
    write_csv(root / 'results.csv', rows)
    return summary


def export_metrics(root):
    analysis = root / 'analysis'
    analysis.mkdir(exist_ok=True)
    steps, errors = [], []
    for path in sorted(root.glob('worker_*/*/steps.jsonl')):
        records, problems = read_jsonl(path)
        steps.extend({'episode': path.parent.name, 'worker': path.parent.parent.name, **r} for r in records)
        errors.extend(problems)
    samples, problems = read_jsonl(root / 'telemetry/samples.jsonl')
    errors.extend(problems)
    gpus = []
    for sample in samples:
        for gpu in sample['gpus']:
            flat = {k: v for k, v in gpu.items() if k not in ('memory', 'utilization')}
            flat.update(gpu.get('memory') or {})
            flat.update(gpu.get('utilization') or {})
            gpus.append({'unix_seconds': sample['unix_seconds'], 'timestamp': sample['timestamp'], **flat})
    prometheus = []
    prom_path = root / 'telemetry/prometheus.jsonl.gz'
    if prom_path.exists():
        # Keep labels verbatim; raw text is retained for any newer metric names.
        pattern = re.compile(r'^([\w:]+)(\{.*\})?\s+([^\s]+)(?:\s+.*)?$')
        try:
            with gzip.open(prom_path, 'rt', encoding='utf-8') as stream:
                for line in stream:
                    scrape = json.loads(line)
                    for metric in scrape.get('body', '').splitlines():
                        match = pattern.match(metric)
                        if not match:
                            continue
                        try:
                            value = float(match[3])
                        except ValueError:
                            continue
                        if math.isfinite(value):
                            prometheus.append({'timestamp': scrape['timestamp'], 'url': scrape['url'],
                                               'metric': match[1], 'labels': match[2] or '', 'value': value})
        except (EOFError, OSError, json.JSONDecodeError) as exc:
            errors.append({'file': str(prom_path), 'error': str(exc)})
    write_csv(analysis / 'steps.csv', steps)
    write_csv(analysis / 'gpu.csv', gpus)
    write_csv(analysis / 'host.csv', [{k: v for k, v in s.items() if k != 'gpus'} for s in samples])
    write_csv(analysis / 'prometheus.csv', prometheus)
    save_json(analysis / 'read_errors.json', errors)
    return steps, gpus, prometheus


def plot_results(root, steps, gpus, prometheus):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from datetime import datetime

    analysis = root / 'analysis'
    rows = json.loads((root / 'results.json').read_text())
    def save(fig, name):
        fig.tight_layout()
        for extension in ('png', 'svg', 'pdf'):
            fig.savefig(analysis / f'{name}.{extension}', dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    counts = Counter(r['status'] for r in rows)
    axes[0, 0].bar(list(counts), list(counts.values()))
    axes[0, 0].set(title=f'Episode status (planned n={len(rows)})', ylabel='Episodes')
    completed = [r for r in rows if r['status'] == 'completed']
    axes[0, 1].hist([r['score'] for r in completed], bins=10)
    axes[0, 1].set(title=f'Verifier score (completed n={len(completed)})', xlabel='Score', ylabel='Episodes')
    axes[1, 0].hist([r['inference_seconds'] for r in steps if 'inference_seconds' in r], bins=20)
    axes[1, 0].set(title='Model request latency (includes queuing)', xlabel='Seconds', ylabel='Requests')
    axes[1, 1].scatter([r.get('steps', 0) for r in completed], [r['score'] for r in completed], alpha=0.6)
    axes[1, 1].set(xlabel='Interaction steps', ylabel='Verifier score', title='Completed episodes')
    save(fig, 'evaluation')
    phase_keys = ['reset_seconds', 'inference_seconds', 'execute_seconds', 'evaluate_seconds']
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([k.removesuffix('_seconds') for k in phase_keys],
           [sum(r.get(k, 0) for r in completed) for k in phase_keys])
    ax.set(ylabel='Sum of episode seconds (parallel times overlap)', title='Completed episode phase costs')
    save(fig, 'phases')
    if gpus:
        fields = [('gpu_percent', 'GPU busy (%)'), ('used_bytes', 'Device memory (GiB)'),
                  ('power_w', 'Power (W)'), ('temperature_c', 'Temperature (C)'),
                  ('sm_clock_mhz', 'SM clock (MHz)'), ('pcie_tx_kib_s', 'PCIe TX (KiB/s)'),
                  ('pcie_rx_kib_s', 'PCIe RX (KiB/s)'), ('energy_j', 'Device cumulative energy (J)')]
        fig, axes = plt.subplots(4, 2, figsize=(13, 13))
        t0 = min(r['unix_seconds'] for r in gpus)
        energy = []
        for index in sorted({r['index'] for r in gpus}):
            group = [r for r in gpus if r['index'] == index]
            for ax, (key, label) in zip(axes.flat, fields):
                available = [r for r in group if r.get(key) is not None]
                ax.plot([r['unix_seconds'] - t0 for r in available],
                        [r[key] / 2**30 if key == 'used_bytes' else r[key] for r in available], label=f'GPU {index}')
                ax.set(xlabel='Seconds since first sample', ylabel=label)
            measured = [r for r in group if r.get('energy_j') is not None]
            if len(measured) >= 2 and measured[-1]['energy_j'] >= measured[0]['energy_j']:
                energy.append({'gpu': index, 'joules': measured[-1]['energy_j'] - measured[0]['energy_j'],
                               'method': 'NVML cumulative counter delta; sampling interval coverage'})
            else:
                pairs = list(zip(group, group[1:]))
                usable = [(a, b) for a, b in pairs if a.get('power_w') is not None and b.get('power_w') is not None]
                energy.append({'gpu': index,
                    'joules': sum((a['power_w'] + b['power_w']) / 2 * (b['unix_seconds'] - a['unix_seconds']) for a, b in usable) if usable else None,
                    'method': 'estimated trapezoid integral of sampled power; not a hardware energy counter'})
        axes.flat[0].legend()
        save(fig, 'gpu_runtime')
        save_json(analysis / 'energy.json', energy)
    # Engine queues/cache/preemptions and DCGM profiling are separately observable.
    selected = sorted({r['metric'] for r in prometheus if any(s in r['metric'].lower() for s in (
        'num_requests_running', 'num_requests_waiting', 'kv_cache_usage', 'gpu_cache_usage',
        'num_preemptions', 'prof_sm_', 'prof_pipe_tensor', 'prof_dram'))})
    for metric in selected:
        fig, ax = plt.subplots(figsize=(10, 4))
        series = [r for r in prometheus if r['metric'] == metric]
        keys = sorted({(r['url'], r['labels']) for r in series})
        for key in keys:
            group = [r for r in series if (r['url'], r['labels']) == key]
            ax.plot([datetime.fromisoformat(r['timestamp']) for r in group], [r['value'] for r in group], label=' '.join(key))
        ax.set(title=metric, ylabel='Raw exporter units; see retained HELP/TYPE', xlabel='UTC')
        ax.legend(fontsize=6)
        save(fig, re.sub(r'[^\w.-]', '_', metric))
    save_json(analysis / 'plot_metadata.json', {
        'missing_gpu_metrics': 'null/missing values are omitted, never zero-filled',
        'smoothing': 'none', 'gpu_energy': 'device total, not attributed to individual tasks',
        'statistics': 'trials are not independent tasks; no confidence intervals inferred',
        'matplotlib_version': matplotlib.__version__,
        'inputs': {str(p.relative_to(root)): sha256(p) for p in analysis.glob('*.csv')}})


def inventory(root):
    rows = [{'path': str(path.relative_to(root)), 'bytes': path.stat().st_size, 'sha256': sha256(path)}
            for path in sorted(root.rglob('*')) if path.is_file() and path.name != 'artifact-manifest.json']
    save_json(root / 'artifact-manifest.json', {'created_at': utc_now(), 'copy_status': 'local_only', 'files': rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--plots', action='store_true')
    parser.add_argument('--inventory', action='store_true')
    args = parser.parse_args()
    summary = summarize(args.run_dir)
    steps, gpus, prom = export_metrics(args.run_dir)
    if args.plots:
        plot_results(args.run_dir, steps, gpus, prom)
    if args.inventory:
        inventory(args.run_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
