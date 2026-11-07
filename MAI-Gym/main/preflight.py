"""Read-only checks on the eventual server. Does not create/start any VM or GPU service."""
import argparse
import os
import platform
import shutil
from pathlib import Path

from artifacts import save_json, utc_now


def inspect(output_path, provider=None):
    import psutil
    from telemetry import NvmlSampler
    result = {'timestamp': utc_now(), 'platform': platform.platform(),
              'logical_cpus': psutil.cpu_count(),
              'cpu_affinity_count': len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
              'ram_bytes': psutil.virtual_memory().total,
              'disk_free_bytes': shutil.disk_usage(Path(output_path).resolve().parent).free,
              'kvm_access': os.access('/dev/kvm', os.R_OK | os.W_OK),
              'docker_cli': shutil.which('docker'), 'gpus': [], 'blockers': []}
    for name in ('cpu.max', 'memory.max'):
        path = Path('/sys/fs/cgroup') / name
        result['cgroup_' + name] = path.read_text().strip() if path.exists() else None
    try:
        sampler = NvmlSampler()
        try:
            result['gpus'] = sampler.sample()
        finally:
            sampler.close()
    except Exception as exc:
        result['gpu_query_error'] = f'{type(exc).__name__}: {exc}'
    if not result['gpus']:
        result['blockers'].append('No visible NVML GPUs; do not start model service')
    if provider == 'docker' and (not result['kvm_access'] or not result['docker_cli']):
        result['blockers'].append('OSWorld Docker needs host deployment verification: Docker CLI or /dev/kvm access missing')
    result['target_match'] = len(result['gpus']) == 2 and all(
        g.get('memory') and g['memory']['total_bytes'] >= 44 * 2**30 for g in result['gpus'])
    result['profile_note'] = 'Target: 2 x nominal 48 GB; 4 desktop workers per replica initially, calibrate 2/4/8; CPU/RAM/VM overhead must be measured'
    save_json(output_path, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--provider')
    args = parser.parse_args()
    print(inspect(args.output, args.provider))
