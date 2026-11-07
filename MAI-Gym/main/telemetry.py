"""Read-only NVML and host sampling plus raw vLLM/DCGM Prometheus scrapes.

No performance-counter value is synthesized from GPU utilization. Unsupported
NVML queries are null with a reason; DCGM profiling requires host support.
"""
import concurrent.futures
import gzip
import json
import platform
import subprocess
import time
import traceback
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from artifacts import save_json, append_jsonl, utc_now


def metrics_url(endpoint):
    parts = urlsplit(endpoint)
    return urlunsplit((parts.scheme, parts.netloc, '/metrics', '', ''))


def command_snapshot(command, path):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=20)
        path.write_text(result.stdout, encoding='utf-8')
        return {'command': command, 'exitcode': result.returncode, 'stderr': result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'command': command, 'error': str(exc)}


class NvmlSampler:
    def __init__(self):
        import pynvml
        self.n = pynvml
        self.n.nvmlInit()
        self.handles = [self.n.nvmlDeviceGetHandleByIndex(i)
                        for i in range(self.n.nvmlDeviceGetCount())]

    def sample(self):
        n = self.n
        rows = []
        for index, handle in enumerate(self.handles):
            errors = {}
            def query(key, fn, *args, convert=lambda x: x):
                try:
                    return convert(getattr(n, fn)(handle, *args))
                except Exception as exc:
                    errors[key] = f'{type(exc).__name__}: {exc}'
                    return None
            row = {'index': index}
            for key, fn in [('uuid', 'nvmlDeviceGetUUID'), ('name', 'nvmlDeviceGetName')]:
                row[key] = query(key, fn, convert=lambda x: x.decode() if isinstance(x, bytes) else x)
            row['memory'] = query('memory', 'nvmlDeviceGetMemoryInfo', convert=lambda x: {
                'total_bytes': x.total, 'used_bytes': x.used, 'free_bytes': x.free})
            row['utilization'] = query('utilization', 'nvmlDeviceGetUtilizationRates', convert=lambda x: {
                'gpu_percent': x.gpu, 'memory_busy_percent': x.memory})
            for key, fn, args, scale in [
                ('power_w', 'nvmlDeviceGetPowerUsage', (), 0.001),
                ('power_limit_w', 'nvmlDeviceGetEnforcedPowerLimit', (), 0.001),
                ('energy_j', 'nvmlDeviceGetTotalEnergyConsumption', (), 0.001),
                ('temperature_c', 'nvmlDeviceGetTemperature', (n.NVML_TEMPERATURE_GPU,), 1),
                ('sm_clock_mhz', 'nvmlDeviceGetClockInfo', (n.NVML_CLOCK_SM,), 1),
                ('memory_clock_mhz', 'nvmlDeviceGetClockInfo', (n.NVML_CLOCK_MEM,), 1),
                ('graphics_clock_mhz', 'nvmlDeviceGetClockInfo', (n.NVML_CLOCK_GRAPHICS,), 1),
                ('pstate', 'nvmlDeviceGetPerformanceState', (), 1),
                ('throttle_reasons_mask', 'nvmlDeviceGetCurrentClocksThrottleReasons', (), 1),
                ('pcie_tx_kib_s', 'nvmlDeviceGetPcieThroughput', (n.NVML_PCIE_UTIL_TX_BYTES,), 1),
                ('pcie_rx_kib_s', 'nvmlDeviceGetPcieThroughput', (n.NVML_PCIE_UTIL_RX_BYTES,), 1),
                ('pcie_link_gen', 'nvmlDeviceGetCurrPcieLinkGeneration', (), 1),
                ('pcie_link_width', 'nvmlDeviceGetCurrPcieLinkWidth', (), 1),
                ('pcie_replay_count', 'nvmlDeviceGetPcieReplayCounter', (), 1),
                ('fan_percent', 'nvmlDeviceGetFanSpeed', (), 1),
                ('ecc_uncorrected_total', 'nvmlDeviceGetTotalEccErrors',
                 (n.NVML_MEMORY_ERROR_TYPE_UNCORRECTED, n.NVML_AGGREGATE_ECC), 1),
            ]:
                row[key] = query(key, fn, *args, convert=lambda x, s=scale: x * s)
            mask = row['throttle_reasons_mask']
            row['throttle_reasons'] = None if mask is None else [
                name.removeprefix('nvmlClocksThrottleReason') for name in dir(n)
                if name.startswith('nvmlClocksThrottleReason') and name not in (
                    'nvmlClocksThrottleReasonAll', 'nvmlClocksThrottleReasonNone')
                and isinstance(getattr(n, name), int) and mask & getattr(n, name)]
            row['compute_processes'] = query('compute_processes', 'nvmlDeviceGetComputeRunningProcesses',
                convert=lambda processes: [{'pid': p.pid,
                    'used_memory_bytes': p.usedGpuMemory if p.usedGpuMemory < 2**63 else None}
                    for p in processes])
            row['missing'] = errors
            rows.append(row)
        return rows

    def close(self):
        self.n.nvmlShutdown()


def scrape(url):
    import requests
    try:
        response = requests.get(url, timeout=(2, 3))
        return {'url': url, 'timestamp': utc_now(), 'status_code': response.status_code,
                'body': response.text}
    except requests.RequestException as exc:
        return {'url': url, 'timestamp': utc_now(), 'error': str(exc)}


def collect(output_dir, endpoints, stop_event, interval=2.0, dcgm_url=None):
    import psutil
    directory = Path(output_dir) / 'telemetry'
    directory.mkdir(parents=True, exist_ok=True)
    status = {'started_at': utc_now(), 'interval_seconds': interval,
              'host': platform.platform(), 'nvml': 'pending',
              'dcgm': dcgm_url or 'not_configured; SM/tensor/DRAM profiling not measured'}
    sampler = None
    try:
        try:
            sampler = NvmlSampler()
            status['nvml'] = 'available'
        except Exception as exc:
            status['nvml'] = f'unavailable: {type(exc).__name__}: {exc}'
        status['snapshots'] = [command_snapshot(['nvidia-smi', '-q', '-x'], directory / 'nvidia-smi.xml'),
                               command_snapshot(['nvidia-smi', 'topo', '-m'], directory / 'topology.txt')]
        save_json(directory / 'status.json', status)
        urls = list(dict.fromkeys(metrics_url(endpoint) for endpoint in endpoints))
        if dcgm_url:
            urls.append(dcgm_url)
        psutil.cpu_percent(interval=None)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(urls))) as pool, \
                gzip.open(directory / 'prometheus.jsonl.gz', 'at', encoding='utf-8') as prom:
            while True:
                start = time.monotonic()
                disk = psutil.disk_usage(output_dir)
                sample = {'timestamp': utc_now(), 'unix_seconds': time.time(),
                          'cpu_percent': psutil.cpu_percent(interval=None),
                          'ram': psutil.virtual_memory()._asdict(),
                          'swap': psutil.swap_memory()._asdict(),
                          'disk': disk._asdict(),
                          'io': psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else None,
                          'network': psutil.net_io_counters()._asdict(),
                          'gpus': sampler.sample() if sampler else [],
                          'gpu_missing_reason': None if sampler else status['nvml']}
                append_jsonl(directory / 'samples.jsonl', sample)
                for result in pool.map(scrape, urls):
                    prom.write(json.dumps(result, ensure_ascii=False) + '\n')
                prom.flush()
                if stop_event.wait(max(0, interval - (time.monotonic() - start))):
                    break
    except BaseException:
        status['error'] = traceback.format_exc()
        raise
    finally:
        if sampler:
            sampler.close()
        status['finished_at'] = utc_now()
        save_json(directory / 'status.json', status)
