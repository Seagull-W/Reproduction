import json
import gzip
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'main'))
from telemetry import NvmlSampler, collect
from analyze import export_metrics, plot_results, inventory
from artifacts import save_json, append_jsonl
from verify_artifacts import verify

class TelemetryTests(unittest.TestCase):
    def test_unsupported_is_null_with_reason(self):
        sampler=NvmlSampler.__new__(NvmlSampler)
        sampler.n=SimpleNamespace(NVML_TEMPERATURE_GPU=0,NVML_CLOCK_SM=1,NVML_CLOCK_MEM=2,
            NVML_CLOCK_GRAPHICS=0,NVML_PCIE_UTIL_TX_BYTES=0,NVML_PCIE_UTIL_RX_BYTES=1,
            NVML_MEMORY_ERROR_TYPE_UNCORRECTED=1,NVML_AGGREGATE_ECC=1,
            nvmlDeviceGetUUID=lambda h:'GPU-test',nvmlDeviceGetName=lambda h:'fake',
            nvmlDeviceGetPowerUsage=lambda h:120000)
        sampler.handles=[object()]
        row=sampler.sample()[0]
        self.assertEqual(row['power_w'],120)
        self.assertIsNone(row['energy_j'])
        self.assertIn('energy_j',row['missing'])
        self.assertIsNone(row['utilization'])

    def test_no_driver_collection_and_prometheus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); stop=threading.Event(); stop.set()
            with patch('telemetry.NvmlSampler',side_effect=RuntimeError('driver missing')), \
                 patch('telemetry.command_snapshot',return_value={'error':'not installed'}), \
                 patch('telemetry.scrape',return_value={'timestamp':'2026-11-06T00:00:00+00:00','url':'mock',
                       'body':'# TYPE vllm:num_requests_running gauge\nvllm:num_requests_running{model="mai"} 2\nDCGM_FI_PROF_SM_ACTIVE{gpu="0"} 0.5\n'}):
                collect(str(root),['http://localhost:8000/v1'],stop)
            row=json.loads((root/'telemetry/samples.jsonl').read_text())
            self.assertEqual(row['gpus'],[])
            self.assertIn('driver missing',row['gpu_missing_reason'])
            _,_,prom=export_metrics(root)
            self.assertEqual(len(prom),2)

    def test_gpu_plots_and_copy_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'telemetry').mkdir(); save_json(root/'results.json',[])
            for second in (0,2):
                append_jsonl(root/'telemetry/samples.jsonl',{'timestamp':f'2026-11-06T00:00:0{second}+00:00',
                    'unix_seconds':100+second,'gpus':[{'index':0,'uuid':'GPU-test',
                    'memory':{'used_bytes':2**30},'utilization':{'gpu_percent':50},
                    'power_w':100,'energy_j':200+second*100,'sm_clock_mhz':1000,
                    'temperature_c':50,'pcie_rx_kib_s':10,'pcie_tx_kib_s':20,'missing':{}}]})
            steps,gpus,prom=export_metrics(root); plot_results(root,steps,gpus,prom)
            self.assertTrue((root/'analysis/gpu_runtime.pdf').exists())
            self.assertEqual(json.loads((root/'analysis/energy.json').read_text())[0]['joules'],200)
            inventory(root); self.assertTrue(verify(root)['verified'])
            (root/'results.json').write_text('changed'); self.assertFalse(verify(root)['verified'])
