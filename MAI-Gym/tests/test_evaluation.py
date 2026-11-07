import base64
import copy
import json
import os
import signal
import time
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
MAIN = Path(__file__).resolve().parents[1] / 'main'
sys.path.insert(0, str(MAIN))
from agent import parse_action, request_model
from evaluate import run_episode
from replay_request import reconstruct
from artifacts import save_json, append_jsonl
from analyze import summarize, export_metrics, plot_results, inventory
from run_batch import load_task_paths, validate_workers
from verifier_trace import trace_verifier, trace_controller
import requests
PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6K1sAAAAASUVORK5CYII=')
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        self.send_response(200); self.end_headers()
        self.wfile.write(json.dumps({'data': [{'id': 'mai-gym'}]}).encode())
    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if payload['model'] == 'error':
            self.send_response(400); self.end_headers(); self.wfile.write(b'{"error":"bad request"}'); return
        body = {'choices': [{'finish_reason': 'stop', 'message': {'content': 'WAIT'}}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 1, 'total_tokens': 11}}
        self.send_response(200); self.end_headers(); self.wfile.write(json.dumps(body).encode())
class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()
    def test_windows_and_exact_reconstruction(self):
        histories = []
        def request(*args, **kwargs):
            histories.append(copy.deepcopy(args[2]))
            return {'choices':[{'finish_reason':'stop','message':{'content':'WAIT'}}],
                    'usage':{'prompt_tokens':10,'completion_tokens':1}}
        env = SimpleNamespace(reset=lambda task_config:{'screenshot':PNG},
             step=lambda action:({'screenshot':PNG},0,False,{}), evaluate=lambda:0.25)
        with patch('evaluate.request_model', request):
            result = run_episode(env, {'instruction':'original'}, 'mock', str(self.root/'episode'), 'system', 27, 100)
        self.assertEqual(result['score'],0.25)
        for step, messages in enumerate(histories,1):
            start = 1 if step<=10 else 9+8*((step-11)//8)
            images = [i for i,m in enumerate(messages[1::2],1) if any(p['type']=='image_url' for p in m['content'])]
            self.assertEqual(images,list(range(start,step+1)))
            self.assertEqual(messages[1]['content'][0]['text'],'original')
            self.assertEqual(len(messages),2*step)
            self.assertEqual(reconstruct(self.root/'episode'/f'step_{step:04d}.request.json')['messages'],messages)
        self.assertEqual(result['completion_tokens'],27)
        (self.root/'episode/step_0027.png').write_bytes(b'corrupt')
        with self.assertRaises(ValueError): reconstruct(self.root/'episode/step_0027.request.json')
    def test_parse_and_truncation(self):
        for value in ('```python\n```','```python\na\n```\n```python\nb\n```'):
            with self.assertRaises(ValueError): parse_action(value)
        called=[]
        env=SimpleNamespace(reset=lambda task_config:{'screenshot':PNG},step=lambda a:called.append(a))
        response={'choices':[{'finish_reason':'length','message':{'content':'WAIT'}}]}
        with patch('evaluate.request_model',return_value=response), self.assertRaises(ValueError):
            run_episode(env,{'instruction':'test'},'mock',str(self.root/'episode'),'system',2,10)
        self.assertFalse(called)
        self.assertTrue((self.root/'episode/step_0001.response.json').exists())
        self.assertEqual(json.loads((self.root/'episode/error.json').read_text())['stage'],'parse')
    def test_manifest_and_workers(self):
        path=self.root/'manifest.json'; path.write_text('["a.json", "b.json"]')
        self.assertEqual(load_task_paths(path),[self.root/'a.json',self.root/'b.json'])
        with self.assertRaises(ValueError):
            validate_workers([{'endpoint':'http://localhost:8000/v1','path_to_vm':'same'}]*2)
    def test_docker_shared_readonly_image(self):
        workers=[{'endpoint':'http://localhost:8000/v1','path_to_vm':'/data/Ubuntu.qcow2'}]*2
        validate_workers(workers,provider='docker')
        with self.assertRaises(ValueError): validate_workers(workers,provider='vmware')
    def test_verifier(self):
        fn=lambda x:x+1; env=SimpleNamespace(metric=fn)
        with trace_verifier(env,self.root): self.assertEqual(env.metric(2),3)
        self.assertIs(env.metric,fn)
        self.assertEqual(json.loads((self.root/'verifier_0001.json').read_text()),3)
    def test_controller_return_capture(self):
        fn=lambda command:{'stdout':command,'returncode':0}
        controller=SimpleNamespace(execute_python_command=fn)
        env=SimpleNamespace(controller=controller)
        with trace_controller(env,self.root):
            result=controller.execute_python_command('print(1)')
        self.assertEqual(result['stdout'],'print(1)')
        self.assertIs(controller.execute_python_command,fn)
        self.assertEqual(json.loads((self.root/'controller.jsonl').read_text())['result'],result)
    def test_summary_missing_and_plots(self):
        save_json(self.root/'run-manifest.json',{'jobs':[{'job_id':str(i),'domain':'test'} for i in range(3)]})
        worker=self.root/'worker_00'; worker.mkdir()
        append_jsonl(worker/'results.jsonl',{'job_id':'0','domain':'test','status':'completed','score':0.5,'steps':2})
        append_jsonl(worker/'events.jsonl',{'job_id':'1','worker_id':0,'event':'task_claimed'})
        summary=summarize(self.root)
        self.assertEqual(summary['status_counts'],{'completed':1,'interrupted':1,'not_started':1})
        self.assertIsNone(summary['mean_score_all'])
        steps,gpus,prom=export_metrics(self.root)
        plot_results(self.root,steps,gpus,prom); inventory(self.root)
        self.assertTrue((self.root/'analysis/evaluation.png').exists())
        self.assertTrue((self.root/'artifact-manifest.json').exists())
class ProcessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proxy_patch = patch.dict(os.environ, {'NO_PROXY': '*', 'no_proxy': '*'})
        cls.proxy_patch.start()
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True); cls.thread.start()
        cls.endpoint=f'http://127.0.0.1:{cls.server.server_port}/v1'
    @classmethod
    def tearDownClass(cls): cls.server.shutdown(); cls.server.server_close(); cls.thread.join(); cls.proxy_patch.stop()
    def test_http_error_body(self):
        with tempfile.TemporaryDirectory() as tmp, requests.Session() as session:
            prefix=Path(tmp)/'request'
            with self.assertRaises(requests.HTTPError):
                request_model(session,self.endpoint,[],1,model='error',artifact_prefix=prefix,read_timeout=5)
            self.assertIn('bad request',Path(str(prefix)+'.http.body').read_text())
    def run_batch(self, instructions, worker_paths=('one','two'), extra=(), interrupt=False):
        tmp=tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup); root=Path(tmp.name); paths=[]
        for i,instruction in enumerate(instructions):
            path=root/f'task_{i}.json'; save_json(path,{'instruction':instruction}); paths.append(str(path))
        save_json(root/'tasks.json',paths)
        save_json(root/'workers.json',[{'endpoint':self.endpoint,'path_to_vm':p} for p in worker_paths])
        (root/'prompt.txt').write_text('system')
        env={**os.environ,'PYTHONPATH':str(Path(__file__).parent/'fake_osworld'),'PYTHONDONTWRITEBYTECODE':'1'}
        command=[sys.executable,str(MAIN/'run_batch.py'),'--manifest-path',str(root/'tasks.json'),
            '--workers-path',str(root/'workers.json'),'--prompt-path',str(root/'prompt.txt'),
            '--provider','mock','--output-dir',str(root/'run'),'--max-steps','3','--max-tokens','10',
            '--model-revision','test-revision','--time-limit','30','--no-telemetry','--min-free-gib','0',*extra]
        if not interrupt:
            return root/'run',subprocess.run(command,env=env,capture_output=True,text=True,timeout=45)
        process=subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
        try:
            deadline=time.monotonic()+10
            state=root/'run/worker_00/state.json'
            while time.monotonic()<deadline:
                if state.exists() and json.loads(state.read_text()).get('status')=='running': break
                time.sleep(0.05)
            else: self.fail('worker did not reach running state')
            process.send_signal(signal.SIGINT)
            stdout,stderr=process.communicate(timeout=20)
            return root/'run',SimpleNamespace(returncode=process.returncode,stdout=stdout,stderr=stderr)
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGKILL)
                process.communicate()

    def test_spawn_dynamic_trials(self):
        root,result=self.run_batch(['a','b','c'],extra=('--trials','2'))
        self.assertEqual(result.returncode,0,result.stderr)
        summary=json.loads((root/'summary.json').read_text())
        self.assertEqual(summary['completed'],6); self.assertEqual(summary['mean_score_all'],0.5)
        rows=json.loads((root/'results.json').read_text())
        self.assertEqual(len({r['job_id'] for r in rows}),6)
        for worker in root.glob('worker_*'):
            if worker.is_dir(): self.assertTrue((worker/'env_cache/closed').exists())
    def test_failure_is_not_zero_score(self):
        root,result=self.run_batch(['fail','b'],worker_paths=('one',))
        self.assertNotEqual(result.returncode,0)
        summary=json.loads((root/'summary.json').read_text())
        self.assertEqual(summary['status_counts'],{'error':1,'not_started':1}); self.assertIsNone(summary['mean_score_all'])
    def test_timeout(self):
        root,result=self.run_batch(['hang'],worker_paths=('one',),extra=('--episode-timeout','0.2'))
        self.assertNotEqual(result.returncode,0)
        self.assertEqual(json.loads((root/'worker_exitcodes.json').read_text())[0]['forced_reason'],'episode_timeout')
        self.assertTrue((root/'worker_00/env_cache/closed').exists())
    def test_dry_run_starts_nothing(self):
        root,result=self.run_batch(['a'],extra=('--dry-run',))
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads((root/'status.json').read_text())['status'],'planned')
        self.assertFalse(list(root.glob('worker_*')))
        self.assertFalse((root/'telemetry').exists())
    def test_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            request=root/'request.json'
            save_json(request,{'model':'mai-gym','max_tokens':10,'temperature':0,
                               'messages':[{'role':'user','content':'test'}]})
            result=subprocess.run([sys.executable,str(MAIN/'calibrate.py'),
                '--requests',str(request),'--endpoints',self.endpoint,'--concurrency','1','2',
                '--repeats','1','--read-timeout','5','--output-dir',str(root/'calibration')],
                capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            data=json.loads((root/'calibration/recommendation.json').read_text())
            self.assertIsNotNone(data['best_measured'])
            self.assertTrue((root/'calibration/telemetry/status.json').exists())
    def test_parent_interrupt_cleans_worker(self):
        root,result=self.run_batch(['hang','later'],worker_paths=('one',),interrupt=True)
        self.assertNotEqual(result.returncode,0)
        self.assertTrue((root/'worker_00/env_cache/closed').exists())
        self.assertEqual(json.loads((root/'status.json').read_text())['stop_reason'],'parent_exception_or_interrupt')
        self.assertEqual(json.loads((root/'summary.json').read_text())['planned'],2)
    def test_init_failure(self):
        root,result=self.run_batch(['a'],worker_paths=('init-fail',))
        self.assertNotEqual(result.returncode,0)
        self.assertTrue((root/'worker_00/worker_error.json').exists())
        self.assertEqual(json.loads((root/'summary.json').read_text())['status_counts'],{'not_started':1})
if __name__=='__main__': unittest.main()
