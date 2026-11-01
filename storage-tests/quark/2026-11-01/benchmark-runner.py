import argparse, hashlib, json, os, selectors, signal, subprocess, time
from pathlib import Path

ROOT = Path('/tmp/quark-storage-test')
RUN = Path((ROOT / 'latest-run.txt').read_text().strip())
CLI = '/home/ubuntu/.codex/skills/quarkclouddrive/scripts/quark-drive.cjs'
parser = argparse.ArgumentParser()
parser.add_argument('action', choices=['info','upload','download','upload-list','download-list','upload-resume','download-resume','interrupt-upload'])
parser.add_argument('--name', default='smoke-1MiB.bin')
parser.add_argument('--fid')
parser.add_argument('--record-id')
parser.add_argument('--interrupt-bytes', type=int, default=0)
parser.add_argument('--timeout', type=int, default=180)
parser.add_argument('--session-id', required=True)
args = parser.parse_args()
if Path(args.name).name != args.name:
    parser.error('name must be a basename')
source = RUN / 'source' / args.name
if args.action == 'interrupt-upload':
    matched=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try: argv=(proc/'cmdline').read_bytes().split(b'\0')
        except (OSError, PermissionError): continue
        if len(argv)>3 and argv[1:4]==[CLI.encode(),b'upload',str(source).encode()]:
            os.kill(int(proc.name),signal.SIGINT);matched.append(int(proc.name))
    event={'event':'controlled_upload_interrupt','file':args.name,'matched_processes':len(matched),'time_ns':time.time_ns()}
    (RUN/'controlled-interruption.json').write_text(json.dumps(event,indent=2)+'\n')
    print(json.dumps(event),flush=True)
    raise SystemExit(0 if matched else 1)
dest = RUN / 'downloads' / args.name
dest.mkdir(parents=True, exist_ok=True)
if args.action == 'info':
    command = ['get-user-info']
elif args.action == 'upload':
    command = ['upload', str(source)]
elif args.action == 'download':
    if not args.fid: parser.error('fid required')
    command = ['download', '--fid', args.fid, '--output-dir', str(dest)]
elif args.action.endswith('-list'):
    command = [args.action.split('-')[0], 'list']
else:
    if not args.record_id: parser.error('record-id required')
    command = [args.action.split('-')[0], 'resume', '--record-id', args.record_id]
    if args.action == 'upload-resume': command += ['--file-path', str(source)]
    else: command += ['--output-dir', str(dest)]
command += ['--session-id', args.session_id, '--session-input', '实际上传文件这种肯定是在无卡模式下进行，下载夸克的CLI进行测试']
label = f'{time.time_ns()}-{args.action}'
log_path = RUN / (label + '.jsonl')
started = time.monotonic()
rows = []
interrupted = False
timed_out = False
last_progress = 0
with log_path.open('x') as log:
    os.chmod(log_path, 0o600)
    process = subprocess.Popen(['node', CLI] + command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    while True:
        elapsed = time.monotonic() - started
        if elapsed > args.timeout and not timed_out:
            timed_out = True
            process.send_signal(signal.SIGINT)
        if elapsed > args.timeout + 15 and process.poll() is None:
            process.kill()
        events = selector.select(timeout=0.2)
        if not events:
            if process.poll() is not None: break
            continue
        line = process.stdout.readline()
        if not line:
            if process.poll() is not None: break
            continue
        log.write(line); log.flush()
        try: row = json.loads(line)
        except ValueError: continue
        rows.append(row)
        if row.get('type') == 'progress':
            current = row.get('data', {}).get('current', 0)
            if args.interrupt_bytes and current >= args.interrupt_bytes and not interrupted:
                interrupted = True
                process.send_signal(signal.SIGINT)
                print(json.dumps({'event':'sent_SIGINT','current_bytes':current}), flush=True)
            if elapsed - last_progress >= 10:
                print(json.dumps({'elapsed_s':round(elapsed,2),'progress':row.get('data')}), flush=True)
                last_progress = elapsed
        elif args.action != 'info':
            print(json.dumps(row, ensure_ascii=False), flush=True)
    exit_code = process.wait(timeout=20)
elapsed = time.monotonic() - started
summary = {'action':args.action,'name':args.name,'elapsed_s':round(elapsed,3),'exit_code':exit_code,'interrupted':interrupted,'timed_out':timed_out,'log_path':str(log_path),'results':[r for r in rows if r.get('type')=='result'],'items':[r for r in rows if r.get('type')=='list']}
if args.action == 'info':
    print(json.dumps({'info_results':summary['results']}, ensure_ascii=False))
if args.action.startswith('download') and not args.action.endswith('-list') and source.exists():
    checks=[]
    for p in dest.iterdir():
        if not p.is_file(): continue
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1024**2), b''): h.update(block)
        manifest=json.loads((RUN/'manifest.json').read_text())
        expected=next((x for x in manifest['files'] if x['name']==args.name),None)
        checks.append({'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest(),'matches_source':bool(expected and h.hexdigest()==expected['sha256'] and p.stat().st_size==expected['bytes'])})
    summary['download_checks']=checks
if source.exists(): summary['source_bytes']=source.stat().st_size
(RUN/(label+'.summary.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'summary':{k:v for k,v in summary.items() if k not in ['results','items']}},ensure_ascii=False),flush=True)
