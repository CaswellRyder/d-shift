"""Reconnect the Mac to an installed Pi lab bundle; run this in a normal terminal."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import time
from urllib.error import URLError
from urllib.request import urlopen
import webbrowser


def read_state(url):
    try:
        with urlopen(url + '/api/state', timeout=2) as response:
            return json.load(response)
    except (OSError, ValueError, URLError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, help='Installed absolute Pi bundle directory')
    parser.add_argument('--host', default='vision.local')
    parser.add_argument('--user', default='pacman')
    parser.add_argument('--port', type=int, default=8770)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    if not args.bundle.startswith('/') or not 1024 <= args.port <= 65535:
        parser.error('Supply an absolute Pi bundle path and a port from 1024 to 65535')
    url = f'http://127.0.0.1:{args.port}'
    state = read_state(url)
    if state is not None:
        if state.get('bundle') != args.bundle:
            parser.exit(1, 'This local port belongs to another server; choose a different --port.\n')
        print(url, flush=True)
        if state.get('error'):
            parser.exit(1, state['error'] + '\n')
        if not args.no_browser:
            webbrowser.open(url)
        return
    socket_dir = tempfile.mkdtemp(prefix='dtr-flight-lab-ssh-')
    socket = str(Path(socket_dir) / 'control')
    destination = args.user + '@' + args.host
    subprocess.run(['ssh', '-M', '-S', socket, '-o', 'ControlPersist=7200',
                    '-o', 'ExitOnForwardFailure=yes', '-o', 'ConnectTimeout=10',
                    '-fN', '-L', f'127.0.0.1:{args.port}:127.0.0.1:{args.port}', destination], check=True)
    ssh = ['ssh', '-S', socket, '-o', 'BatchMode=yes', destination]
    before = time.time()
    remote_epoch = float(subprocess.check_output(ssh + ['date +%s'], text=True).strip())
    offset = (before + time.time()) / 2 - remote_epoch
    code = f'''from pathlib import Path
import json,os,subprocess
root=Path({args.bundle!r})
script=root/'scripts/pi_flight_lab.py'
if not script.is_file():
    raise SystemExit('Installed lab bundle is missing: '+str(script))
pid_file=root/'server.pid'
running=False
if pid_file.is_file():
    try:
        pid=int(pid_file.read_text())
        running=str(script) in Path('/proc/'+str(pid)+'/cmdline').read_text()
    except (OSError,ValueError):
        pass
if not running:
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(root/'src')+':'+str(root),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',DTR_TFLITE_LIBRARY='/home/pacman/dtr-runtime-build-20261006/libtensorflowlite_c.candidate.so')
    command=['python3','-u',str(script),'--balloon-model','/home/pacman/flight-loop-20261015/models/green-purple-context.tflite','--goal-model','/home/pacman/flight-loop-lab/models/v1.0.0/goal/student.float.tflite','--output',str(root/'sessions'),'--port',{str(args.port)!r},'--wall-clock-offset-s',{str(offset)!r}]
    with (root/'server.log').open('ab') as log:
        process=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    pid_file.write_text(str(process.pid))
print(json.dumps(dict(running=True,bundle=str(root))))
'''
    subprocess.run(ssh + ['python3 -c ' + shlex.quote(code)], check=True)
    print('Waiting for Pi models and camera. ' + url, flush=True)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        state = read_state(url)
        if state is not None:
            if state.get('bundle') != args.bundle:
                parser.exit(1, 'A different bundle is already using the Pi port.\n')
            if state.get('error'):
                parser.exit(1, state['error'] + '\n')
            if state.get('ready'):
                print(url, flush=True)
                if not args.no_browser:
                    webbrowser.open(url)
                return
        time.sleep(.5)
    parser.exit(1, 'Pi viewer did not become ready. Inspect ' + args.bundle + '/server.log on the Pi.\n')


if __name__ == '__main__':
    main()
