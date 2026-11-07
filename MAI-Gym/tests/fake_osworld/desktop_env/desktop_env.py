import base64
import time
from pathlib import Path
PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6K1sAAAAASUVORK5CYII=')
class DesktopEnv:
    def __init__(self, **kwargs):
        self.cache = Path(kwargs['cache_dir']); self.cache.mkdir()
        if kwargs['path_to_vm'] == 'init-fail': raise RuntimeError('test init failure')
        self.steps = 0
    def reset(self, task_config):
        self.steps = 0
        instruction = task_config['instruction']
        if instruction == 'fail': raise ValueError('test reset failure')
        if instruction == 'hang': time.sleep(20)
        return {'screenshot': PNG, 'instruction': instruction, 'accessibility_tree': None}
    def step(self, action):
        self.steps += 1
        return {'screenshot': PNG}, 0, self.steps >= 2, {'mock': True}
    def evaluate(self): return 0.5
    def close(self): (self.cache / 'closed').write_text('yes')
