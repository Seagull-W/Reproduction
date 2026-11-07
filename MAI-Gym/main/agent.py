import base64
import re
from pathlib import Path

import requests
from artifacts import save_json, utc_now

ENDPOINTS = ('http://127.0.0.1:8000/v1', 'http://127.0.0.1:8001/v1')


def make_observation_message(text: str, screenshot_png: bytes) -> dict:
    encoded = base64.b64encode(screenshot_png).decode('ascii')
    return {'role': 'user', 'content': [
        {'type': 'text', 'text': text},
        {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{encoded}'}},
    ]}


def request_model(sessions: requests.Session, endpoint: str, messages: list[dict],
                  max_tokens: int, *, model='mai-gym', artifact_prefix=None,
                  read_timeout=600) -> dict:
    payload = {'model': model, 'messages': messages, 'temperature': 0,
               'max_tokens': max_tokens}
    # No automatic retry: another generation is a separate experimental attempt.
    response = sessions.post(f'{endpoint.rstrip("/")}/chat/completions',
                             json=payload, timeout=(10, read_timeout))
    if artifact_prefix is not None:
        prefix = str(artifact_prefix)
        Path(prefix + '.http.body').write_bytes(response.content)
        save_json(prefix + '.http.json', {
            'timestamp': utc_now(), 'status_code': response.status_code,
            'elapsed_seconds': response.elapsed.total_seconds(),
            'headers': {k: v for k, v in response.headers.items()
                        if k.lower() in {'content-type', 'x-request-id', 'date'}},
        })
    response.raise_for_status()
    return response.json()


def parse_action(raw_output: str) -> str:
    text = raw_output.strip()
    if text in {'WAIT', 'DONE', 'FAIL'}:
        return text
    blocks = re.findall(r'```(?:python[ \t]*\n)?(.*?)```', text, flags=re.DOTALL)
    if len(blocks) != 1:
        raise ValueError(f'预期一个动作代码块，实际得到 {len(blocks)} 个')
    action = blocks[0].strip()
    if not action:
        raise ValueError('动作代码块为空')
    return action
