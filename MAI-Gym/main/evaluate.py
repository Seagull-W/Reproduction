"""One episode: full text history, N=10/O=2 image windows, complete evidence."""
import copy
from contextlib import ExitStack
import hashlib
import struct
import time
import traceback
from pathlib import Path

import requests
from agent import make_observation_message, request_model, parse_action
from artifacts import save_json, append_jsonl, utc_now
from verifier_trace import trace_verifier, trace_controller

PLACEHOLDER = '[第 {step} 轮截图已省略]'


def request_artifact(messages, image_refs, max_tokens, model):
    # Replace only base64 URLs in the saved copy; original PNGs reconstruct the wire payload.
    recorded = []
    for index, message in enumerate(messages):
        if message['role'] != 'user':
            recorded.append(copy.deepcopy(message))
            continue
        content = []
        for part in message['content']:
            if part['type'] == 'image_url':
                content.append({'type': 'image_url', 'image_url': image_refs[(index + 1) // 2]})
            else:
                content.append(copy.deepcopy(part))
        recorded.append({'role': 'user', 'content': content})
    return {'model': model, 'temperature': 0, 'max_tokens': max_tokens, 'messages': recorded}


def save_observation(directory, name, observation):
    screenshot = observation['screenshot']
    (directory / f'{name}.png').write_bytes(screenshot)
    # Keep every observation field OSWorld actually returns, not just the image.
    save_json(directory / f'{name}.observation.json',
              {k: v for k, v in observation.items() if k != 'screenshot'})
    size = list(struct.unpack('>II', screenshot[16:24])) if screenshot[:8] == b'\x89PNG\r\n\x1a\n' else None
    return {'artifact': f'{name}.png', 'sha256': hashlib.sha256(screenshot).hexdigest(),
            'bytes': len(screenshot), 'size_px': size}


def run_episode(env, task: dict, endpoint: str, output_dir: str, system_prompt: str,
                max_steps: int, max_tokens: int, *, model='mai-gym', read_timeout=600,
                stop_event=None) -> dict:
    if max_steps < 1 or max_tokens < 1:
        raise ValueError('max_steps and max_tokens must be positive')
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    started_at = utc_now()
    messages = [{'role': 'system', 'content': system_prompt}]
    image_refs = {}
    window_size, overlap_steps, window_start_step = 10, 2, 1
    step, stage = 0, 'reset'
    totals = {'inference_seconds': 0.0, 'execute_seconds': 0.0,
              'prompt_tokens': 0, 'completion_tokens': 0, 'usage_missing_steps': 0}
    save_json(directory / 'task.json', task)
    save_json(directory / 'settings.json', {
        'endpoint': endpoint, 'system_prompt': system_prompt, 'model': model,
        'max_steps': max_steps, 'max_tokens': max_tokens, 'temperature': 0,
        'history_policy': 'full_text_windowed_images', 'window_size': window_size,
        'overlap_steps': overlap_steps, 'image_placeholder_template': PLACEHOLDER,
        'read_timeout': read_timeout, 'started_at': started_at,
        'input_image_transform': 'none; original screenshot bytes',
    })
    tracing = ExitStack()
    try:
        reset_start = time.perf_counter()
        observation = env.reset(task_config=task)
        totals['reset_seconds'] = time.perf_counter() - reset_start
        tracing.enter_context(trace_controller(env, directory))
        termination = 'max_steps'
        with requests.Session() as session:
            for step in range(1, max_steps + 1):
                stage = 'stop_check'
                if stop_event is not None and stop_event.is_set():
                    raise InterruptedError('Run stop requested')
                prefix = f'step_{step:04d}'
                metric = {'step': step, 'started_at': utc_now()}
                step_start = time.perf_counter()
                stage = 'observation'
                image_refs[step] = save_observation(directory, prefix, observation)
                text = task['instruction'] if step == 1 else '根据当前截图继续完成任务'
                if step - window_start_step >= window_size:
                    new_start = step - overlap_steps
                    for old_step in range(window_start_step, new_start):
                        message = messages[2 * old_step - 1]
                        for index, part in enumerate(message['content']):
                            if part['type'] == 'image_url':
                                message['content'][index] = {
                                    'type': 'text', 'text': PLACEHOLDER.format(step=old_step)}
                    window_start_step = new_start
                messages.append(make_observation_message(text, observation['screenshot']))
                save_json(directory / f'{prefix}.input.json', {
                    'text': text, 'screenshot': f'{prefix}.png',
                    'image_window_start_step': window_start_step})
                save_json(directory / f'{prefix}.request.json',
                          request_artifact(messages, image_refs, max_tokens, model))
                stage = 'inference'
                infer_start = time.perf_counter()
                response = request_model(session, endpoint, messages, max_tokens,
                                         model=model, artifact_prefix=directory / prefix,
                                         read_timeout=read_timeout)
                metric['inference_seconds'] = time.perf_counter() - infer_start
                totals['inference_seconds'] += metric['inference_seconds']
                save_json(directory / f'{prefix}.response.json', response)
                usage = response.get('usage') or {}
                metric.update({key: usage.get(key) for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')})
                if usage.get('prompt_tokens') is None or usage.get('completion_tokens') is None:
                    totals['usage_missing_steps'] += 1
                for key in ('prompt_tokens', 'completion_tokens'):
                    totals[key] += usage.get(key) or 0
                stage = 'parse'
                choice = response['choices'][0]
                metric['finish_reason'] = choice.get('finish_reason')
                if choice['finish_reason'] == 'length':
                    raise ValueError('模型输出被截断，不执行不完整动作')
                raw_output = choice['message']['content']
                action = parse_action(raw_output)
                save_json(directory / f'{prefix}.action.json', {'action': action})
                stage = 'execute'
                execute_start = time.perf_counter()
                observation, reward, done, info = env.step(action)
                metric['execute_seconds'] = time.perf_counter() - execute_start
                totals['execute_seconds'] += metric['execute_seconds']
                save_json(directory / f'{prefix}.result.json', {'reward': reward, 'done': done, 'info': info})
                messages.append({'role': 'assistant', 'content': raw_output})
                metric.update({'wall_seconds': time.perf_counter() - step_start,
                               'finished_at': utc_now(), 'image_count': step - window_start_step + 1,
                               'status': 'completed'})
                append_jsonl(directory / 'steps.jsonl', metric)
                if done:
                    termination = action
                    break
        stage = 'final_observation'
        save_observation(directory, 'final', observation)
        stage = 'evaluate'
        evaluate_start = time.perf_counter()
        with trace_verifier(env, directory):
            score = env.evaluate()
        totals['evaluate_seconds'] = time.perf_counter() - evaluate_start
        summary = {'status': 'completed', 'steps': step, 'termination': termination,
                   'score': score, 'started_at': started_at, 'finished_at': utc_now(),
                   'wall_seconds': time.perf_counter() - started, **totals}
        save_json(directory / 'summary.json', summary)
        return summary
    except BaseException as exc:
        error = {'status': 'cancelled' if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else 'error',
                 'step': step, 'stage': stage, 'error_type': type(exc).__name__,
                 'message': str(exc), 'traceback': traceback.format_exc(),
                 'started_at': started_at, 'finished_at': utc_now(),
                 'wall_seconds': time.perf_counter() - started, **totals}
        if step and 'metric' in locals() and stage in ('observation', 'inference', 'parse', 'execute'):
            append_jsonl(directory / 'steps.jsonl', {**metric, 'status': error['status'], 'failed_stage': stage})
        save_json(directory / 'error.json', error)
        raise
    finally:
        tracing.close()
