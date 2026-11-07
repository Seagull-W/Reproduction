"""Capture OSWorld getter/metric results without changing their return values."""
import functools
from contextlib import contextmanager
from pathlib import Path
from artifacts import append_jsonl, save_json, utc_now


@contextmanager
def trace_verifier(env, directory):
    directory = Path(directory)
    originals = {}
    counter = 0
    def wrap(function, name):
        @functools.wraps(function)
        def traced(*args, **kwargs):
            nonlocal counter
            counter += 1
            call_id = counter
            try:
                result = function(*args, **kwargs)
            except BaseException as exc:
                append_jsonl(directory / 'verifier.jsonl', {'call': call_id, 'name': name,
                    'timestamp': utc_now(), 'status': 'error', 'error_type': type(exc).__name__, 'message': str(exc)})
                raise
            filename = f'verifier_{call_id:04d}'
            try:
                if isinstance(result, bytes):
                    filename += '.bin'
                    (directory / filename).write_bytes(result)
                else:
                    filename += '.json'
                    save_json(directory / filename, result)
                evidence = {'artifact': filename}
            except TypeError:
                # Do not change scoring because a custom getter returned an opaque object.
                evidence = {'unserialized_type': type(result).__name__, 'repr': repr(result)}
            append_jsonl(directory / 'verifier.jsonl', {'call': call_id, 'name': name,
                'timestamp': utc_now(), 'status': 'completed', **evidence})
            return result
        return traced
    for attribute in ('result_getter', 'expected_getter', 'metric'):
        if not hasattr(env, attribute):
            continue
        value = getattr(env, attribute)
        originals[attribute] = value
        if isinstance(value, list):
            setattr(env, attribute, [wrap(fn, f'{attribute}[{i}]') if callable(fn) else fn for i, fn in enumerate(value)])
        elif callable(value):
            setattr(env, attribute, wrap(value, attribute))
    try:
        yield
    finally:
        for attribute, value in originals.items():
            setattr(env, attribute, value)


@contextmanager
def trace_controller(env, directory):
    """OSWorld step() may discard controller return values; preserve them here."""
    import time
    directory = Path(directory)
    controller = getattr(env, 'controller', None)
    originals = {}
    counter = 0
    def wrap(function, name):
        @functools.wraps(function)
        def traced(*args, **kwargs):
            nonlocal counter
            counter += 1
            call_id = counter
            started = time.perf_counter()
            record = {'call': call_id, 'name': name, 'timestamp': utc_now(),
                      'args': args, 'kwargs': kwargs}
            try:
                result = function(*args, **kwargs)
            except BaseException as exc:
                append_jsonl(directory / 'controller.jsonl', {**record, 'status': 'error',
                    'seconds': time.perf_counter() - started,
                    'error_type': type(exc).__name__, 'message': str(exc)})
                raise
            if isinstance(result, bytes):
                name_on_disk = f'controller_{call_id:04d}.bin'
                (directory / name_on_disk).write_bytes(result)
                recorded_result = {'artifact': name_on_disk}
            else:
                recorded_result = result
            append_jsonl(directory / 'controller.jsonl', {**record, 'status': 'completed',
                'seconds': time.perf_counter() - started, 'result': recorded_result})
            return result
        return traced
    if controller is not None:
        for name in ('execute_python_command', 'execute_action', 'execute_command', 'get_file'):
            method = getattr(controller, name, None)
            if callable(method):
                originals[name] = method
                setattr(controller, name, wrap(method, name))
    try:
        yield
    finally:
        for name, method in originals.items():
            setattr(controller, name, method)
