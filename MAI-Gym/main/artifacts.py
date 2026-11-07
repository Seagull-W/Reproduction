"""Small shared helpers for durable, reconstructable evaluation evidence."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_default(value):
    # OSWorld evaluators can return numpy scalars/arrays.
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError(f'Cannot serialize {type(value).__name__}')


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    default=json_default, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def append_jsonl(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=json_default,
                                allow_nan=False) + '\n')
        stream.flush()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    """Keep completed records after an interrupted final write; report that tail."""
    path = Path(path)
    if not path.exists():
        return [], []
    rows, errors = [], []
    with path.open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append({'file': str(path), 'line': line_number, 'error': str(exc)})
    return rows, errors
