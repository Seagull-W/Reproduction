"""Reconstruct an exact JSON request from image references, without network access."""
import argparse
import base64
import json
from pathlib import Path
from artifacts import sha256


def reconstruct(path):
    path = Path(path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    for message in payload['messages']:
        if not isinstance(message['content'], list):
            continue
        for part in message['content']:
            if part['type'] == 'image_url':
                ref = part['image_url']
                image = path.parent / ref['artifact']
                if sha256(image) != ref['sha256']:
                    raise ValueError(f'Image checksum mismatch: {image}')
                part['image_url'] = {'url': 'data:image/png;base64,' + base64.b64encode(image.read_bytes()).decode('ascii')}
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request_file', type=Path)
    parser.add_argument('output_file', type=Path)
    args = parser.parse_args()
    with args.output_file.open('x', encoding='utf-8') as stream:
        json.dump(reconstruct(args.request_file), stream, ensure_ascii=False)
