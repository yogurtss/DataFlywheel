"""ERNIEKit PaddleOCR-VL single-region SFT adapter."""
from pathlib import Path
import os
import tempfile

import httpx
from PIL import Image

from .io import digest

PROMPT_TASK = {'OCR:': 'text', 'Table Recognition:': 'table', 'Formula Recognition:': 'formula'}


def adapt_sft(row):
    if 'image_info' not in row and 'text_info' not in row:
        return dict(row)
    images, texts = row.get('image_info'), row.get('text_info')
    if row.get('is_system') or row.get('video_info') or row.get('tools'):
        raise ValueError('unsupported SFT system/video/tool context')
    if not isinstance(images, list) or len(images) != 1 or not isinstance(texts, list) or len(texts) != 2:
        raise ValueError('SFT requires one crop image and one mask/no_mask turn; multi-image/multi-turn unsupported')
    if not all(isinstance(x, dict) for x in [*images, *texts]):
        raise ValueError('invalid SFT image_info/text_info entries')
    if texts[0].get('tag') != 'mask' or texts[1].get('tag') != 'no_mask':
        raise ValueError('SFT text_info must be mask query followed by no_mask answer')
    if type(images[0].get('matched_text_index', 0)) is not int or images[0].get('matched_text_index', 0) != 0:
        raise ValueError('SFT image must match query at text_info index 0')
    prompt, answer = texts[0].get('text'), texts[1].get('text')
    if not isinstance(prompt, str) or prompt.strip() not in PROMPT_TASK:
        raise ValueError('unsupported SFT task prompt; expected OCR:, Table Recognition:, or Formula Recognition:')
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError('SFT no_mask answer must be a nonempty string')
    image = images[0].get('image_url')
    if not isinstance(image, str) or not image:
        raise ValueError('SFT image_url must be a nonempty string')
    task = PROMPT_TASK[prompt.strip()]
    fmt = ('html' if answer.lstrip().startswith('<table') else 'otsl') if task == 'table' else ('latex' if task == 'formula' else 'text')
    for key, value in [('task', task), ('image', image), ('gt', answer), ('gt_format', fmt)]:
        if key in row and row[key] is not None and row[key] != value:
            raise ValueError(f'SFT {key} conflicts with explicit annotation')
    return {**row, 'task': task, 'image': image, 'gt': answer, 'gt_format': fmt,
            'input_format': 'erniekit_sft', 'sft_prompt': prompt,
            'task_detection': 'official_prompt', 'original_image_url': image}


def resolve_image(value, base, options):
    if not value.startswith(('http://', 'https://')):
        path = Path(value).expanduser()
        root = Path(options['image_root']).expanduser().resolve() if options.get('image_root') else base
        return (root / path).resolve()
    cache = Path(options.get('image_cache', 'runs/image-cache')).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (digest(value) + '.image')
    if path.exists():
        with Image.open(path) as im:
            im.verify()
        return path
    fd, name = tempfile.mkstemp(dir=cache, suffix='.part')
    try:
        with os.fdopen(fd, 'wb') as out:
            with httpx.stream('GET', value, timeout=options.get('image_timeout', 60), follow_redirects=True) as response:
                response.raise_for_status()
                size = 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > options.get('max_image_bytes', 50_000_000):
                        raise ValueError('remote image exceeds data.max_image_bytes')
                    out.write(chunk)
        with Image.open(name) as im:
            im.verify()
        os.replace(name, path)
    except httpx.HTTPError:
        raise ValueError('SFT remote image download failed') from None
    finally:
        Path(name).unlink(missing_ok=True)
    return path
