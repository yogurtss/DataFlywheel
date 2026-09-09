"""Diagnose the actual training Python/CUDA environment without loading models."""
import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dtype', choices=['bfloat16', 'float16', 'float32'], default='bfloat16')
    parser.add_argument('--expected-devices', type=int, default=1)
    args = parser.parse_args()
    try:
        import torch
    except ImportError:
        print('PyTorch is not installed in ' + sys.executable)
        return 1
    report = {'python': sys.executable, 'torch': torch.__version__, 'cuda_build': torch.version.cuda,
              'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'cuda_available': torch.cuda.is_available(), 'devices': [], 'errors': []}
    if not report['cuda_available']:
        report['errors'].append('CUDA unavailable: check CUDA-enabled PyTorch, NVIDIA driver, GPU visibility, and Docker --gpus all. Changing dtype does not fix missing CUDA.')
    else:
        count = torch.cuda.device_count()
        if count != args.expected_devices:
            report['errors'].append(f'Expected {args.expected_devices} visible GPUs, found {count}; check training.gpus / CUDA_VISIBLE_DEVICES.')
        for i in range(count):
            try:
                with torch.cuda.device(i):
                    try:
                        native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
                    except TypeError:  # Older PyTorch has no including_emulation keyword.
                        native_bf16 = torch.cuda.is_bf16_supported() and (bool(torch.version.hip) or torch.cuda.get_device_capability(i)[0] >= 8)
                    report['devices'].append({'index': i, 'name': torch.cuda.get_device_name(i),
                                              'capability': list(torch.cuda.get_device_capability(i)),
                                              'native_bf16': native_bf16})
                    if args.dtype == 'bfloat16' and not native_bf16:
                        report['errors'].append(f'GPU {i} lacks native BF16 support. Select training.torch_dtype: float32 for diagnosis, or explicitly test float16 training.')
                    torch.ones(1, device=f'cuda:{i}', dtype=getattr(torch, args.dtype))
                    torch.cuda.synchronize()
            except Exception as error:
                report['errors'].append(f'GPU {i}: {type(error).__name__}: {error}')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(bool(report['errors']))


if __name__ == '__main__':
    raise SystemExit(main())
