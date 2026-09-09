# Upstream references

- OvisOCR2 Technical Report, https://arxiv.org/html/2607.13639v1 : design inspiration for learnable hard cases and weighted model fusion. This project does not reproduce its GRPO/OPD training.
- PaddlePaddle/PaddleX, `paddlex/inference/pipelines/paddleocr_vl/uilts.py`, Apache-2.0: reference OTSL token semantics. The strict codec here is independently implemented; it does not copy the permissive grid repair routine.
- IBM PubTabNet, `src/metric.py`, copyright 2020 IBM, author peter.zhong@au1.ibm.com, Apache-2.0: TEDS tree construction, rename cost and descendant-count normalization inform the adapted metric implementation in `dataflywheel/metrics.py`.
- modelscope/ms-swift, Apache-2.0: model registration and DPO template/CLI APIs. In the inspected 4.x implementation, PaddleOCR-VL-1.6 uses `paddleocr_vl` and `paddle_ocr_1_5`, with native Transformers >=5.0.
- `dataflywheel/paddle_compat.py` adapts `PaddleOCR1_5Template._post_encode` from modelscope/ms-swift `swift/template/templates/baidu.py`, copyright ModelScope Contributors, Apache-2.0. Modified to concatenate per-image feature tuples before dtype/device conversion; source: https://github.com/modelscope/ms-swift/blob/main/swift/template/templates/baidu.py .
- opendatalab/OmniDocBench: CDM is loaded from a separately installed upstream checkout, not vendored. Preserve its license and pin that checkout separately. See https://github.com/opendatalab/OmniDocBench .

The files downloaded for source inspection are not bundled in this repository. Source hashes and inspected URLs are recorded in `docs/upstream-inspection.json`.
