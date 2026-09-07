# 验证边界

本仓库的本地验证覆盖真实 Python 数据处理、HTTP 客户端的模拟传输、CPU safetensors 运算；模拟输出不表示真实模型效果。

已通过 43 项 pytest 测试。另直接运行下载的官方转换/指标定义，完成 3 个 PaddleX OTSL 转换对照（含实体字符、空格和联合跨度）及 8 个 IBM TEDS/TEDS-S 数值对照，结果分别保存在 `otsl-reference-validation.json`、`teds-reference-validation.json`。离线完整演示生成三类偏好对，检查了 HTML 本地资源链接和 MathML 输出。

当前环境：CPU 可用，CUDA 不可用；没有用户的原模型/SFT checkpoint、推理端口、ms-swift 训练环境和 TeX/CDM 环境。因此以下硬件集成不会被标记为已验证：

- PaddleOCR-VL-1.6 真实服务的 processor/chat-template 一致性。
- 官方 CDM 固定公式的真实渲染分数。
- ms-swift 三任务真实模板预处理与单卡/多卡短训练。
- 官方 PaddleOCR 完整页级推理及 OmniDocBench Docker 评分。
- 融合后的真实 PaddleOCR-VL 在 Transformers 中重新加载并推理。

这些入口已实现：CDM 有依赖和自匹配检查；训练有真实模板长度预检查；页级导出使用官方 save 方法；融合验证分片索引。测试中分别模拟外部接口，实际验证核心调用数据与失败处理。

真实环境验收顺序：

1. `convert` 三类真实样本，检查表格 HTML/OTSL 往返和图片。
2. `infer` 各请求一次，检查任务输出、服务端动态分辨率和模板；随后 `score` 在 CDM 环境运行。
3. `build-dpo` 后执行 `train --preflight-only`，确认三类都有非空两侧 loss token。
4. 用小数据集和缩短 epoch 的训练配置分别运行单卡与多卡；确认日志 loss 有限、checkpoint 可重新加载。训练成功才产生 `requirements.verified.txt`。
5. 用同一页级产线分别导出 base/SFT/DPO/融合模型；自行在固定版本 OmniDocBench Docker 下比较。

核心依赖版本见 `requirements-core.lock`，测试结果与环境见 `docs/validation-local.json`。依赖锁只覆盖实际验证范围；未声称上述真实集成已通过。

## 合成管线增量验证

新增 GT 驱动表格合成后，完整测试共 49 项通过。使用真实 Playwright Chromium 和 Noto CJK 字体渲染 4 个模板家族、每家族 2 个变体，共 8 张裁剪图与对应整页。8 张均通过字体覆盖、DOM 文本、溢出、HTML/OTSL 往返、图片校验及现有 `convert` 导入检查。已目视检查总览和原分辨率窄列换行样本。详见 `synthesis-validation.json`。

示例产物位于 `runs/synthesis/preview-v3/index.html`（运行产物不纳入源码包）。这里未使用真实用户 hardcase 或 VLM 生成模板，也未测量 PaddleOCR-VL 识别难度、DPO 收益。

## VLM Agent 增量验证

当前完整测试共 **57 项通过**。新增 OpenAI 兼容 URL/model/Bearer token、PNG 图片消息、截断拒绝、凭据脱敏、视觉修正反馈、模板扩展请求、评测文档隔离及样式字段校验。

使用真实 Chromium 与 `httpx.MockTransport` 模拟 VLM，完成 6 次协议请求、2 张入选样本和 1 次看图反馈修正。两张均通过 `synth-check` 与 `convert`，目视检查修正后表格预览。脚本为 `scripts/smoke_synthesis_agent.py`，记录见 `synthesis-agent-validation.json`。没有发送真实供应商请求，真实 VLM 模板生成质量与 hardcase 风格还原尚未验证。
