from types import SimpleNamespace, ModuleType
import sys

import pytest
import torch

from dataflywheel.paddle_compat import image_feature_tensor, install_patch, post_encode
from dataflywheel.training import train_command


@pytest.mark.parametrize('sequence', [tuple, list])
def test_all_images_keep_order_and_gradients(sequence):
    first = torch.arange(6.).reshape(2, 3).requires_grad_()
    second = torch.arange(9.).reshape(3, 3).requires_grad_()
    merged = image_feature_tensor(sequence([first, second]))
    assert torch.equal(merged[:2], first) and torch.equal(merged[2:], second)
    merged.sum().backward()
    assert torch.equal(first.grad, torch.ones_like(first))
    assert torch.equal(second.grad, torch.ones_like(second))


def test_tensor_passthrough_and_invalid_returns():
    value = torch.ones(2, 3)
    assert image_feature_tensor(value) is value
    for invalid in ((), None, (value, 'bad')):
        with pytest.raises(TypeError):
            image_feature_tensor(invalid)
    with pytest.raises(ValueError, match='shape'):
        image_feature_tensor((torch.ones(1, 2, 3),))


@pytest.mark.parametrize('as_tuple', [False, True])
def test_post_encode_scatter_and_visual_backprop(as_tuple):
    features = torch.arange(12.).reshape(4, 3).requires_grad_()
    embedding = torch.nn.Embedding(3, 3)
    ids = torch.tensor([[0, 1, 1], [0, 1, 1]])
    def get_features(pixels, grid, return_dict):
        assert return_dict is True
        return SimpleNamespace(pooler_output=features.split(2) if as_tuple else features)
    def get_mask(input_ids, inputs_embeds, image_features):
        assert image_features.shape == (4, 3)
        mask = (input_ids == 1).unsqueeze(-1)
        assert mask.sum() * 3 == image_features.numel()
        return mask
    model = SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(embed_tokens=embedding),
                            get_image_features=get_features, get_placeholder_mask=get_mask))
    template = SimpleNamespace(is_training=True, get_base_model=lambda x: x)
    result = post_encode(template, model, {'input_ids': ids, 'pixel_values': torch.ones(1), 'image_grid_thw': None})
    output = result['inputs_embeds']
    assert torch.equal(output[ids == 1], features)
    output.sum().backward()
    assert features.grad is not None and torch.equal(features.grad, torch.ones_like(features))
    assert embedding.weight.grad[0].sum() > 0


def test_inference_untouched_and_patch_idempotent(monkeypatch):
    inputs = {'input_ids': 'unchanged'}
    assert post_encode(SimpleNamespace(is_training=False), None, inputs) is inputs
    module = ModuleType('swift.template.templates.baidu')
    class Template:
        def _post_encode(self, model, inputs):
            return inputs
    module.PaddleOCR1_5Template = Template
    monkeypatch.setitem(sys.modules, module.__name__, module)
    install_patch()
    method = Template._post_encode
    install_patch()
    assert Template._post_encode is method


def test_plugin_in_single_and_ddp_launch(config):
    config['training']['model'] = '/sft'
    for gpus, count in [('0', 1), ('0,1', 2)]:
        config['training'].update(gpus=gpus, nproc_per_node=count)
        cmd, _ = train_command(config, 'dpo.jsonl', 'output')
        assert cmd[cmd.index('--external_plugins')+1].endswith('/scripts/swift_paddle_compat.py')
    config['training']['paddle_feature_compat'] = False
    assert '--external_plugins' not in train_command(config, 'dpo.jsonl', 'output')[0]
