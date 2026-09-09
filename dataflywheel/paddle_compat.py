# Copyright (c) ModelScope Contributors. All rights reserved.
# Modified for DataFlywheel: normalize Tensor/tuple image features before casting.
"""PaddleOCR ms-swift/Transformers image-feature return-type compatibility.

Matches the PaddleOCR1_5Template _post_encode flow, with tensor/tuple handling.
Upstream: modelscope/ms-swift swift/template/templates/baidu.py (Apache-2.0).
Transformers splits pooler_output by image; restore token order with cat(dim=0).
"""
import logging


def image_feature_tensor(features):
    import torch
    if isinstance(features, torch.Tensor):
        return features
    if isinstance(features, (tuple, list)) and features and all(isinstance(x, torch.Tensor) for x in features):
        if any(x.ndim != 2 for x in features):
            raise ValueError('PaddleOCR image feature chunks must have shape [tokens, hidden]')
        return torch.cat(features, dim=0)
    raise TypeError('PaddleOCR pooler_output must be a Tensor or nonempty sequence of Tensors')


def post_encode(self, model, inputs):
    if not self.is_training:
        return inputs
    base_model = self.get_base_model(model)
    input_ids = inputs['input_ids']
    pixel_values = inputs.pop('pixel_values')
    image_grid_thw = inputs.get('image_grid_thw')
    inputs_embeds = base_model.model.language_model.embed_tokens(input_ids)
    if pixel_values is not None:
        output = base_model.model.get_image_features(pixel_values, image_grid_thw, return_dict=True)
        image_embeds = image_feature_tensor(output.pooler_output)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask = base_model.model.get_placeholder_mask(input_ids, inputs_embeds, image_features=image_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
    return {'inputs_embeds': inputs_embeds}


def install_patch():
    from swift.template.templates.baidu import PaddleOCR1_5Template
    if getattr(PaddleOCR1_5Template._post_encode, '_dataflywheel_compat', False):
        return
    post_encode._dataflywheel_compat = True
    PaddleOCR1_5Template._post_encode = post_encode
    logging.getLogger(__name__).warning('DataFlywheel: PaddleOCR Tensor/tuple image-feature compatibility enabled')
