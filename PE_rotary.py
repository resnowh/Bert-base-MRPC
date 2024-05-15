# PE_rotary.py
from venv import logger

import torch
import math

from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions


def rotate_every_two(x):
    x1 = x[:, :, 0::2]
    x2 = x[:, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return x


def apply_rotary_position_embeddings(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    x = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return x


def convert_to_rotary_position_embedding(model):
    max_length = model.config.max_position_embeddings
    embedding_dim = model.config.hidden_size

    # 生成旋转位置编码的正弦和余弦部分
    position_ids = torch.arange(max_length, dtype=torch.float32)
    freqs = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float32) * -(math.log(10000.0) / embedding_dim))
    args = position_ids[:, None] * freqs[None, :]
    cos_embedding = torch.cos(args)
    sin_embedding = torch.sin(args)

    # 将位置编码的正弦和余弦部分注册为BertEncoder的缓冲区
    for encoder_layer in model.bert.encoder.layer:
        encoder_layer.register_buffer('cos_embedding', cos_embedding)
        encoder_layer.register_buffer('sin_embedding', sin_embedding)

    # 修改模型的前向传播函数,在计算注意力之前应用RoPE
    def new_forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None, use_cache=None, output_attentions=False, output_hidden_states=False, return_dict=True):
        for i, layer_module in enumerate(self.layer):
            # 获取序列长度
            seq_length = hidden_states.shape[1]

            # 应用RoPE
            cos = layer_module.cos_embedding[:seq_length, :]
            sin = layer_module.sin_embedding[:seq_length, :]
            hidden_states = apply_rotary_position_embeddings(hidden_states, cos, sin)

        # 继续原有的前向传播代码
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

    model.bert.encoder.forward = new_forward.__get__(model.bert.encoder, model.bert.encoder.__class__)

    # 冻结位置编码权重,确保在训练过程中不会被更新
    model.bert.embeddings.position_embeddings.weight.requires_grad = False

    return model
