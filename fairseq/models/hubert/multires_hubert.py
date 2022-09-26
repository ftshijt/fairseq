# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, List, Tuple

import math
import torch
import torch.nn as nn
from omegaconf import II, MISSING

from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, FairseqEncoder, register_model
from fairseq.models.hubert.hubert import MASKING_DISTRIBUTION_CHOICES
from fairseq.models.wav2vec.wav2vec import norm_block
from fairseq.tasks import FairseqTask


import logging


@dataclass
class HubertMultiResAsrConfig(FairseqDataclass):
    # expected input:
    #    w2v_path: a.pth--b.pth--c.pth
    #     label_rate: 5,2,2,3
    #                 (imply (5,2), (2,3))
    w2v_path: str = field(default=MISSING, metadata={"help": "path to hubert model"})
    label_rates: str = field(default=MISSING, metadata={"help": "tuple for label rates e.g., [(5,2), (2,3)]"})
    no_pretrained_weights: bool = field(
        default=False,
        metadata={"help": "if true, does not load pretrained weights"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout after transformer and before final projection"},
    )
    dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability inside hubert model"},
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights " "inside hubert model"
        },
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN " "inside hubert model"
        },
    )

    # fusion
    multi_ctc: bool = field(
        default=False, metadata={"help": "apply multiple CTC for different units"}
    )

    # masking
    apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    mask_length: int = field(
        default=10, metadata={"help": "repeat the mask indices multiple times"}
    )
    mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a token with mask "
            "(normalized by length)"
        },
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    freeze_finetune_updates: int = field(
        default=0,
        metadata={"help": "dont finetune hubert for this many updates"},
    )
    feature_grad_mult: float = field(
        default=0.0,
        metadata={"help": "reset feature grad mult in hubert to this"},
    )
    layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a layer in hubert"},
    )
    normalize: bool = II("task.normalize")
    data: str = II("task.data")

    # this holds the loaded hubert args
    w2v_args: Any = None


@dataclass
class HubertMultiresCtcConfig(HubertMultiResAsrConfig):
    pass


@register_model("hubert_multires_ctc", dataclass=HubertMultiresCtcConfig)
class HubertMultiresCTC(BaseFairseqModel):
    def __init__(self, cfg: HubertMultiResAsrConfig, w2v_encoder: BaseFairseqModel):
        super().__init__()
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: HubertMultiResAsrConfig, task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = MultiResHubertEncoder(cfg, task)
        return cls(cfg, w2v_encoder)

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["encoder_out"]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def get_logits(self, net_output):
        logits = net_output["encoder_out"]
        padding = net_output["encoder_padding_mask"]
        if padding is not None and padding.any():
            padding = padding.T
            logits[padding][..., 0] = 0
            logits[padding][..., 1:] = float("-inf")

        return logits

    def forward(self, **kwargs):
        x = self.w2v_encoder(**kwargs)
        return x


class MultiResHubertEncoder(FairseqEncoder):
    def __init__(self, cfg: HubertMultiResAsrConfig, task):
        super(MultiResHubertEncoder, self).__init__(None) # set None for dictionary
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }

        if cfg.w2v_args is None:
            w2v_args = []
            states = []
            w2v_path = cfg.w2v_path.split("--")
            for single_w2v_path in w2v_path:
                state = checkpoint_utils.load_checkpoint_to_cpu(single_w2v_path, arg_overrides)
                states.append(state)
                single_w2v_args = state.get("cfg", None)
                if single_w2v_args is None:
                    single_w2v_args = convert_namespace_to_omegaconf(state["args"])
                w2v_args.append(single_w2v_args)
            cfg.w2v_args = w2v_args # for some common info
        else:
            raise NotImplementedError("Not implemented")

        self.w2v_model = nn.ModuleList()
        if cfg.label_rates == "None":
            self.label_rates = None
        else:
            self.label_rates = []
            label_rates = cfg.label_rates.split(",")
            for i in range(len(label_rates) // 2):
                self.label_rates.append((int(label_rates[i * 2]), int(label_rates[i * 2 + 1])))
        self.res_num = len(w2v_args)
        ds = []
        for i in range(self.res_num):
            single_w2v_args = w2v_args[i]
            state = states[i]
            assert cfg.normalize == single_w2v_args.task.normalize, (
                "Fine-tuning works best when data normalization is the same. "
                "Please check that --normalize is set or unset for "
                "both pre-training and here"
            )

            single_w2v_args.task.data = cfg.data
            pretrain_task = tasks.setup_task(single_w2v_args.task)
            if state is not None and "task_state" in state:
                # This will load the stored "dictionaries" object
                pretrain_task.load_state_dict(state["task_state"])
            else:
                pretrain_task.load_state_dict(task.state_dict())

            model = pretrain_task.build_model(single_w2v_args.model, from_checkpoint=True)
            if state is not None and not cfg.no_pretrained_weights:
                # set strict=False because we omit some modules
                model.load_state_dict(state["model"], strict=False)

            model.remove_pretraining_modules()

            # super().__init__(pretrain_task.source_dictionary)

            self.w2v_model.append(model)
            ds.append(single_w2v_args.model.encoder_embed_dim)

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        # Common process size
        COMMON_DIM = 512

        # project different representation dim to common space
        self.proj_common = nn.ModuleList()
        # project common space to prediction
        self.proj_pred = nn.ModuleList()
        for d in ds:
            self.proj_common.append(Linear(d, COMMON_DIM))
            if task.target_dictionary is not None:
                self.proj_pred.append(Linear(COMMON_DIM, len(task.target_dictionary)))
            # elif getattr(cfg, "decoder_embed_dim", d) != d:
            #     self.proj = Linear(d, cfg.decoder_embed_dim)
            else:
                raise NotImplementedError("Need target directionary to be not NOne")
                # self.proj = None
        
        # default activation as gelu
        activation = nn.GELU()
        
        # conv module for each w2v model
        self.conv_refine = nn.ModuleList()
        for i in range(self.res_num):
            conv_refiner = ConvResidual(
                start_dim=512,
                conv_layers=[(512, 3, 1), (512, 5, 1)],
                dropout=cfg.dropout,
                activation=activation,
            )
            self.conv_refine.append(conv_refiner)
        
        # conv module for combine each w2v model
        self.conv_combinor = nn.ModuleList()
        logging.info(self.label_rates)
        for i in range(self.res_num - 1):
            conv_combinor = ConvCombinor(
                k = 5, 
                label_rate=self.label_rates[i], 
                dropout=cfg.dropout, 
                channels=COMMON_DIM,
                activation=activation
            )
            self.conv_combinor.append(conv_combinor)



    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, tbc=True, **kwargs):

        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            x = []
            padding_mask = []
            for i in range(self.res_num):
                single_x, single_padding_mask = self.w2v_model[i].extract_features(**w2v_args)
                x.append(single_x)
                padding_mask.append(single_padding_mask)


        
        projected_x = []
        for i in range(self.res_num):
            # logging.info("x_shape {} , padding_mask_shape {}".format(x[i].shape, padding_mask[i].shape))
            common_projected = self.proj_common[i](x[i]) # projecting into common space
            projected_x.append(self.conv_refine[i](common_projected)) # conv refine
        
        # fuse the representation if have multiple projected features
        if len(projected_x) > 1:
            running_x = projected_x[0]
            running_padding = padding_mask[0]
            for i in range(self.res_num - 1):
                x2 = projected_x[i + 1]
                padding2 = padding_mask[i + 1]
                running_x, running_padding = self.conv_combinor[i](running_x, x2, padding1=running_padding, padding2=padding2)
            x = running_x
            padding_mask = running_padding
        else:
            x = projected_x[0]
            padding_mask = padding_mask[0]

        if tbc:
            # B x T x C -> T x B x C
            x = x.transpose(0, 1)
        x = self.final_dropout(x)

        x = self.proj_pred[-1](x)

        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
                1, new_order
            )
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class ConvResidual(nn.Module):
    def __init__(
        self,
        start_dim,
        conv_layers,
        dropout,
        activation,
        log_compression = False,
        skip_connections = True,
        residual_scale = 0.5,
        non_affine_group_norm = False,
    ):
        super().__init__()

        def block(n_in, n_out, k, stride):
            return nn.Sequential(
                # with padding (k - 1) // 2 to keep the same size
                nn.Conv1d(n_in, n_out, k, stride=stride, bias=False, padding=(k - 1) // 2),
                nn.Dropout(p=dropout),
                norm_block(
                    is_layer_norm=False, dim=n_out, affine=not non_affine_group_norm
                ),
                activation,
            )

        in_d = start_dim
        self.conv_layers = nn.ModuleList()
        for dim, k, stride in conv_layers:
            self.conv_layers.append(block(in_d, dim, k, stride))
            in_d = dim

        self.log_compression = log_compression
        self.skip_connections = skip_connections
        self.residual_scale = math.sqrt(residual_scale)

    def forward(self, x):
        # assume X: (B, T, C)
        x = x.permute(0, 2, 1)

        for conv in self.conv_layers:
            residual = x
            x = conv(x)
            if self.skip_connections and x.size(1) == residual.size(1):
                tsz = x.size(2)
                r_tsz = residual.size(2)
                residual = residual[..., :: r_tsz // tsz][..., :tsz]
                x = (x + residual) * self.residual_scale

        if self.log_compression:
            x = x.abs()
            x = x + 1
            x = x.log()
        
        x = x.permute(0, 2, 1)
        return x


class ConvCombinor(nn.Module):
    def __init__(
        self,
        k,
        label_rate,
        dropout,
        channels,
        activation,
        log_compression = False,
        skip_connections = True,
        residual_multi = False,
        residual_scale = 0.4,
        non_affine_group_norm = False,
        cat = True,
    ):
        super().__init__()

        def block(channel, k, stride):
            return nn.Sequential(
                # with padding (k - 1) // 2 to keep the same size
                nn.ConvTranspose1d(
                    channel, 
                    channel, 
                    k, 
                    stride=stride, 
                    bias=False, 
                    padding=0, # padding=(k - 1) // 2, 
                    output_padding=(stride - 1)
                ),
                nn.Dropout(p=dropout),
                norm_block(
                    is_layer_norm=False, dim=channel, affine=not non_affine_group_norm
                ),
                activation,
            )
        
        assert len(label_rate) == 2, "label_rate should be sized two to apply fusion"
        # Lout =(Lin−1)×stride−2×padding+dilation×(kernel_size−1)+output_padding+1
        self.conv_layers_res1 = block(channels, k, label_rate[0])
        self.conv_layers_res2 = block(channels, k, label_rate[1])

        self.label_rate = label_rate
        self.log_compression = log_compression
        self.skip_connections = skip_connections
        self.residual_multi = residual_multi
        self.residual_scale = math.sqrt(residual_scale)
        self.cat = cat
        if self.cat:
            self.fuse = nn.Linear(3 * channels, channels)

    def forward(self, x1, x2, padding1=None, padding2=None):
        x1, x2 = x1.permute(0, 2, 1), x2.permute(0, 2, 1)
        residual1, residual2 = x1, x2
        x1 = self.conv_layers_res1(x1)
        x2 = self.conv_layers_res2(x2)
        size1, size2 = x1.size(2), x2.size(2)
        if self.skip_connections:
            residual = torch.repeat_interleave(residual2, self.label_rate[1], dim=2)
            final_size = min(size1, size2, residual.size(2), residual1.size(2) * self.label_rate[0])
            if self.residual_multi:
                temp_residual = torch.repeat_interleave(residual1, self.label_rate[0], dim=2)
                # final_size = min(final_size, temp_residual.size(2))
                residual = residual[..., :final_size] + temp_residual[..., :final_size]
            if self.cat:
                x = torch.cat((x1[..., :final_size], x2[..., :final_size], residual[..., :final_size]), dim=1)
                x = x.permute(0, 2, 1)
                x = self.fuse(x)
                x = x.permute(0, 2, 1)
            else:
                x = (x1[..., :final_size] + x2[..., :final_size] + residual[..., :final_size]) * self.residual_scale
        else:
            final_size = min(size1, size2)
            x = (x1[..., :final_size] + x2[..., :final_size]) * self.residual_scale

        if self.log_compression:
            x = x.abs()
            x = x + 1
            x = x.log()

        x = x.permute(0, 2, 1)
        
        # process padding
        if padding1 is not None:
            padding1 = torch.repeat_interleave(padding1, self.label_rate[0], dim=1)
        if padding2 is not None:
            padding2 = torch.repeat_interleave(padding2, self.label_rate[1], dim=1)
        if padding1 is not None and padding2 is not None:
            padding = torch.logical_or(padding1[..., :final_size], padding2[..., :final_size])
        else:
            padding = padding1[..., :final_size] if padding1 is not None else padding2[..., :final_size]
        return x, padding