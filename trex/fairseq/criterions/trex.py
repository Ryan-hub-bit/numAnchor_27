# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import metrics, modules, utils
from command import configs
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion("trex")
class TrexLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, task, tpu=False):
        super().__init__(task)
        self.tpu = tpu

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        masked_code = sample["target"][configs.static_field].ne(self.padding_idx_dict[configs.static_field])
        masked_value = sample["target"][configs.byte_fields[0]].ne(1)

        # Some adapted datasets do not provide value targets for every batch.
        # Keep all-False masks so the LM heads return empty tensors and the
        # corresponding loss cleanly becomes zero.
        if self.tpu:
            masked_code = None
            masked_value = None

        output = model(**sample["net_input"], masked_code=masked_code, masked_value=masked_value)[0]

        pred_logits_code, pred_value = output['code'], output['value']
        targets_code = sample["target"][configs.static_field]

        if masked_code is not None:
            targets_code = targets_code[masked_code]

        if masked_value is not None:
            targets_value_stacked = torch.stack(
                [sample["target"][field][masked_value] for field in configs.byte_fields], dim=1)

        sample_size_code = masked_code.int().sum() if masked_code is not None else targets_code.numel()
        sample_size_value = masked_value.int().sum() * configs.byte_len if masked_value is not None else 0
        sample_size = sample_size_code + sample_size_value

        code_loss = modules.cross_entropy(
            pred_logits_code.view(-1, pred_logits_code.size(-1)),
            targets_code.view(-1),
            reduction="sum",
            ignore_index=self.padding_idx_dict[configs.static_field],
        )

        if masked_value is not None:
            value_loss = F.mse_loss(
                pred_value.float(),
                targets_value_stacked.float(),
                reduction='sum'
            )
        else:
            value_loss = code_loss.new_zeros(())

        loss = code_loss + configs.code_value_loss_alpha * value_loss

        logging_output = {
            "loss": loss.data,
            'code_loss': code_loss.data,
            'value_loss': value_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
            "sample_size_code": sample_size_code,
            "sample_size_value": sample_size_value,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        code_loss_sum = sum(log.get("code_loss", 0) for log in logging_outputs)
        value_loss_sum = sum(log.get("value_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        sample_size_code = sum(log.get("sample_size_code", 0) for log in logging_outputs)
        sample_size_value = sum(log.get("sample_size_value", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / max(sample_size, 1), sample_size, round=3)
        if sample_size_code > 0:
            metrics.log_scalar("code_loss", code_loss_sum / sample_size_code / math.log(2), sample_size_code, round=3)
            metrics.log_derived("code_ppl", lambda meters: utils.get_perplexity(meters["code_loss"].avg))
        else:
            metrics.log_scalar("code_loss", 0.0, 0, round=3)
        if sample_size_value > 0:
            metrics.log_scalar("value_loss_mse", value_loss_sum / sample_size_value, sample_size_value, round=3)
        else:
            metrics.log_scalar("value_loss_mse", 0.0, 0, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
