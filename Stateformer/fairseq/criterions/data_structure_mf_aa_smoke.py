# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion


PAD_LABEL_ID = -100


@register_criterion("data_structure_mf_aa_smoke")
class DataStructureMFAASmokeCriterion(FairseqCriterion):
    def __init__(self, task, classification_head_name):
        super().__init__(task)
        self.classification_head_name = classification_head_name

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--classification-head-name",
            default="data_structure_head",
            help="name of the classification head to use",
        )
        parser.add_argument(
            "--focal-gamma",
            type=float,
            default=0.0,
            help="optional focal-loss gamma for downstream token classification; 0 disables focal scaling",
        )

    def forward(self, model, sample, reduce=True):
        assert (
            hasattr(model, "classification_heads")
            and self.classification_head_name in model.classification_heads
        ), "model must provide a token classification head for --criterion=data_structure_mf_aa_smoke"

        real_tokens = sample["target"].ne(PAD_LABEL_ID)
        logits, _ = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.classification_head_name,
        )
        targets = sample["target"][real_tokens].view(-1)
        sample_size = real_tokens.int().sum().float()

        weight = None
        if getattr(self.task, "class_weights", None) is not None:
            weight = self.task.class_weights.to(logits.device)
        token_logits = logits[real_tokens, :]
        per_token_loss = F.cross_entropy(
            token_logits,
            targets,
            reduction="none",
            weight=weight,
        )
        focal_gamma = float(getattr(self.task.args, "focal_gamma", 0.0))
        if focal_gamma > 0:
            probs = torch.softmax(token_logits, dim=-1)
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
            per_token_loss = ((1.0 - pt) ** focal_gamma) * per_token_loss
        loss = per_token_loss.sum()

        preds = token_logits.argmax(dim=1)
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample_size,
            "sample_size": sample_size,
            "ncorrect_total": (preds == targets).sum(),
            "ncorrect": ((preds == targets) * (targets != self.task.no_access_idx)).sum(),
            "ntype": (targets != self.task.no_access_idx).sum().item(),
            "ntype_pred": (preds != self.task.no_access_idx).sum().item(),
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs):
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar("nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3)

        if len(logging_outputs) > 0 and "ncorrect_total" in logging_outputs[0]:
            ncorrect_total = sum(log.get("ncorrect_total", 0) for log in logging_outputs)
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            ntype = sum(log.get("ntype", 0) for log in logging_outputs)
            ntype_pred = sum(log.get("ntype_pred", 0) for log in logging_outputs)

            precision = ncorrect / (ntype_pred + 1e-5)
            recall = ncorrect / (ntype + 1e-5)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-5)
            metrics.log_scalar("precision", 100.0 * precision, sample_size, round=1)
            metrics.log_scalar("recall", 100.0 * recall, sample_size, round=1)
            metrics.log_scalar("F1", 100.0 * f1, sample_size, round=1)
            metrics.log_scalar("accuracy", 100.0 * ncorrect_total / sample_size, sample_size, round=1)

    @staticmethod
    def logging_outputs_can_be_summed():
        return True
