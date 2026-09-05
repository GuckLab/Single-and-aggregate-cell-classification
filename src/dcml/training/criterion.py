import logging

import torch
from torch.nn.functional import cross_entropy
from torch import Tensor
from typing import Optional, List
from torch.nn.modules.loss import _WeightedLoss
import numpy as np
import torch.nn as nn

logger = logging.getLogger(__name__)


# class FocalLoss(torch.nn.Module):
#     Commented out during cleanup: not used by current train/evaluate flow.


# class WeightedFocalLoss(torch.nn.Module):
#     Commented out during cleanup: not used by current train/evaluate flow.


# class SupportWeightedFocalLoss(torch.nn.Module):
#     Commented out during cleanup: not used by current train/evaluate flow.


class CELoss_with_costs(_WeightedLoss):

    def __init__(self, n_classes, weights: Optional[Tensor] = None, confidence: List | float | None = None,
                 costs=None, inv_costs=None, device='cpu', ignore_index=-100) -> None:
        """

        Parameters
        ----------
        n_classes
        weights
        confidence: regularization confidence parameter in [0, 1], or a list with n_classes parameters. When 1 or list
        of ones or None, cross entropy is computed in a standard way. To have a sensible influence confidence should be
        in [1/n, 0.5], where n is the number of classes or better the number of allowed-error classes with not high cost
        costs: square matrix of n_classes size with the confusion costs. The rows correspond to observations while
        columns correspond to predictions. The values on the diagonal do not mutter (not used). The values in a row
        do not necessary need to sum to one, they will automatically be normalized. Only the ratio of off-diagonal
        values within a row matters. High costs mean undesired outcomes.
        inv_costs: square matrix of n_classes size with the inverse confusion costs. The rows correspond to observations
        while columns correspond to predictions. Inverse confusion values are relative to the diagonal value on the same
        row. The values in a row do not need to sum to one, they will automatically be normalized. Low inverse costs
        mean undesired outcomes.
        Either costs and confidence or inv_costs should be provided (or nothing)
        # TODO Costs can also directly be given in probability terms in [0, 1] range, and diagonal values can be 1.0
        # TODO by default (they will be automatically normalized to sum-up to 1, so no constraint for the user).
        # TODO In that high numbers correspond to not critical allowed errors and confidence parameter is not needed.
        device:
        """
        super().__init__(weights, reduction='mean')

        # assert len(costs) == n_classes
        # assert len(costs[0]) == n_classes

        self.device = device
        self.ignore_index = ignore_index
        self.n_classes = n_classes
        self.target_soft_hot = None

        # confidence = 1
        # costs = [[1, 2, 2, 1], [1, 0, 2, 2], [3, 3, 0, 1], [1, 1, 1, 0]]

        if inv_costs is not None:
            assert costs is None and confidence is None

            inv_costs = np.array(inv_costs, dtype=np.float32)
            assert inv_costs.shape[0] == inv_costs.shape[1]
            assert inv_costs.shape[0] == n_classes

            assert np.all(inv_costs <= 1)
            assert np.all(inv_costs >= 0)

            factor = np.sum(inv_costs, axis=1, keepdims=True)
            costs = inv_costs / factor

        # elif confidence is not None and costs is not None:
        #
        #     # set confidence vector, if scalar, then multiply n_classes times
        #     if isinstance(confidence, (float, int)):
        #         conf = np.full((n_classes, 1), confidence)
        #     else:  # list
        #         assert len(confidence) == n_classes
        #         conf = np.expand_dims(confidence, axis=1)
        #
        #     # Prepare regularizing confusion matrix:
        #     # Values on diagonal will be the provided confidence values
        #     # Off-diagonal values are inverse values of provided costs normalized such that sum of each row equals one
        #     costs = 1 / np.array(costs, dtype=np.float32)
        #
        #     assert costs.shape[0] == costs.shape[1]
        #     assert costs.shape[0] == n_classes
        #
        #     diag_ind = np.diag_indices(n_classes)
        #     costs[diag_ind] = 0
        #     factor = np.sum(costs, axis=1, keepdims=True) / (1 - conf)
        #     costs = costs / factor
        #     costs[diag_ind] = confidence

        else:
            # reduces to regular cross entropy without costs
            costs = np.diag(np.ones(n_classes))

        # self.costs = torch.tensor([[1, 0, 0, 0],
        #                            [0, 1, 0, 0],
        #                            [0, 0, 1, 0],
        #                            [0, 0, 0, 1]], dtype=torch.float64).to(self.device)

        # self.costs = torch.tensor([[1, 0, 0, 0, 0, 0, 0],
        #                                 [0, 1, 0, 0, 0, 0, 0],
        #                                 [0, 0, 1, 0, 0, 0, 0],
        #                                 [0, 0, 0, 1, 0, 0, 0],
        #                                 [0, 0, 0, 0, 1, 0, 0],
        #                                 [0, 0, 0, 0, 0, 1, 0],
        #                                 [0, 0, 0, 0, 0, 0, 1]], dtype=torch.float64).to(self.device)

        # self.costs = torch.tensor([[0.7, 0.1, 0.1, 0.1],
        #                            [0, 0.7, 0.3, 0],
        #                            [0, 0, 0.7, 0.3],
        #                            [0, 0.3, 0, 0.7]], dtype=torch.float64).to(self.device)

        # self.costs = torch.tensor([[0.5, 0.1, 0.1, 0.1, 0, 0.1, 0.1],
        #                               [0.05, 0.7, 0.05, 0.05, 0.05, 0.05, 0.05],
        #                               [0.05, 0.05, 0.7, 0.05, 0.05, 0.05, 0.05],
        #                               [0.1, 0.1, 0.0, 0.5, 0.1, 0.1, 0.1],
        #                               [0.05, 0.05, 0.05, 0.05, 0.7, 0.05, 0.05],
        #                               [0.05, 0.05, 0.05, 0.05, 0.05, 0.7, 0.05],
        #                               [0, 0, 0, 0, 0, 0.4, 0.6]], dtype=torch.float64).to(self.device)

        # if costs is None:
        #     costs = np.diag(np.ones(n_classes))
        # else:
        #     costs = np.array(costs)

        # costs = np.array([[1,   0.7, 0, 0],
        #                   [0, 1,   0.7, 0],
        #                   [0.1, 0.1, 1,   0.1],
        #                   [0.1, 0.1, 0.1, 1]])

        with np.printoptions(precision=3, suppress=True, linewidth=225):
            logger.info(f"cost matrix:\n {costs}")

        # TODO do I need it here on GPU?
        self.costs = torch.tensor(costs, dtype=torch.float32).to(self.device)

        #self.target_soft_hot = torch.FloatTensor(torch.Size([batch_size, n_classes])).to(self.device)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:

        assert input.ndim == 2, "input should be of shape (batch_size, n_classes)"
        assert target.ndim == 1, "target should be of shape (batch_size,)"
        assert input.shape[1] == self.n_classes, f"input should have {self.n_classes} classes, got {input.shape[1]}"
        assert target.shape[0] == input.shape[0], f"target should have the same batch size as input, got {target.shape[0]} and {input.shape[0]}"

        batch_size = len(target)

        # Lazy initialization of target_soft_hot on first use, since batch size can be smaller on the last epoch
        if self.target_soft_hot is None:
            self.target_soft_hot = torch.FloatTensor(torch.Size([batch_size, self.n_classes])).to(self.device)

        target_soft_hot = self.target_soft_hot[:batch_size]

        # filter observations that should be ignored
        mask_valid = (target != self.ignore_index)
        if not torch.any(mask_valid):
            return torch.tensor(0., device=input.device, dtype=input.dtype, requires_grad=True)

        target_valid = target[mask_valid]
        target_soft_hot_valid = target_soft_hot[mask_valid]
        input_valid = input[mask_valid]

        # transform targets from vector of indexes (true classes) to matrix of probabilities (observation over classes)
        # for ind, t in enumerate(target_valid):
        #     target_soft_hot_valid[ind] = self.costs[t]
        target_soft_hot_valid[:] = self.costs[target_valid]

        # print(input.device)
        # print(target_soft_hot.device)
        ce = cross_entropy(input_valid, target_soft_hot_valid, weight=self.weight, reduction='mean', label_smoothing=0.0)
        return ce


class MultiTaskCrossEntropy(torch.nn.Module):
    def __init__(self, num_heads, weights_MTL: Tensor = None, weights_MC: Tensor = None, weights_MTL_MC: Tensor = None,
                 pos_weights: list = None, inv_costs=None, device='cpu'):
        super(MultiTaskCrossEntropy, self).__init__()

        if pos_weights:
            assert len(pos_weights) == num_heads
        else:
            pos_weights = torch.ones(num_heads)

        self.device = device
        # TODO Can also be -100 right?
        self.mc_ignore_label = -1
        #self.bc_loss = nn.BCELoss(reduction='mean')
        self.bc_loss = []
        for bc_ind, pos_weight in enumerate(pos_weights):
            pos_weight = torch.tensor(pos_weight)
            self.bc_loss.append(nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight))

        #self.bc_loss = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=None)

        # TODO generate tensors with equal weights if weights_MC is none
        # TODO continue here
        if inv_costs is not None:
            self.mc_loss = CELoss_with_costs(len(weights_MC), inv_costs=inv_costs, weights=weights_MC,
                                                   device=self.device, ignore_index=self.mc_ignore_label)
        else:
            self.mc_loss = nn.CrossEntropyLoss(weight=weights_MC, reduction='mean', ignore_index=self.mc_ignore_label)
            
        self.weights_MTL = weights_MTL
        self.weights_MTL_MC = weights_MTL_MC
        self.num_heads = num_heads
        assert len(self.weights_MTL_MC) == 2, "length for MTL_MC_weights should be 2"
        if weights_MTL is not None:
            assert len(weights_MTL) == self.num_heads, "the number of heads must be equal to the length of weights_MTL"

    def forward(self, inputs: Tensor, targets: Tensor):
        """

        Parameters
        ----------
        inputs: tensor of predictions of size batch size x (number of multitask heads + number multiclass categories)
        targets: tensor with zeros and ones of size batch size x (number of multitask heads + number multiclass categories)

        Returns: loss
        -------

        """

        MTL_outputs = inputs[:, :self.num_heads] # rows are n-hot vectors of MTL tasks
        MC_output = inputs[:, self.num_heads:] # rows are one-hot vectors of multicalss problem
        assert isinstance(MTL_outputs, Tensor)
        assert isinstance(MC_output, Tensor)
        targets = targets.T.float()
        MTL_outputs = MTL_outputs.T

        mtl_targets = targets[:self.num_heads]  # first targets correspond to MTL, the rest to multiclass
        mc_targets = targets[self.num_heads:]  # first targets correspond to MTL, the rest to multiclass

        #n_tasks= len(MTL_outputs)
        assert targets.shape[0] >= self.num_heads

        # compute MTL losses
        #losses = torch.empty(self.num_heads, device=targets.device)
        losses = torch.full((self.num_heads,), float('nan'), device=targets.device)

        # for each class
        for n, (task_output, target) in enumerate(zip(MTL_outputs, mtl_targets)):
            task_output = torch.flatten(task_output)

            # debugging
            # target[target == 0.5] = 0

            ind_valid = (target == 1) | (target == 0) # filter out samples without definite labels
            if torch.any(ind_valid):
                losses[n]  = self.bc_loss[n](task_output[ind_valid], target[ind_valid])

        # weight losses
        #print(losses.device)
        not_nan_ind = ~torch.isnan(losses)
        if self.weights_MTL is not None:
            assert len(self.weights_MTL) == self.num_heads, f"length of task_weights should be {self.num_heads}"

            #weights_MTL_tensor = torch.tensor(self.weights_MTL, device=losses.device)
            av_loss_MTL = torch.dot(losses[not_nan_ind], self.weights_MTL[not_nan_ind])
            # av_loss_MTL = 0
            # for n, loss in enumerate(losses):
            #     av_loss_MTL += loss * self.weights_MTL[n]

            #av_loss_MTL = av_loss_MTL / torch.sum(self.weights_MTL)
            av_loss_MTL = av_loss_MTL / torch.sum(self.weights_MTL[not_nan_ind])
        else:
            av_loss_MTL = torch.mean(losses[not_nan_ind])

        # prepare labels for multiclass loss
        mc_targets_idx =  self.mc_ignore_label * torch.ones(mc_targets.shape[1], dtype=torch.int64, device=MC_output.device)
        # for n, column in enumerate(mc_targets.T):
        #     idx = torch.nonzero(column == 1)
        #     n_val = idx.nelement()
        #     assert n_val <= 1, "one-hot vector is expected or all zeros, got several non zero values"
        #     if n_val == 1:
        #         mc_targets_idx[n] = idx.item()
        #
        #         # debugging, WBC class should have 1 if one of wbc categories has one
        #         assert targets[self.num_heads - 1, n] == 1, "WBC class should have label 1, since target ategory is wbc"
        #
        # #mc_targets_idx_2 = self.mc_ignore_label * torch.ones(mc_targets.shape[1], dtype=torch.int64, device=MC_output.device)

        idx_sample, idx_mc_class = torch.nonzero(mc_targets.T == 1, as_tuple=True)
        assert len(set(idx_sample.cpu().numpy())) == len(idx_sample.cpu().numpy()), "1-hot vector is expected for rows of mc_targets.T or all zeros, got several 1 values"
        mc_targets_idx[idx_sample] = idx_mc_class
        #assert torch.all(mc_targets_idx_2 == mc_targets_idx), "check up"

        # multiclass loss
        loss_mc = 0
        # if there is at least one multi-class label, calculate mc loss
        if not torch.all(mc_targets_idx == self.mc_ignore_label):
            loss_mc = self.mc_loss(MC_output, mc_targets_idx)

        #compute weighted MTL, MC loss
        final_loss = (self.weights_MTL_MC[0] * av_loss_MTL + self.weights_MTL_MC[1] * loss_mc) / sum(self.weights_MTL_MC)

        return final_loss

