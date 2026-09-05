import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import time
from torch import Tensor
import re

def MC_prediction(probs: Tensor, thr=0.5):
    """
    Logic for making predictions for multi-class model

    Parameters
    ----------
    probs - tensor of shape N x L, with N samples and L features that contains multi-class model probabilities
    thr - threshold to be used for predictions (currently not used)

    Returns
    -------
    preds - tensor of size N with integer predictions (index of the correct class)
    """
    # TODO do I need these conversions to numpy? why not to use tensors?
    # TODO I can also avoid using list, why not just concatenating tensors?
    _, preds = torch.max(probs, 1)

    return preds


def mtl_full_prediction(mtl_scores, wbc_ind, aggr_ind, mc_scores, thr, free_dependent_aggr: tuple=([], [])):

    """
    Makes predictions (True/False) from the scores of MTL model, for major mtl_scores and for WBC mc_scores classes

    Parameters
    ----------
    mtl_scores - Tensor with the shape NxL, with N the number of samples, and L the number of MTL features (5 currently)
    wbc_ind - index of the entry [0, L-1] in MTL features, where is WBC score is
    aggr_ind - index of the entry [0, L-1] in MTL features, where is aggregation score is
    mc_scores - Tensor of shape N x K, with N samples and K features (4 WBC cells currently) for multi-class WBC prediction
    free_dependent_aggr - a tuple of lists of class indexes. The first list is for independent classes, while the second
    for the corresponding dependent classes. For example ([0, 3], [1, 4]), where 0 is RBC, 3 is RBC aggregation,
    1 is thrombocyte, 4 is thrombocyte aggregation.
    thr - threshold to use for predictions (currently the same for all classes)

    Returns
    -------
    preds_mtl - Tensor of the same shape as  mtl_scores, but with boolean predictions
    pred_mc - Tensor with boolean predictions of the same shape as mc_scores
    """

    # make predictions for MTL with thresholding
    #preds_mtl = MTL_mtl_prediction2(mtl_scores, thr, aggr_ind)
    preds_mtl = MTL_mtl_prediction3(mtl_scores, thr, aggr_ind, free_dependent_aggr=free_dependent_aggr)

    # make predictions for multiclass - WBCs (one-hot vectors)
    pred_wbc = preds_mtl[:, wbc_ind]
    pred_mc, pred_wbc = MTL_mc_prediction(mc_scores, pred_wbc)

    # update wbc
    preds_mtl[:, wbc_ind] = pred_wbc

    return preds_mtl, pred_mc


# def MTL_mtl_prediction(mtl_scores: Tensor, mtl_thr: float, aggr_ind: int) -> Tensor:
#     """
#     MTL prediction logic for main MTL classes: if aggr = False (after thresholding), only single maximal entry from other
#     entries in the row will be True, namely the one with the highest score from the entries other than aggr.
#     Currently implemented with a single threshold for all scores.
#
#     Parameters
#     ----------
#     mtl_scores - Tensor with the shape NxL, with N the number of samples, and L the number of MTL features (5 currently)
#     mtl_thr
#     aggr_ind
#
#     Returns
#     -------
#     preds_mtl - Tensor of the same shape as  mtl_scores, but with boolean values """
#
#     # make predictions
#     preds_mtl = mtl_scores >= mtl_thr
#
#     # refine predictions, with the logic: if aggr = 0, only single maximal entry from other entries can be 1,
#     # namely the one with the highest score from the entries other than aggr
#     for i in range(preds_mtl.shape[0]):
#         if preds_mtl[i, aggr_ind] == False: # if no aggregation
#             scores = mtl_scores[i].clone() # scores for one sample
#
#             # find maximal value and its index for entries beside aggr.
#             scores[aggr_ind] = -np.inf
#             max_val, max_ind = torch.max(scores, 0)
#
#             # set True for a single entry if it is higher than a threshold
#             preds_mtl[i, :] = False
#             if max_val >= mtl_thr:
#                 preds_mtl[i, max_ind] = True
#
#     return preds_mtl


# def MTL_mtl_prediction2(mtl_scores: Tensor, mtl_thr: float, aggr_ind: int) -> Tensor:
#     """
#     MTL prediction logic for main MTL classes: all MTL classes are thresholded independently of each other.
#     If more than one class except aggregation is predicted (True), then also aggregation feature is set True.
#     Otherwise, aggregation feature is not changed after thresholding (so it can be that we have aggr=true, but only one
#     of cell types was detected).
#     Currently implemented with a single threshold for all scores.
#
#     Parameters
#     ----------
#     mtl_scores - Tensor with the shape NxL, with N the number of samples, and L the number of MTL features (5 currently)
#     mtl_thr
#     aggr_ind
#
#     Returns
#     -------
#     preds_mtl - Tensor of the same shape as  mtl_scores, but with boolean values """
#
#     # make predictions
#     preds_mtl = mtl_scores >= mtl_thr
#
#     # create a tensor without aggregation column
#     preds_mtl_no_aggr = torch.cat((preds_mtl[:, :aggr_ind], preds_mtl[:, aggr_ind+1:]), dim=1)
#     mult_det = preds_mtl_no_aggr.sum(dim=1) > 1
#
#     preds_mtl[mult_det, aggr_ind] = True
#
#     return preds_mtl


def MTL_mtl_prediction3(mtl_scores: Tensor, mtl_thr: float, aggr_ind: int, free_dependent_aggr: tuple=([], [])) -> Tensor:
    """
    MTL prediction logic for main MTL classes: all MTL classes are thresholded independently of each other.
    If more than one class except aggregation is predicted (True), then also aggregation feature is set True.
    Otherwise, aggregation feature is not changed after thresholding (so it can be that we have aggr=true, but only one
    of cell types was detected).
    Currently implemented with a single threshold for all scores.

    Parameters
    ----------
    mtl_scores - Tensor with the shape NxL, with N the number of samples, and L the number of MTL features (5 currently)
    free_dependent_aggr - free_dependent_aggr - a tuple of lists of class indexes. The first list is for independent classes, while the second
    for the corresponding dependent classes. For example ([0, 3], [1, 4]), where 0 is RBC, 3 is RBC aggregation,
    1 is thrombocyte, 4 is thrombocyte aggregation.
    mtl_thr
    aggr_ind

    Returns
    -------
    preds_mtl - Tensor of the same shape as  mtl_scores, but with boolean values """

    # make predictions
    preds_mtl = mtl_scores >= mtl_thr

    # create a tensor without aggregation column
    preds_mtl_no_aggr = torch.cat((preds_mtl[:, :aggr_ind], preds_mtl[:, aggr_ind+1:]), dim=1)
    mult_det = preds_mtl_no_aggr.sum(dim=1) > 1

    # Set aggregation to True idf there were Trues for at least two different single cells
    preds_mtl[mult_det, aggr_ind] = True

    # Set generic aggregation to True if specific cell aggregation was True
    agg_det = preds_mtl[:, free_dependent_aggr[1]].any(dim=1)
    preds_mtl[agg_det, aggr_ind] = True

    # keep specific cell aggregation True only if the corresponding single cell was also True
    preds_mtl[:, free_dependent_aggr[1]] = preds_mtl[:, free_dependent_aggr[1]] & preds_mtl[:, free_dependent_aggr[0]]

    return preds_mtl


def MTL_mc_prediction(outputs_mc: Tensor, pred_wbc: Tensor):
    """

    MTL prediction logic for multi-class (wbc cells). If WBC was not predicted for a sample, all L cell predictions for that sample will be False.
    Otherwise, an entry with the maximum among L values for a sample will be True, others will be False

    Parameters
    ----------
    outputs_mc - Tensor of shape N x L, with N samples and L features (4 WBC cells currently) for multi-class prediction
    pred_wbc - Tensor of length N with boolean values, which are prediction for WBC cell

    Returns
    -------
    pred_mc - Tensor with boolean values of the same shape as outputs_mc
    pred_wbc - Tensor with bolean values. (currently it is the same as input pred_wbc)
    """
    # most probable class - one hot vector
    _, preds_mc_ind = torch.max(outputs_mc, 1)
    pred_mc = torch.zeros(outputs_mc.shape, device=pred_wbc.device)
    pred_mc[torch.arange(len(pred_mc)), preds_mc_ind] = 1

    # mc one-hot prediction only if WBC
    pred_mc[~pred_wbc] = 0

    # high_prob = outputs_mc[torch.arange(outputs_mc.size(0)), preds_mc_ind] > 0.8
    # pred_mc[~high_prob, :] = 0

    return pred_mc, pred_wbc

def find_dependent_aggregations(inv_dic: dict, aggr_word: str):

    """
    Finds out indexes of dependent aggregations and its mother class.
    For example, if inv_dic = {"Erythrocyte": 0, "Erythrocyte clot": 1, "Thrombocyte": 2, "Thrombocyte_clot": 3,
               "wbc_mono": 4, "wbc_mono_clot": 5, "wbc_baso": 6, "wbc_baso clot":7, "nucl_rbc": 8}
    the function will output ([0, 2, 4, 6], [1, 3, 5, 7])

    Parameters
    ----------
    inv_dic: dictionary with keys as labels and indexes as values
    aggr_word: word used to point to aggregations (e.g. clot). aggrword can be separated from basic independent label
    by either empty space of underscore

    Returns
    -------
    Lists of indexes mother and dependent classes
    """

    labels = list(inv_dic.keys())
    free_inds = []
    aggr_inds = []
    for lab in labels:
        #lab_parts = lab.rsplit(' ', 1)
        lab_parts = re.split('[ _]+', lab) # split string based on either empty space or an underscore
        if len(lab_parts) > 1:
            agg_label = lab_parts[-1] # dependent label - last split part
            if agg_label == aggr_word:
                base_label = lab[:(len(lab) - len(agg_label) - 1)] # the label without split part (dependent label)
                free_inds.append(labels.index(base_label))
                aggr_inds.append(inv_dic[lab])

    return free_inds, aggr_inds


def predict(model: nn.Module, dataloader: DataLoader, criterion, device, class_label_dict, pred_thr=0.5):

    """

    Parameters
    ----------
    model
    dataloader
    criterion
    device
    pred_thr

    Returns
    -------
    target_values, prediction_scores, predictions - Tensors with float numbers (also targets and predictions are float
     not integers)
    loss_values - list of floats
    """

    inv_dic = {v: k for k, v in class_label_dict.items()}

    num_heads = getattr(model, 'num_heads', False) # number of heads for Multi-Task learning
    MTL = bool(num_heads) == True
    model.eval() # model should already be on device
    loss_values = []

    prediction_scores = torch.empty(0, device=device)
    predictions = torch.empty(0, dtype=torch.int, device=device)

    if MTL:
        target_values = torch.empty((0, model.num_mc_classes + num_heads), device=device)

        ind_wbc_class = inv_dic['wbc']
        ind_aggregation_class = inv_dic['Aggregation']
        free_inds, aggr_inds = find_dependent_aggregations(inv_dic, "clot")
    else:
        target_values = torch.empty(0, dtype=torch.int, device=device)

    with_targets = True
    with torch.no_grad():
        for n_batch, batch in enumerate(dataloader):
            # # # TODO remove debugging
            # if n_batch > 100:
            #     break

            inputs = batch["image"].to(device)

            if with_targets and "target" in batch.keys():
                targets = batch["target"]
                if MTL:
                    targets = targets.to(device)
                    target_values = torch.cat((target_values, targets), dim=0)
                else:
                    targets = targets.long().to(device)
                    # target_values.extend(targets.cpu().numpy().astype(int))
                    target_values = torch.cat((target_values, targets), dim=0)
            else:
                with_targets = False
                if criterion is not None:
                    raise TypeError("no target field in the dataloader objects, cannot apply criterion")

            outputs = model(inputs)
            if criterion is not None:
                loss = criterion(outputs, targets)
                loss_values.append(loss.item())

            if MTL:

                # separate outputs for MTL and multiclass
                outputs_mc = outputs[:, model.num_heads:] # white blood cells
                outputs_mtl = outputs[:, :model.num_heads]

                outputs_mc = nn.Softmax(dim=1)(outputs_mc)
                outputs_mtl = nn.Sigmoid()(outputs_mtl)

                # multiply conditional multi-class probabilities by probability of WBC
                wbc_prob = outputs_mtl[:, ind_wbc_class].unsqueeze(1)
                outputs_mc = outputs_mc * wbc_prob

                # multiply conditional aggregation probabilities by probability of the corresponding "mother" class
                for aggr_ind, free_ind in zip(aggr_inds, free_inds):
                    outputs_mtl[:, aggr_ind] = outputs_mtl[:, aggr_ind] * outputs_mtl[:, free_ind]

                # outputs_mtl[:, 1] = outputs_mtl[:, 1] * outputs_mtl[:, 0] # RBC aggregation update
                # outputs_mtl[:, 4] = outputs_mtl[:, 4] * outputs_mtl[:, 3] # thrombocytes aggregation update
                # #outputs_mtl[:, 3] = outputs_mtl[:, 3] * outputs_mtl[:, 2]  # thrombocytes aggregation update
                # #outputs_mtl[:, 6] = outputs_mtl[:, 6] * outputs_mtl[:, 7]  # WBC aggregation update
                # #outputs_mtl[:, 5] = outputs_mtl[:, 5] * outputs_mtl[:, 6]  # WBC aggregation update

                preds_mtl, pred_mc = mtl_full_prediction(outputs_mtl, ind_wbc_class, ind_aggregation_class, outputs_mc, pred_thr, free_dependent_aggr=(free_inds, aggr_inds))

                preds = torch.cat((preds_mtl, pred_mc), dim=1)
                outputs = torch.cat((outputs_mtl, outputs_mc), dim=1)

            else:

                outputs = nn.Softmax(dim=1)(outputs)
                preds = MC_prediction(outputs, pred_thr)

            predictions = torch.cat((predictions, preds), dim=0)
            prediction_scores = torch.cat((prediction_scores, outputs), dim=0)

    if MTL:
        check_predictions_logic(predictions, ind_wbc_class, ind_aggregation_class)

    prediction_scores = prediction_scores.cpu()
    predictions = predictions.cpu()
    target_values = target_values.cpu()

    return target_values, prediction_scores, predictions, loss_values


def check_predictions_logic(predictions, ind_wbc, ind_aggr):

    wbc_present = predictions[:, ind_wbc] == 1
    n_wbc_predictions = torch.sum(predictions[wbc_present, (ind_wbc+1):], dim=1)
    condition = torch.all(n_wbc_predictions == 1)

    assert condition, "at least one wbc cell types output must be 1, if WBC category was detected"

    aggr_not_present = predictions[:, ind_aggr] == 0
    single_mtl = torch.cat((predictions[aggr_not_present, :ind_aggr],
                            predictions[aggr_not_present, ind_wbc][:, torch.newaxis]), dim=1)
    n_single_mtl = torch.sum(single_mtl, dim=1)
    condition = torch.all((n_single_mtl == 0) | (n_single_mtl == 1))
    assert condition, "if aggregation detected, either none or a single mtl output should be detected"

    # TODO add tests



