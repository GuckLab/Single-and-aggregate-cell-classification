import argparse
import os
import time
from dotenv import load_dotenv

import dclab
import mlflow
import pandas as pd
from src.dcml.predict import mtl_full_prediction, MC_prediction
import torch
import numpy as np


# load env variables, needed if run as main
load_dotenv()

class rtdc_detections:

    def __init__(self, filepath, thr=0.5, ml_score_feature_names=None):

        with dclab.new_dataset(filepath) as ds:

            self.n_events = len(ds)

            if "ml_score_wbc" in ds.features:

                self.ind_wbc = ml_score_feature_names.index('ml_score_wbc')
                self.ind_agg = ml_score_feature_names.index('ml_score_agg')

                # it is supposed that MTL features are going up to WBC, following by MC features
                MTL_mtl_feature_names = ml_score_feature_names[:(self.ind_wbc + 1)]
                MTL_mc_feature_names = ml_score_feature_names[(self.ind_wbc + 1):]

                self.free_dependent_aggr = ([], [])
                if "ml_score_rba" in ds.features:
                    rbc_aggregation_idx = ml_score_feature_names.index('ml_score_rba')
                    rbc_idx = ml_score_feature_names.index('ml_score_rbc')
                    self.free_dependent_aggr[0].append(rbc_idx)
                    self.free_dependent_aggr[1].append(rbc_aggregation_idx)

                if "ml_score_tra" in ds.features:
                    thrmb_aggregation_idx = ml_score_feature_names.index('ml_score_tra')
                    thrmb_idx = ml_score_feature_names.index('ml_score_trb')
                    self.free_dependent_aggr[0].append(thrmb_idx)
                    self.free_dependent_aggr[1].append(thrmb_aggregation_idx)


                self.cell_predictions_mtl_mc, self.all_MTL_features = self._MTL_detect(ds, MTL_mtl_feature_names,
                                                                                       MTL_mc_feature_names, thr)
                self.type = "MTL"

            else:
                self.cell_ind_predictions_mc = self._MC_detect(ds, ml_score_feature_names, thr)
                self.type = "MC"
                self.MC_feature_names = ml_score_feature_names


    def _find_idx(self, all_features, used_features):
        """
        Finds indexes of all_features that are in used_features

        Parameters
        ----------
        all_features - itarable with all feature names
        used_features - itarable with used feature names

        Returns
        -------
        used_ind - indexes of all_features that have strings from used_features
        """

        # find indexes of used features
        used_ind = []
        for ind_f in range(len(all_features)):
            if all_features[ind_f] in used_features:
                used_ind += [ind_f]

        return used_ind


    def _MC_detect(self, ds, MC_feature_names, thr: float) -> np.ndarray:
        """

        Parameters
        ----------
        ds: rtdc dataset
        MC_feature_names: list/tuple of ml_feature names of multi-class classifier
        thr: decision threshold in [0, 1]

        Returns
        -------
        cell_ind_predictions_mc: 1D numpy array of predictions - indices for of MC_feature_names
        """

        cell_scores_mc = torch.zeros(len(ds), len(MC_feature_names))
        for entry in range(len(MC_feature_names)):
            cell_scores_mc[:, entry] = torch.tensor(ds[MC_feature_names[entry]])

        cell_ind_predictions_mc = MC_prediction(cell_scores_mc, thr)
        cell_ind_predictions_mc = cell_ind_predictions_mc.numpy()

        return cell_ind_predictions_mc


    def _MTL_detect(self, ds, MTL_mtl_feature_names, MTL_mc_feature_names, thr: float) -> (np.ndarray, list):
        """

        Parameters
        ----------
        ds: rtdc dataset
        MTL_mtl_feature_names: list or tuple of string cell type names (ml_) for MTL part of MTL classifier
        MTL_mc_feature_names: list or tuple of string cell type names (ml_) for MC part of MTL classifier
        ind_wbc: index for WBC cell type in MTL_mtl_feature_names
        ind_agg: index for aggregation cell type in MTL_mtl_feature_names
        free_dependent_aggr: a tuple of lists of class indexes. The first list is for independent classes, while the second
        for the corresponding dependent classes. For example ([0, 3], [1, 4]), where 0 is RBC, 3 is RBC aggregation,
        1 is thrombocyte, 4 is thrombocyte aggregation.
        thr: decision threshold in [0, 1]

        Returns
        -------
        cell_predictions_mtl_mc: 2D numpy array with the number of rows as the number of samples and the number of
        columns as the number cell types (detected features). The number of cell types equals the number of cell type
        names in both MTL_mtl_feature_names, and MTL_mc_feature_names together.
        The array contains decisions, 0s or 1s (float).

        all_MTL_features: a list of cell type names concatenated from both MTL_mtl_feature_names and
        MTL_mc_feature_names. Indices of all_MTL_features correspond to the indices of columns of cell_predictions_mtl_mc
        """

        # create matrix of mtl detections, where a rows contain different detections for the same sample
        cell_scores_mtl = torch.zeros(len(ds), len(MTL_mtl_feature_names))
        for entry in range(len(MTL_mtl_feature_names)):
            cell_scores_mtl[:, entry] = torch.tensor(ds[MTL_mtl_feature_names[entry]])

        # create matrix of mc detections, where a rows contain different detections for the same sample
        cell_scores_mc = torch.zeros(len(ds), len(MTL_mc_feature_names))
        for entry in range(len(MTL_mc_feature_names)):
            cell_scores_mc[:, entry] = torch.tensor(ds[MTL_mc_feature_names[entry]])

        cell_predictions_mtl, cell_predictions_mc = mtl_full_prediction(cell_scores_mtl,
                                                                        self.ind_wbc,
                                                                        self.ind_agg,
                                                                        cell_scores_mc, thr, self.free_dependent_aggr)

        cell_predictions_mtl = cell_predictions_mtl.numpy()
        cell_predictions_mc = cell_predictions_mc.numpy()

        cell_predictions_mtl_mc = np.concatenate((cell_predictions_mtl, cell_predictions_mc), axis=1)
        all_MTL_features = MTL_mtl_feature_names + MTL_mc_feature_names

        return cell_predictions_mtl_mc, all_MTL_features

    def _remove_detected_cells_if_aggr(self):

        # If cell aggregation is detected specific cells (and specific cell aggregations) detections are zeroed ->
        # allow only single cell detections

        cell_predictions_mtl_mc = self.cell_predictions_mtl_mc.copy()

        # find events with aggregated cells
        aggr_pred = cell_predictions_mtl_mc[:, self.ind_agg].astype(bool)

        # index of cell specific aggregation categories
        ind_non_aggr = np.ones(cell_predictions_mtl_mc.shape[1]).astype(bool)
        ind_non_aggr[self.ind_agg] = False

        # remove detections of specific cell for events having cell aggregation
        msk_non_aggr = aggr_pred[:, None] & ind_non_aggr[None, :]
        cell_predictions_mtl_mc[msk_non_aggr] = 0

        return cell_predictions_mtl_mc

    def _remove_aggr_cells(self):

        # # remove detections of general aggregated cells

        cell_predictions_mtl_mc = self.cell_predictions_mtl_mc.copy()

        cell_predictions_mtl_mc[:, self.ind_agg] = 0

        return cell_predictions_mtl_mc




    def cell_detections(self, cell_type_codes=None, single_cell=True) -> int:
        """

        Parameters
        ----------
        cell_type_codes - None, string, or list of strings with ml_ names of the cell types
        single_cell - relevant only for MTL case. For True, the detection of any specific cell is ignored if
        aggregation state (cells) was detected. (Only single cells are counted).

        Returns
        -------
        cell_detections: the number of detected cells (int) for given cell types, all together. If cell_type_codes is None, it is the
        overall number of detections (irrespectively of single_cell flag all cells are counted)
        """

        if self.type == "MTL":

            if cell_type_codes is None:

                # count all detections. Generic aggregations are excluded unless no specific cell was detected for an event
                #cell_detections = np.any(cell_predictions_mtl_mc, axis=1).sum()
                cell_predictions_mtl_mc = self._remove_aggr_cells()
                cell_detections = np.maximum(np.sum(cell_predictions_mtl_mc, axis = 1), cell_predictions_mtl_mc[:, self.ind_agg])
                cell_detections = np.sum(cell_detections)

            elif isinstance(cell_type_codes, str):
                cell_entry = self.all_MTL_features.index(cell_type_codes)
                if single_cell:
                    cell_predictions_mtl_mc = self._remove_detected_cells_if_aggr()
                    cell_detections = cell_predictions_mtl_mc[:, cell_entry].sum()
                else:
                    cell_detections = self.cell_predictions_mtl_mc[:, cell_entry].sum()

            elif isinstance(cell_type_codes, list):
                used_ind = self._find_idx(self.all_MTL_features, cell_type_codes)
                if single_cell:
                    cell_predictions_mtl_mc = self._remove_detected_cells_if_aggr()
                    cell_detections = np.any(cell_predictions_mtl_mc[:, used_ind], axis=1).sum()
                else:
                    cell_detections = self.cell_predictions_mtl_mc[:, used_ind].sum()

            else:
                print("cell_type_codes can only be either string or list")
                raise

        else:

            if cell_type_codes is None:

                cell_detections = len(self.cell_ind_predictions_mc)

            elif isinstance(cell_type_codes, str):
                cell_detections = (self.MC_feature_names.index(cell_type_codes) == self.cell_ind_predictions_mc).sum()

            elif isinstance(cell_type_codes, list):
                used_ind = self._find_idx(self.MC_feature_names, cell_type_codes)
                cell_detections = np.isin(self.cell_ind_predictions_mc, used_ind).sum()

            else:
                print("cell_type_codes can only be either string or list")
                raise

        return cell_detections

def evaluate_confusion_on_gmm(measurements: list, ml_score_features: list, cell_type_naming: dict, predictions_dir_path: str,
                    thr=0.5, # currently needed for MTL only
                    artifact_path: str ="",
                    single_cell_mode = True
                    ):

    """
    
    Parameters
    ----------
    measurements: list of measurements - folder names
    ml_score_features: list of ml_score_ features (strings)
    cell_type_naming: dictionary with keys as ml_score_ features and values as string names of the beginning of the
    labeled rtdc files (measurement_labels on configuration file)
    predictions_dir_path: path to predictions
    thr: prediction threshold
    artifact_path: name of the folder in MLflow Artifacts
    single_cell_mode: (relevant only for MTL case) if True, the detection of any specific cell is ignored if aggregation state (cells) was detected.
    Only single cells are counted. If False, all detected cells are counted irrespectively of the aggregation state.

    -------

    """

    other_type = "other"
    dig_after_comma = 4

    true_codes_list = [value for value in cell_type_naming.values() if value]

    dict_confusion_matrices = {}
    dict_n_events = {}
    percentages_dict_confusion_matrices = {}
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)

    existing_measurements = []
    for measurement in measurements:
        print(f"analysing measurement: {measurement},"f"{time.ctime()}", flush=True)
        dir_path = os.path.join(predictions_dir_path, measurement)

        if not os.path.isdir(dir_path):
            print(f"Measurement {measurement} not found, skipping")
            continue
        else:
            existing_measurements.append(measurement)

        existing_files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        confusion_matrix = pd.DataFrame(index=true_codes_list)
        n_events = pd.Series(index=true_codes_list)

        for true_cell_type, rtdc_label in cell_type_naming.items():

            if not rtdc_label:
                continue # skip if no value for this cell type

            # Construct filepath
            filepath = None
            for file_name in existing_files:
                if file_name.startswith(rtdc_label + "_"):
                    filepath = os.path.join(dir_path, file_name)
                    break # assumes only one file of particular class

            if  not filepath:
                print(f"no measurement that starts with proper cell type name {rtdc_label}")
                confusion_matrix.loc[rtdc_label] = 0
            else:

                # Calculate Percentage of all cell types detected in filepath (true type - cell_type_naming[true_cell_type])

                rtdc_det = rtdc_detections(filepath, thr, ml_score_features)

                n_detections = rtdc_det.cell_detections(cell_type_codes=list(cell_type_naming.keys()), single_cell=single_cell_mode)

                # build confusion matrix for single cells
                for detected_cell_type in cell_type_naming.keys():

                    confusion_matrix.at[rtdc_label, detected_cell_type] = (
                        rtdc_det.cell_detections(cell_type_codes=detected_cell_type, single_cell=single_cell_mode))

                n_events[rtdc_label] = rtdc_det.n_events

                if single_cell_mode:
                    confusion_matrix.at[rtdc_label, other_type] = rtdc_det.n_events - n_detections


        dict_confusion_matrices[measurement] = confusion_matrix
        dict_n_events[measurement] = n_events

        # summ up confusion matrices for all measurements
        try:
            sum_confusion_matrix = sum_confusion_matrix + confusion_matrix
            sum_n_events = sum_n_events + n_events
        except UnboundLocalError:
            sum_confusion_matrix = confusion_matrix
            sum_n_events = n_events


    # calculate average confusion matrix in percentages
    percentages_confusion_matrix = sum_confusion_matrix.divide(sum_n_events.replace(0.0, 1.0), axis=0) #


    # rounding for some reason works only for float64 (float)
    percentages_confusion_matrix = percentages_confusion_matrix.astype(float).round(dig_after_comma)

    # compute confusion matrices with percentages
    for measurement in existing_measurements:
        percentages_dict_confusion_matrices[measurement] = dict_confusion_matrices[measurement].divide(dict_n_events[measurement], axis=0)


        # rounding for some reason works only for float64 (float)
        percentages_dict_confusion_matrices[measurement] = (
            percentages_dict_confusion_matrices[measurement].astype(float).round(dig_after_comma))

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(f"Average confusion matrix over all GMM measurements:\n {percentages_confusion_matrix}")


    # Save csv file for average confusion matrix
    percentages_confusion_matrix["True cell type"] = true_codes_list
    av_cfg = 'gmm_confusion_mat_av' + '.csv'
    percentages_confusion_matrix.to_csv(av_cfg)

    mlflow.log_artifact(local_path=av_cfg, artifact_path=artifact_path)
    os.remove(av_cfg)

    for measurement in existing_measurements:

        # Save csv file with confusion matrix for each measurement separately
        percentages_dict_confusion_matrices[measurement]["True cell type"] = true_codes_list
        measurement_cfg = 'gmm_confusion_mat_' + measurement + '.csv'
        percentages_dict_confusion_matrices[measurement].to_csv(measurement_cfg)

        mlflow.log_artifact(local_path=measurement_cfg, artifact_path=artifact_path)
        os.remove(measurement_cfg)


def evaluate_proportions_on_gmm(measurements: list, ml_score_features: list, predictions_dir_path: str,
                    thr=0.5, # currently needed for MTL only
                    artifact_path: str ="",
                    full_measurement: str = "",
                    not_wbc_cell_types: dict = None,
                    wbc_cell_types: dict = None,
                    single_cell_mode = True
                    ):

    """

    Parameters
    ----------
    measurements: list of measurements - folder names
    ml_score_features: list of ml_score_ features (strings)
    predictions_dir_path: path to predictions
    thr: prediction threshold
    artifact_path: name of the folder in MLflow Artifacts
    full_measurement: beginning of the file name of the full measurement rtdc file
    not_wbc_cell_types: dictionary with keys as non wbc cell types and values as lists of the corresponding ml_score_ codes
    wbc_cell_types: dictionary with keys as wbc cell types and values as lists of the corresponding ml_score_ codes
    single_cell_mode: (relevant only for MTL case) if True, the detection of any specific cell is ignored if aggregation state (cells) was detected.
    Only single cells are counted. If False, all detected cells are counted irrespectively of the aggregation

    -------

    """

    dig_after_comma = 4
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    cell_proportions = pd.DataFrame()

    existing_measurements = []
    for measurement in measurements:
        print(f"analysing measurement: {measurement},"f"{time.ctime()}", flush=True)
        dir_path = os.path.join(predictions_dir_path, measurement)

        if not os.path.isdir(dir_path):
            print(f"Measurement {measurement} not found, skipping")
            continue
        else:
            existing_measurements.append(measurement)

        existing_files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

        # -- calculate proportions of detected cell types --
        filepath = None
        if full_measurement:
            for file_name in existing_files:
                if file_name.startswith(full_measurement):
                    filepath = os.path.join(dir_path, file_name)
                    break
        else:
            filepath = [os.path.join(dir_path, file_name) for file_name in existing_files]

        if not filepath:
            print(f"no full measurement data was found for {measurement}")
        else:

            rtdc_det = rtdc_detections(filepath, thr, ml_score_features)
            n_events = rtdc_det.n_events
            n_all_detections = rtdc_det.cell_detections()

            wbc_numbers = count_cells(rtdc_det, wbc_cell_types, single_cell=single_cell_mode)
            n_WBCs = sum(wbc_numbers.values())

            non_wbc_numbers = count_cells(rtdc_det, not_wbc_cell_types, single_cell=single_cell_mode)

            for cell_type in wbc_numbers.keys():
                cell_proportions.loc[measurement, cell_type + "/WBC"] = wbc_numbers[cell_type] / n_WBCs

            cell_proportions.loc[measurement, "WBC/detections"] = n_WBCs / n_all_detections

            for cell_type in non_wbc_numbers.keys():
                cell_proportions.loc[measurement, cell_type + "/detections"] = non_wbc_numbers[cell_type] / n_all_detections

            cell_proportions.loc[measurement, "detections/events"] = n_all_detections / n_events
            cell_proportions.loc[measurement, "events"] = n_events


    cell_proportions.loc['average'] = cell_proportions.mean()
    cell_proportions.loc['std'] = cell_proportions.std()
    cell_proportions = cell_proportions.astype(float).round(dig_after_comma)

    # create column of named index to make it visible in mlflow
    cell_proportions.index.name = "Measurement"
    cell_proportions = cell_proportions.reset_index()

    print(f"Cell proportions over all GMM measurements:\n {cell_proportions}")

    av_perc = 'cell_percentages' + '.csv'
    cell_proportions.to_csv(av_perc, index=False)

    mlflow.log_artifact(local_path=av_perc, artifact_path=artifact_path)
    os.remove(av_perc)


def count_cells(rtdc_det, cell_types, single_cell):

    n_cells = {}
    for cell_type, cell_type_mlcodes in cell_types.items():
        for cell_type_mlcode in cell_type_mlcodes:
            # this loop can be skipped since rtdc_det.cell_detections()
            # can also handle lists of strings cell_type_mlcodes directly

            n_cells_current = rtdc_det.cell_detections(cell_type_codes=cell_type_mlcode, single_cell=single_cell)

            if cell_type in n_cells:
                n_cells[cell_type] = n_cells_current + n_cells[cell_type]
            else:
                n_cells[cell_type] = n_cells_current

    return n_cells



