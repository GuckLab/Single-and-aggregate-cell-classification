from abc import ABC, abstractmethod
# from pathlib import Path
# from typing import Union
from dcml.evaluation.helpers import get_key_ordered_values_from_dict, add_ml_score_prefix
from dcml.utils.params import get_saved_params
from dcml.utils.models import split_script_file
from mlflow.pytorch import load_model as mlf_load_model
from mlflow.pytorch import pickle_module as mlf_pickle_module

import torch
import os
# import mlflow as mlflow_
# from ..utils.models import load_jit_model, load_mlflow_model
# from ..utils.params import load_ml_feats_from_params


class BaseModel(ABC):
    """Class to standardize application of different model types

    In some cases it is necessary to apply multiple PyTorch-models to data to
    generate the desired output, such as combining output of models which are
    able to predict disjoint sets of cell types.
    """

    @abstractmethod
    def __call__(self, input):
        pass


class OneModel(BaseModel):
    def __init__(self, model_path: str, device: str, model_name: str = ""):
        """

        Parameters
        ----------
        model_path: rund_id (mlflow) or path to the stand alone *.pth model
        device
        model_name: the name of the (best) model. Only provided when model_path is mlflow run_id
        """

        self.device = device
        if os.path.splitext(model_path)[1] == ".pth":  # stand-alone model
            # self.model, transform, extra_files = load_jit_model(
            #     model_path,
            #     map_location=torch.device(self.device))

            print(f"reading the model from {model_path}")
            script_file = torch.load(model_path, map_location=torch.device(self.device), pickle_module=mlf_pickle_module)

        else:  # mlflow model
            assert model_name != '', "model name should be provided"
            model_path_mlflow = f"runs:/{model_path}/{model_name}"
            print(f"reading the model from mlflow at {model_path_mlflow}")
            script_file = mlf_load_model(model_path_mlflow, map_location=torch.device(self.device))

        params = get_saved_params(model_path)
        self.model, transform = split_script_file(script_file)
        extra_files = {"params": params}

        # self.model, transform, extra_files = load_mlflow_model(run_id=model_path,
        #                                                        map_location=torch.device(self.device),
        #                                                        model_name=model_name)
        # self.epoch = extra_files["epoch"]
        self.params = extra_files["params"]
        self.transform = transform
        self.model.eval()
        self.model.to(self.device)

        ml_score_abbrvs, _ = get_key_ordered_values_from_dict(self.params["create_trainer"]["ml_score_features"])
        self.feature_names = [add_ml_score_prefix(abbrv) for abbrv in ml_score_abbrvs]
        # self.feature_names = [f"ml_score_{abbrv}" for abbrv in ml_score_abbrvs]
        # self.feature_names = load_ml_feats_from_params(self.params)

    def __call__(self, model_input):

        model_input = model_input.to(self.device)
        output = self.model(model_input)
        output = output.to("cpu").numpy()  # TODO is it needed here?

        return output
