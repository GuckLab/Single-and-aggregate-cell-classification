import torch
from torch import nn


def build_model(params, MTL=False):
    """Builds the model objects from configurations set in the params-file

    Parameter
    ---------
    params: dict
        Dictionary containing all necessary configuration settings for the
        training.
    Returns
    -------
    model: torch.nn.Module
    """
    arch_type = params["architecture"]["type"]
    arch_name = params["architecture"].get("name", arch_type)
    if arch_name is None:
        arch_name = arch_type
    parent = "dcml.models."
    _temp = __import__(parent + arch_type+"_models", fromlist=[arch_type])
    model_creator_func = getattr(_temp, arch_name)

    num_classes = len(params["class_label_dict"])
    if "MTL_classes" in params:
        #inv_dic = {v: k for k, v in params["class_label_dict"].items()}
        model = model_creator_func(**params["architecture"]["params"], num_classes=num_classes,
                                   num_heads=params["MTL_classes"])
    else:
        model = model_creator_func(**params["architecture"]["params"], num_classes=num_classes)

    return model


# def load_jit_model(path: Union[str, pathlib.Path],
#                    map_location):
#     """Loads a jit-model and its corresponding metadata dictionary
#
#     Loads the jit model and the configurations that have been used during the
#     training of the model. This includes things like:
#         - dataset paths
#         - augmentations
#         - model architecture
#         - loss-function
#         - etc.
#
#     Parameter
#     ---------
#     path: str|pathlib.Path
#         Path to the jit-model
#     map_location: str
#         Options: ["cpu", "cuda"]
#         String indicating if model should be loaded onto cpu or cuda
#
#     Returns
#     -------
#     model: torch.jit.ScriptModule
#     transform: torch.jit.ScriptModule
#     params_dict: dict
#         dictionary containing the training configurations of the training
#         process
#     """
#     extra_files = {"params": ''}
#     script_file = torch.jit.load(path, _extra_files=extra_files, map_location=map_location)
#     # extra_files["epoch"] = (int(extra_files["epoch"])
#     #                         if extra_files["epoch"] else 0)
#     extra_files["params"] = yaml.safe_load(extra_files["params"])
#     model, transform = split_script_file(script_file)
#     return model, transform, extra_files


# def load_mlflow_model(run_id: Union[str, pathlib.Path],
#                       map_location, model_name: str):
#     """
#     """
#
#     model_path = f"runs:/{run_id}/{model_name}"
#     script_file = mlflow.pytorch.load_model(model_path, map_location=map_location)
#     model, transform = split_script_file(script_file)
#
#     params = get_saved_params(run_id)
#     extra_files = {"params": params}
#
#     # local_path = mlflow.artifacts.download_artifacts(
#     #     run_id=run_id,
#     #     artifact_path="params.yaml"
#     # )
#     # # Load the YAML file
#     # with open(local_path, 'r') as file:
#     #     params = yaml.safe_load(file)
#     #     extra_files["params"] = params
#
#     # extra_files["epoch"] = (int(params["training"]["num_epochs"])
#     #                         if params["training"]["num_epochs"] else 0)
#
#     return model, transform, extra_files


def split_script_file(script_file):
    # TODO: Find a better way to figure out if the script_file contains
    # only a model or also the transformations
    # Testing for "Sequential" is not robust enough as it does not uniquely
    # identify a Sequential of [transform, model]
    if script_file.original_name == "Sequential":
        modules = list(script_file.children())
        transform = modules[0]
        model = modules[1]
    else:
        transform = None
        model = script_file
    return model, transform


def combine_to_jit(model,
                   transform
                   ):
    """Combines model and Transformations into one jit-model

    If `transform` is `None`, just turns the model into a script-model

    Parameter
    ---------
    model: torch.nn.Module
        Pytorch model for which a jit-model is returned
    transform: None or torch.nn.Module
        Preprocessing functions that need to be applied before the model.
        In most cases a concatenation of torch.nn.Modules that wrap around
        `toTensor` and `Normalize`.

    Returns
    -------
    seq_scripted: torch.jit.ScriptModule
        ScriptModule version of the concatenation of `transform` and `model`
    """
    if transform:
        seq = nn.Sequential(transform, model)
    else:
        seq = model
    seq_scripted = torch.jit.script(seq)
    return seq_scripted


# def save_jit_model(model, transform, path, params_dict, epoch) -> None:
#     Commented out during cleanup: unused in current train/evaluate flow.


# def add_params_to_jit_model(model_path, ml_feature_list, params_dict, epoch):
#     Commented out during cleanup: unused in current train/evaluate flow.


# TODO remove it, seems to be not used
# def evaluate(model, dataloader, batch_count=None):
#     Commented out during cleanup: unused legacy helper.


#
# Everything below is not used afaik
#
# Will be removed soon
# def load_model_old(params):
#     """deprecated"""
#     arch = params["architecture"]
#     _temp = __import__("torchvision.models", fromlist=[arch])
#     model_creator = getattr(_temp, arch)
#
#     model = model_creator(num_classes=len(params["class_label_dict"]))
#
#     conv1_out = model.conv1.out_channels
#     model.conv1 = copy_conv2d(model.conv1, new_args={'in_channels': 1})
#
#     return model


# def copy_conv2d(conv, new_args={}):
#     Commented out during cleanup: unused in current train/evaluate flow.


# def class_weights(target, num_classes):
#     Commented out during cleanup: unused in current train/evaluate flow.


# def list_original_module_names(module):
#     Commented out during cleanup: unused in current train/evaluate flow.
