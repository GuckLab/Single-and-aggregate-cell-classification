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
    parent = "src.dcml.models."
    _temp = __import__(parent + arch_type+"_models", fromlist=[arch_type])
    model_creator_func = getattr(_temp, arch_name)

    num_classes = len(params["class_label_dict"])
    if "MTL_classes" in params:

        model = model_creator_func(**params["architecture"]["params"], num_classes=num_classes,
                                   num_heads=params["MTL_classes"])
    else:
        model = model_creator_func(**params["architecture"]["params"], num_classes=num_classes)

    return model

def split_script_file(script_file):

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



