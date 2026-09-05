### Models

This sub-directory contains the code for different model architectures.

The way to create and add new models is the following:

- You create a class that inherits from `torch.nn.Module` and define the `forward()`-function which describes how the model passes data through.
- You need to define a function that returns an instance of this class:
E.g. the file `resnet_models.py` contains the function:
```python
def resnet(layers: List[int], input_dim: int,
           inplanes: int, num_classes: int) -> ResNet:
    return ResNet(block=BasicBlock, layers=layers, input_dim=input_dim,
                  inplanes=inplanes, num_classes=num_classes)
```
- It is important that the function-name coincides with the first part of the file-name. In our case, the file is called `resnet_models.py`, so the function in that module that returns the model instance needs to be called `resnet()`. If the file is called `resnetC_models.py` then the function needs to be called `resnetC()`.

This is also the same name (e.g. `resnet` or `resnetC`) that is used in the `params.yaml`-file that defines which model is used during training.