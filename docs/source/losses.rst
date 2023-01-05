Losses
######

TorchOk supports all loss functions from PyTorch as well as some customized loss functions.

Customized loss functions
*************************

Classification
==============

.. automodule:: torchok.losses.classification.binary_cross_entropy

Segmentation
============

.. automodule:: torchok.losses.segmentation.dice

Representation
==============

.. automodule:: torchok.losses.representation.pairwise

Detection
=========

At the moment, we support all 
`mmdetection loss functions <https://github.com/open-mmlab/mmdetection/tree/master/mmdet/models/losses>`. 
They might be accessed via the prefix `MM_*`, for example, `MM_FocalLoss`.
