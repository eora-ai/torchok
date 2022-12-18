Datasets
########

.. _dataset_interface:

Interface
*********

TODO: about difference between transforms and augments
This rule 
isn't regulated and is a recommendation for users on how to implement their custom datasets: an image after 
augmentations can be accessed via `dataset.get_raw(idx: int)` method

.. automodule:: torchok.data.datasets

Classification
**************

.. automodule:: torchok.data.datasets.classification

Segmentation
************
.. automodule:: torchok.data.datasets.segmentation

Representation
**************

.. automodule:: torchok.data.datasets.representation

Ready-to-go datasets
********************

.. automodule:: torchok.data.datasets.examples
