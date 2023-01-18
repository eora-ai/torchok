Datasets
########

.. _dataset_interface:

Interface
*********

Each dataset in TorchOk has to be inherited from a single interface `ImageDataset`. There are a few methods that need 
to be implemented. Follow the general principles when you implement your dataset:

* Constructor

.. code-block:: python

    def __init__(self,
                transform: Optional[Union[BasicTransform, BaseCompose]],
                augment: Optional[Union[BasicTransform, BaseCompose]] = None,
                input_dtype: str = 'float32',
                image_format: str = 'rgb',
                rgba_layout_color: Union[int, Tuple[int, int, int]] = 0,
                test_mode: bool = False):
        # Use transforms and augments for two different purposes: augmentations should be applied to get a randomly 
        # manipulated image version while transformations are used to get a fixed transformation of each input image 
        # to be able to pass it to the neural network model (like resizing, normalization and to-tensor conversion)

* Length of the dataset

.. code-block:: python

    def __len__(self) -> int:
        # Return total expected length of the dataset

* Getting access to a raw item of the dataset

.. code-block:: python

    def get_raw(self, idx: int) -> dict:
        # Read a sample from disk or whatever your dataset is using. You can utilize self._read_image(image_path) call.
        # Use augmentations on numpy images here
        # Return a dictionary with string keys and tensor values. Usually, images are returned as numpy arrays before 
        # normalization here, so that a user can directly call this method to get understanding on how an output image 
        # looks like

* Getting access to a tensor item of the dataset

.. code-block:: python

    def __getitem__(self, idx: int) -> dict:
        # Usually, a self.get_raw(idx) is called here.
        # Then you should use transformations to transform numpy images and other samples to PyTorch tensors

.. automodule:: torchok.data.datasets

Classification
**************

.. automodule:: torchok.data.datasets.classification.classification

Segmentation
************

.. automodule:: torchok.data.datasets.segmentation.image_segmentation

Representation
**************

.. automodule:: torchok.data.datasets.representation.unsupervised_contrastive_dataset

.. automodule:: torchok.data.datasets.representation.validation

Detection
*********

.. automodule:: torchok.data.datasets.detection.detection

Ready-to-go
***********

.. automodule:: torchok.data.datasets.examples.cifar10

.. automodule:: torchok.data.datasets.examples.coco_detection

.. automodule:: torchok.data.datasets.examples.coco_segmentation

.. automodule:: torchok.data.datasets.examples.sop

.. automodule:: torchok.data.datasets.examples.sweet_pepper

.. automodule:: torchok.data.datasets.examples.triplet_sop
