.. torchok documentation master file, created by
   sphinx-quickstart on Sun Jun 12 13:52:53 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TorchOk's documentation!
###################################

TorchOk is a toolkit for fast Deep Learning experiments in Computer Vision.
It's based on `PyTorch`_ and utilizes `PyTorch Lightning`_ for training pipeline routines.

The toolkit consists of:

* Neural Network models which are proved to be the best not only on `PapersWithCode`_ but in practice. All models are under plug&play interface that easily connects backbones, necks and heads for reuse across tasks
* Out-of-the-box support of common Computer Vision tasks: classification, segmentation, image representation and detection coming soon
* Commonly used datasets, image augmentations and transformations (from `Albumentations`_)
* Fast implementations of retrieval metrics (with the help of `FAISS`_ and `ranx`_) and lots of other metrics from `torchmetrics`_
* Export models to ONNX and ability to test the exported model without changing the datasets
* All components can be customized inheriting the unified interfaces: Lightning's training loop, tasks, models, datasets, augmentations and transformations, metrics, loss functions, optimizers and LR schedulers
* Training, validation and testing configurations are represented by YAML config files and managed by `Hydra`_
* Only straightforward training techniques are implemented. No whistles and bells

.. _PyTorch: https://github.com/pytorch/pytorch
.. _PyTorch Lightning: https://github.com/PyTorchLightning/pytorch-lightning
.. _PapersWithCode: https://paperswithcode.com/
.. _Albumentations: https://albumentations.ai/
.. _FAISS: https://github.com/facebookresearch/faiss
.. _ranx: https://github.com/AmenRa/ranx
.. _torchmetrics: https://torchmetrics.readthedocs.io/
.. _Hydra: https://hydra.cc/

Installation
************
Conda
============
To remove previous installation of TorchOk environment, run:

.. code-block:: bash

   conda remove --name torchok --all

To install TorchOk locally, run:

.. code-block:: bash

   conda env create -f environment.yml

This will create a new conda environment **torchok** with all dependencies.

Docker
======
Another way to install TorchOk is through Docker. The built image supports SSH access, 
Jupyter Lab and Tensorboard ports exposing. If you don't need any of this, just omit the corresponding arguments. 
Build the image and run the container:

.. code-block:: bash

   docker build -t torchok --build-arg SSH_PUBLIC_KEY="<public key>" .
   docker run -d --name <username>_torchok --gpus=all -v <path/to/workdir>:/workdir -p <ssh_port>:22 \
   -p <jupyter_port>:8888 -p <tensorboard_port>:6006 torchok


Getting started
***************
The folder ``examples/configs`` contains YAML config files with some predefined training and inference configurations.

Train
=====
For training example we can use the default configuration ``examples/configs/classification_cifar10.yml``, 
where the CIFAR-10 dataset and the classification task are specified. 
The CIFAR-10 dataset will be automatically downloaded into your `~/.cache/torchok/data/cifar10` folder (341 MB).

**To train on all available GPU devices (default config):**

.. code-block:: bash

   python train.py -cp examples/configs -cn classification_cifar10

**To train on all available CPU cores:**

.. code-block:: bash

   train.py -cp examples/configs -cn classification_cifar10 trainer.accelerator='cpu'

During the training you can access the training and validation logs by starting a local TensorBoard:

.. code-block:: bash

   tensorboard --logdir ~/.cache/torchok/logs/cifar10

Export to ONNX
==============
TODO

Test ONNX model
===============
TODO

Run tests
*********
.. code-block:: bash

   python -m unittest discover -s tests/ -p "test_*.py"

User Guide
**********

.. toctree::
   :maxdepth: 2
   
   tasks
   models
   datasets
   transforms_augments
   metrics
   onnx_export
   yaml_config

To be added soon (TODO)
***********************
Tasks
=====
* MOBY (unsupervised training)
* DetectionTask
* InstanceSegmentationTask

Backbones
=========
* Swin-v2
* HRNet
* ViT
* EfficientNet
* MobileNetV3

Segmentation models
===================
* HRNet neck + OCR head
* U-Net neck

Detection models
================
* YOLOR neck + head
* DETR neck + head

Datasets
========
* Stanford Online Products
* Cityscapes
* COCO

Losses
======
* Pytorch Metric Learning losses
* NT-ext (for unsupervised training)

Metrics
=======
* Segmentation IoU
* Segmentation Dice
* Detection metrics
