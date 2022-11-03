Getting Started
###############

.. _installation:

Installation
************
pip
============
Installation via pip can be done in two steps:
1. Install PyTorch that meets your hardware requirements via `official instructions`_
2. Install TorchOk by running `pip install --upgrade torchok`
3. *[Optional]* In case you want to use built-in examples,
install additional dependencies via `pip install -r requirements/examples`

.. _official instructions: https://pytorch.org/get-started/locally/

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


Ready-to-play examples
**********************
.. note:: 
    Additional dependencies should be installed before working with the examples. See :ref:`installation` instructions

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
