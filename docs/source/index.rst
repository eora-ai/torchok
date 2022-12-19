Welcome to TorchOk's documentation!
###################################

TorchOk is a toolkit for fast Deep Learning experiments in Computer Vision.
It's based on `PyTorch`_ and utilizes `PyTorch Lightning`_ for training pipeline routines.

The toolkit consists of:

* Neural Network models which are proved to be the best not only on `PapersWithCode`_ but in practice. All models are under plug&play interface that easily connects backbones, necks and heads for reuse across tasks
* Out-of-the-box support of common Computer Vision tasks: classification, segmentation, image representation and detection
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

User Guide
**********

.. toctree::
  :maxdepth: 2
  
  getting_started
  examples
  tasks
  models
  losses
  datasets
  transforms_augments
  metrics
  callbacks
  loggers
  contribute
