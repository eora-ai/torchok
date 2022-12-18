Getting Started
###############

.. _installation:

Installation
************
pip
============
Installation via pip can be done in two steps:

#. Install PyTorch that meets your hardware requirements via `official instructions`_
#. Install TorchOk by running `pip install --upgrade torchok`
#. *[Optional]* In case you want to use built-in examples, install additional dependencies via `pip install -r requirements/examples`

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


Custom experiments
******************
You can create your experiments by copying and modifying one of the example's YAML configuration file.
The whole configuration is done in a single YAML file. Each component of the experiment is referenced via its `name`
(a Python class from the codebase), and you can specify parameters of the component under `params` key. 
To be able to reference a certain component via its name, it should be registered in the TorchOk's registry of 
components (see `Registry` for details: TODO).

Below is an overview of all the parameters that you can specify in the configuration file of your experiment.

Task
====
Task is an entity that configures main components of your experiment: the model, CV task's specific parameters, 
checkpoints, etc. Task is inherited from the `LightningModule`_ and gives you all the power of PyTorch Lightning 
including train/valid loop handling in a multi-node multi-GPU environment with all the needed hooks for 
on batch/epoch/etc. actions, support of correct mixed precision training and more.

For currently supported tasks, you can visit :ref:`tasks`, or you can implement your own task by taking one of 
the existing tasks as a reference.

Example:

.. code-block:: yaml

  task:
    name: PairwiseLearnTask
    params:
      backbone_name: resnet18
      backbone_params:
        pretrained: true
        in_channels: 3
      pooling_name: Pooling
      head_name: LinearHead
      head_params:
        out_channels: &emb_size 512
        normalize: true
      inputs:
        - shape: [3, &height 224, &width 224]
          dtype: &input_dtype float32
      num_classes: 11318

In the example above the `PairwiseLearnTask` task is referenced with parameters of the neural network model 
to be trained (see `model`_ for details) and the parameters `inputs` and `num_classes` (task-specific). 
`inputs` are used by a `BaseTask` and should be passed if you are going to use the `CheckpointONNX` callback.

TODO: add internals of the tasks: forwarding, steps, outputs, etc.

.. _model:

Model
=====
Each neural network model in TorchOk is represented by a sequence of high granularity modules that can be 
interchanged to construct a new model in a plug and play fashion. The main modules are:
- Backbone: the main module responsible for feature extraction as feature stages. Examples: ResNet50, HRNet, DarkNet
- Neck: a module that defines connections between feature stages. Examples: U-Net / HRNet decoder, FPN
- Pooling: a feature aggregation module. Examples: AvgPool2d, GeM, SPOC
- Head: the task-specific prediction module. Examples: linear classification head, OCR segmentation head, RetinaNet head

A task can use this model structure, though it isn't necessary, but will speed up the experiment implementation and 
reproducibility. This model structure also gives you access to the TorchOk's predefined set of models - some backbones
come from `PyTorch Image Models <timm>`_ while some are implemented internally for better connectivity with task-specific 
necks, poolings and heads.

If you are creating a new task, you can integrate the model quickly via:

.. code-block:: python

  # BACKBONE
  backbone_name = self._hparams.task.params.get('backbone_name')
  backbones_params = self._hparams.task.params.get('backbone_params', dict())
  self.backbone = BACKBONES.get(backbone_name)(**backbones_params)

  # NECK
  neck_name = self._hparams.task.params.get('neck_name')
  neck_params = self._hparams.task.params.get('neck_params', dict())
  neck_in_channels = self.backbone.out_channels

  if neck_name is not None:
      self.neck = NECKS.get(neck_name)(in_channels=neck_in_channels, **neck_params)
      pooling_in_channels = self.neck.out_channels
  else:
      self.neck = nn.Identity()
      pooling_in_channels = neck_in_channels

  # POOLING
  pooling_params = self._hparams.task.params.get('pooling_params', dict())
  pooling_name = self._hparams.task.params.get('pooling_name')
  self.pooling = POOLINGS.get(pooling_name)(in_channels=pooling_in_channels, **pooling_params)

  # HEAD
  head_name = self._hparams.task.params.get('head_name')
  head_params = self._hparams.task.params.get('head_params', dict())
  head_in_channels = self.pooling.out_channels
  self.head = HEADS.get(head_name)(in_channels=head_in_channels, **head_params)

Here `self._hparams` are the experiment hyperparameters taken from the YAML configuration. 
See ClassificationTask for details.

Fine-tuning
===========
You can use the following methods to fine-tune your model:
#. Use `pretrained: true` for backbone. Each model in TorchOk comes with pre-trained weights - most of the backbones
are trained for ImageNet classification, but some might be trained on the task-specific datasets (see :ref:`models`)
#. Set `resume_path` to fully load your experiment, including the task, its optimizers' and schedulers' states.
This option is mainly used when you want to continue training of the experiment from a task's checkpoint
#. Specify `load_checkpoint` parameter to load a custom checkpoint or/and override/exclude some of the modules of
a task (see below for instructions). This option overrides the `pretrained: true` and `resume_path` options

Specifying `resume_path`
------------------------

You can resume the existing experiment in full including model weights and optimization parameters as long as state 
of a learning rate scheduler:

.. code-block:: yaml

  resume_path: 'path/to/checkpoint.ckpt'

Specifying `load_checkpoint`
----------------------------

Imagine your model consists of the following keys with their initial values:

.. code-block::
  :caption: initial model weights

  backbone.linear.1: torch.tensor(0),
  backbone.linear.2: torch.tensor(0),
  head.linear.1: torch.tensor(0),
  head.linear.2: torch.tensor(0),
  head.linear.3: torch.tensor(0)

The whole model checkpoint weights:

.. code-block::
  :caption: ./task/checkpoint/path.ckpt

  backbone.linear.1: torch.tensor(1),
  backbone.linear.2: torch.tensor(1),
  head.linear.1: torch.tensor(1),
  head.linear.2: torch.tensor(1),
  head.linear.3: torch.tensor(1)

In addition, you have another pre-trained model, its head's weights:

.. code-block::
  :caption: ./head2/checkpoint/path.ckpt

  linear.1: torch.tensor(3),
  linear.2: torch.tensor(5),
  linear.3: torch.tensor(7)

Simple checkpoints merge
^^^^^^^^^^^^^^^^^^^^^^^^

Based on your checkpoints, you want to load a pre-trained model while taking head weights from another model. 
To merge the checkpoints' weights in this scenario, your `load_checkpoint` configuration should look like:

.. code-block:: yaml

  task:
  # ...
    load_checkpoint:
      base_ckpt_path: ./task/checkpoint/path.ckpt
      overridden_name2ckpt_path:
        head: ./head2/checkpoint/path.ckpt
      strict: true

`strict` parameter means exactly the same what it means in 
`PyTorch's model loading <https://pytorch.org/tutorials/beginner/saving_loading_models.html#warmstarting-model-using-parameters-from-a-different-model>`_. 
Setting it to `true` here means that we expect the merged checkpoint keys are exactly matched with the model's keys. 
Otherwise, model loading will throw an error.

After loading, your total model weights will look like:

.. code-block::

  backbone.linear.1: torch.tensor(1),
  backbone.linear.2: torch.tensor(1),
  head.linear.1: torch.tensor(3),
  head.linear.2: torch.tensor(5),
  head.linear.3: torch.tensor(7)

Overriding keys directly
^^^^^^^^^^^^^^^^^^^^^^^^

Now, imagine that in addition to this you want to take one of the linear layer's weights from a third model overriding 
the weights from the first checkpoint's head. Let the weights from the third model be:

.. code-block::
  :caption: ./head3_linear2/checkpoint/path.ckpt

  linear.2: torch.tensor(10)

In this case, your configuration should look like this:

.. code-block:: yaml

  task:
  # ...
    load_checkpoint:
      base_ckpt_path: ./task/checkpoint/path.ckpt
      overridden_name2ckpt_path:
        head: ./head2/checkpoint/path.ckpt
        head.linear.2: ./head3_linear2/checkpoint/path.ckpt
      strict: true

And the total model weights will be (a deeper specified key overrides a shallower specified key):

.. code-block::

  backbone.linear.1: torch.tensor(1),
  backbone.linear.2: torch.tensor(1),
  head.linear.1: torch.tensor(3),
  head.linear.2: torch.tensor(10),
  head.linear.3: torch.tensor(7)

Excluding keys
^^^^^^^^^^^^^^

Let's now see how will your config look like if you decide to leave one of the layers with default initialization:

.. code-block:: yaml

  task:
  # ...
    load_checkpoint:
      base_ckpt_path: ./task/checkpoint/path.ckpt
      overridden_name2ckpt_path:
        head: ./head2/checkpoint/path.ckpt
        head.linear.2: ./head3_linear2/checkpoint/path.ckpt
      exclude_keys:
        head.linear.3
      strict: false

In this case, `strict: false` allows us to load model weights partially skipping the `head.linear.3` layer.

After loading, your total model weights will be:

.. code-block::

  backbone.linear.1: torch.tensor(1),
  backbone.linear.2: torch.tensor(1),
  head.linear.1: torch.tensor(3),
  head.linear.2: torch.tensor(10),
  head.linear.3: torch.tensor(0)

Absolute model keys
^^^^^^^^^^^^^^^^^^^

If one of your checkpoints has full length keys (we call them *absolute* keys), you don't need to convert the keys 
to *relative* ones, TorchOk will read them as having an overridden key prefix included. Let your second model's head's 
weights be:

.. code-block::
  :caption: ./head2/checkpoint/path.ckpt (absolute keys)

  head.linear.1: torch.tensor(3),
  head.linear.2: torch.tensor(5),
  head.linear.3: torch.tensor(7)

The resulting merged checkpoints won't change from the above examples.

Skip loss calculation on validation
===================================

Sometimes you don't need to calculate loss during the validation. If you don't override `validation_step` of the 
`BaseTask` you can set `compute_loss_on_valid: false` (default is `true`) to skip loss calculation on validation.

Seeding randomizers
===================

To increase reproducibility of your experiments, you can provide seed values to internal randomizers. Usually, 
Deep Learning Engineers configure seeds for all the randomizers. This can be achieved in TorchOk by specifying 
`seed_params`:

.. code-block:: yaml

  task:
    # ...
    seed_params:
      seed: 42
      workers: true

This will set the specified seed value for all randomizers including PyTorch, numpy and Python. With `workers: true` 
(default `false`) you will also initialize the seed of PyTorch's DataLoader's worker processes 
(see `PyTorch Lightning documentation <https://pytorch-lightning.readthedocs.io/en/stable/model/build_model_expert.html#seed-everything>`_)

Losses
======

To configure losses, TorchOk provides a simple interface that can help you specify multiple losses, their weights 
and assignment rules so that you can easily map your model outputs and targets to a specific loss function's parameters.

Let's see how losses can be configured via an example:

.. code-block:: yaml

  joint_loss:
    normalize_weights: true
    losses:
      - name: Loss1
        tag: loss1
        weight: 200
        mapping:
          input: x
          target: y
      - name: Loss2
        tag: loss2
        weight: 100
        mapping:
          input: x
          target: y

Here two loss functions are specified. The total applied loss will be a weighted sum of the defined losses.

* `name` is used to find a class in TorchOk's components registry
* `tag` (optional) specifies how this loss function can be accessed from a task
* `weight` (optional) defines a multiplier for a given loss (the weighted average sum is used to calculate the final loss value). If `normalize_weights: true` (default is `true`) then the weights will be normalized to sum up to 1
* `mapping` configures how model outputs (in this case `x`) and targets (in this case `y`) are passed to a specific loss function (in this case both loss functions have parameters `input` and `target`)

If you want to access specific losses directly, you can find them by tags inside your task:

.. code-block:: python

   print(self.losses['loss1'], self.losses['loss2'])

TorchOk supports all PyTorch's loss functions, predefined and custom losses registered through the losses registry.

Optimization
============

Optimizers and schedulers are specified alongside each other, so that each optimizer is assigned to only one scheduler:

.. code-block:: yaml

  optimization:
    - optimizer:
        name: Adam
        params:
          lr: 0.0001
      scheduler:
        name: ExponentialLR
        params:
          gamma: 0.97
    - optimizer:
        name: SGD
        params:
          lr: 0.0001

Here two optimizers are specified, one of them has a learning rate scheduler attached, 
so that its learning rate will be changed every epoch according to the exponential decaying rule.

If you want to control learning rate on each step, you can specify PyTorch Lightning parameters for schedulers:

.. code-block:: yaml

  optimization:
    - optimizer:
        name: Adam
        params:
          lr: &base_lr 0.0001
      scheduler:
        name: CyclicLR
        params:
          base_lr: *base_lr
          max_lr: 0.01
        pl_params:
          # See https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
          # for more information
          interval: 'step'

Data
====

To train a model, multiple datasets and dataloaders can be specified. A dataset can be defined via TorchOk's 
:ref:`ImageDataset <dataset_interface>` interface. And a dataloader can be defined via PyTorch's interface 
`torch.utils.data.DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_. 
You can configure a predefined or your custom dataset registered in the Datasets registry:

.. code-block:: yaml

  data:
    TRAIN:
      - dataloader:
          batch_size: 128
          num_workers: 8
          drop_last: true
          shuffle: true
        dataset:
          name: CIFAR10
          params:
            input_dtype: *input_dtype
            train: true
            download: true
            data_folder: &data_folder ${oc.env:HOME}/.cache/torchok/data/cifar10
          transform:
            - &resize
              name: Resize
              params:
                  height: *height
                  width: *width
            - &normalize
              name: Normalize
              params:
                  mean: [ 0.485, 0.456, 0.406 ]
                  std: [ 0.229, 0.224, 0.225 ]
            - &totensor
              name: ToTensorV2
          augment:
            - name: HorizontalFlip
            - name: VerticalFlip
            - name: OneOf
              params:
                transforms:
                  - name: JpegCompression
                  - name: GaussNoise
            - name: ColorJitter
              params:
                p: 0.2
    VALID:
      - dataloader:
          batch_size: 128
          num_workers: 8
          drop_last: false
          shuffle: false
        dataset:
          name: CIFAR10
          params:
            input_dtype: *input_dtype
            train: false
            download: true
            data_folder: *data_folder
          transform:
            - *resize
            - *normalize
            - *totensor

The are 4 phases for datasets configuration: `TRAIN`, `VALID`, `TEST`, `PREDICT`. Each of them is optional. 
`TRAIN` is for training, `VALID` is for validation while the model is being trained, `TEST` is for testing an already 
existing model and `PREDICT` is to get predictions from an already existing model. Data batches are received in 
`training_step`, `validation_step`, `test_step` and `predict_step` of the task respectively.

Each phase (except `TRAIN`) can contain multiple datasets (dataloaders). A dataset and a dataloader are specified 
in pair - the dataset accepts parameters defined in its implementation while dataloader only accepts parameters 
defined in the `PyTorch's interface <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_.

Each dataset can take a list of transformations and augmentations. The difference between the two is that 
augmentations produce an image while transformations produce a tensor (after to-tensor conversion). 
In other words, you can think of it as transformations are always applied to the images (like to-tensor conversion) 
while augmentations are random transformations, and they might be not applied if randomness decides to skip them. 
More details are in :ref:`Datasets interface <dataset_interface>`. 

For transformations and augmentations, TorchOk supports all image manipulation functions from `Albumentations`_ 
library. You can also define your own transformations via `Albumentations`_ interface. Transformations/augmentations 
are applied in sequence with their default probabilities (usually, 0.5 for image manipulations and 1.0 for essential 
transformations like `Resize`, `Normalize` and `ToTensor`). Transformations/augmentations can be composed with one of 
the composition functions such as `OneOf` (only one augmentation from the list will be applied).
Transformations/augmentations can be applied with their own probabilities by specifying the `p: float` parameter. 
For image manipulation parameters, please, refer to `Albumentations`_ documentation.

Metrics
=======

To evaluate your models, TorchOk provides you access to the whole `torchmetrics`_ zoo of metrics as well as custom 
metrics registered in the Metrics registry. You can also implement your own metrics based on the common torchmetrics's 
interface that allows your metrics to collect predictions by batches and run in a distributed mode.

.. code-block:: yaml

  metrics:
    - name: Accuracy
      mapping:
        preds: prediction
        target: target
    - name: F1Score
      mapping:
        preds: prediction
        target: target

You can specify a list of metrics that are accessible at each phase (`TRAIN`, `VALID`, `TEST`, `PREDICT`) through 
the `MetricsManager`. You can get access to the `MetricsManager` from any task as it's created in the `BaseTask`. 
The following methods of the `MetricsManager` are available:

- `update(self, phase: Phase, *args, **kwargs)` - add a batch of predictions and targets to all metrics of the given phase
- `on_epoch_end(self, phase: Phase) -> Dict[str, Tensor]` - get dictionary of metrics calculated during the current epoch. Important: after calling this method, the metrics are reset, so that you can start collecting predictions and targets for a new epoch

To be able to easily manage multiple metrics with different interfaces at the same time, the mapping is utilized: 
you can specify which neural network output is assigned to which metric's input.

If you want to calculate metrics on specific phases, you can list them under the `phases` special parameter:

.. code-block:: yaml

  metrics:
    - name: PrecisionAtKMeter
      params:
        k: 1
        dataset_type: classification
        normalize_vectors: True
        search_batch_size: 256
      mapping:
        vectors: emb1
        group_labels: target
      phases: [VALID]


Callbacks
=========

If you want to do certain actions during training, you can use callbacks. A callback is an abstraction from PyTorch 
Lightning, you can follow `their documentation <https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html>`_ 
to create one. To specify callbacks for your experiment, you can add a section `callbacks` into your configuration file:

.. code-block:: yaml

  callbacks:
    - name: ModelCheckpoint
      params:
        dirpath: *logs_dir
        monitor: valid/F1Score
        save_top_k: 1
        save_last: true
        mode: max
        save_weights_only: False

Trainer
=======

To specify global training parameters, use the `Trainer` section. All the parameters here come from the PyTorch 
Lightning, so follow their `documentation <https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html>` 
for details.

.. code-block:: yaml

  trainer:
    accelerator: 'gpu'
    max_epochs: 30
    log_every_n_steps: 100
    precision: 16

TODO: describe Logger and Hydra parameters
TODO: Components registries

.. _timm: https://github.com/rwightman/pytorch-image-models
.. _LightningModule: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html
.. _Albumentations: https://albumentations.ai/
.. _torchmetrics: https://torchmetrics.readthedocs.io/

Export to ONNX
**************
TODO

Test ONNX model
***************
TODO
