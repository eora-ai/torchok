<div align="center">

<img src="docs/source/_static/TorchOk_logo.png" alt="TorchOk" style="width:300px; horizontal-align:middle"/>

**The toolkit for fast Deep Learning experiments in Computer Vision**

</div>

## A day-to-day Computer Vision Engineer backpack
TorchOk is based on [PyTorch](https://github.com/pytorch/pytorch) and utilizes [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) for training pipeline routines.

The toolkit consists of:
- Neural Network models which are proved to be the best not only on [PapersWithCode](https://paperswithcode.com/) but in practice. All models are under plug&play interface that easily connects backbones, necks and heads for reuse across tasks
- Out-of-the-box support of common Computer Vision tasks: classification, segmentation, image representation and detection coming soon
- Commonly used datasets, image augmentations and transformations (from [Albumentations](https://albumentations.ai/))
- Fast implementations of retrieval metrics (with the help of [FAISS](https://github.com/facebookresearch/faiss) and [ranx](https://github.com/AmenRa/ranx)) and lots of other metrics from [torchmetrics](https://torchmetrics.readthedocs.io/)
- Export models to ONNX and ability to test the exported model without changing the datasets
- All components can be customized inheriting the unified interfaces: Lightning's training loop, tasks, models, datasets, augmentations and transformations, metrics, loss functions, optimizers and LR schedulers
- Training, validation and testing configurations are represented by YAML config files and managed by [Hydra](https://hydra.cc/)
- Only straightforward training techniques are implemented. No whistles and bells

## Installation
### Conda
To remove previous installation of TorchOk environment, run:
```bash
conda remove --name torchok --all
```
To install TorchOk locally, run:
```bash
conda env create -f environment.yml
```
This will create a new conda environment **torchok** with all dependencies.
### Docker
Another way to install TorchOk is through Docker. The built image supports SSH access, Jupyter Lab and Tensorboard ports exposing. If you don't need any of this, just omit the corresponding arguments. Build the image and run the container:
```bash
docker build -t torchok --build-arg SSH_PUBLIC_KEY="<public key>" .
docker run -d --name <username>_torchok --gpus=all -v <path/to/workdir>:/workdir -p <ssh_port>:22 -p <jupyter_port>:8888 -p <tensorboard_port>:6006 torchok
```

## Getting started
The folder `examples/configs` contains YAML config files with some predefined training and inference configurations.
### Train
For training example we can use the default configuration `examples/configs/classification_cifar10.yml`, where the CIFAR-10 dataset and the classification task are specified. The CIFAR-10 dataset will be automatically downloaded into your `~/.cache/torchok/data/cifar10` folder (341 MB).

**To train on all available GPU devices (default config):**
```bash
python train.py -cp examples/configs -cn classification_cifar10
```
**To train on all available CPU cores:**
```bash
train.py -cp examples/configs -cn classification_cifar10 trainer.accelerator='cpu'
```
During the training you can access the training and validation logs by starting a local TensorBoard:
```bash
tensorboard --logdir ~/.cache/torchok/logs/cifar10
```
### Export to ONNX
TODO
### Test ONNX model
TODO

## Run tests
```bash
python -m unittest discover -s tests/ -p "test_*.py"
```
## To be added soon (TODO)
Tasks
- MOBY (unsupervised training)
- DetectionTask
- InstanceSegmentationTask

Backbones
- Swin-v2
- HRNet
- ViT
- EfficientNet
- MobileNetV3

Segmentation models
- HRNet neck + OCR head
- U-Net neck

Detection models
- YOLOR neck + head
- DETR neck + head

Datasets
- Stanford Online Products
- Cityscapes
- COCO

Losses
- Pytorch Metric Learning losses
- NT-ext (for unsupervised training)

Metrics
- Segmentation IoU
- Segmentation Dice
- Detection metrics
