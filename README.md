<div align="center">

# TorchOk

**The toolkit for fast Deep Learning experiments in Computer Vision**

</div>

## What is it?
The toolkit consists of:
- Popular neural network models and custom modules implementations used in our company
- Metrics used in CV such that mIoU, mAP, etc.
- Commonly used datasets and data loaders

The framework is based on [PyTorch](https://github.com/pytorch/pytorch) and 
utilizes [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) for 
training pipeline routines.

## Installation
### Docker
One of the ways to install TorchOk is to use Docker:
```bash
docker build -t torchok --build-arg SSH_PUBLIC_KEY="<public key>" .
docker run -d --name <username>_torchok --gpus=all -v <path/to/workdir>:/workdir -p <ssh_port>:22 -p <jupyter_port>:8888 -p <tensorboard_port>:6006 torchok
```
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
## Getting started
Training is configured by YAML configuration files which each forked project should store inside `configs` folder 
(see `configs/cifar10.yml` for example). The configuration supports environment variables substitution, 
so that you can easily change base directory paths without changing the config file for each environment. 
The most common environment variables are:  
**SM_CHANNEL_TRAINING** — directory to all training data  
**SM_OUTPUT_DATA_DIR** — directory where logs for all runs will be stored
**SM_NUM_CPUS** - number of used CPUs for dataloader
### Start training locally
Download CIFAR10 dataset running all cells in `notebooks/Cifar10.ipynb`, 
the dataset will appear in `data/cifar10` folder.
```bash
docker exec -it torchok bash
cd torchok
SM_NUM_CPUS=8 SM_CHANNEL_TRAINING=./data/cifar10 SM_OUTPUT_DATA_DIR=/tmp python train.py --config config/classification_resnet_example.yml
```
### Start SageMaker Training Jobs
Start the job using one of the 
[AWS SageMaker instances](https://docs.aws.amazon.com/sagemaker/latest/dg/notebooks-available-instance-types.html).
You have 2 ways to provide data inside your training container:
- Slow downloaded S3 bucket: `s3://<bucket-name>/<dirpath>`. Volume size is needed to be set when you use S3 bucket. 
  For other cases it can be omitted.
- Fast FSx access: `fsx://<file-system-id>/<mount-name>/<directory>`. To create FSx filesystem follow 
  [this instructions](https://aws.amazon.com/blogs/machine-learning/speed-up-training-on-amazon-sagemaker-using-amazon-efs-or-amazon-fsx-for-lustre-file-systems/)

Example with S3:
```bash
python run_sagemaker.py --config configs/cifar10.yml --input_path s3://sagemaker-mlflow-main/cifar10 --instance_type ml.g4dn.xlarge --volume_size 5
```
Example with FSx:
```bash
python run_sagemaker.py --input_path fsx://fs-0f79df302dcbd29bd/z6duzbmv/tz_jpg --config configs/siloiz_pairwise_xbm_resnet50_512d.yml --instance_type ml.g4dn.xlarge
```
In case something isn't working inside the Sagemaker container you can debug your model locally. 
Specify `local_gpu` instance type when starting the job:
```bash
python run_sagemaker.py --config configs/cifar10.yml --instance_type local_gpu --volume_size 5 --input_path file://../data/cifar10
``` 

## Run tests
```bash
docker exec -it torchok bash
cd torchok
python -m unittest discover -s tests/ -p "test_*.py"
```

## Differences in configs sagemaker vs local machine
### 1. Path to data folder
#### sagemaker
```yml
data:
  dataset_name: ExampleDataset
  common_params:
    data_folder: "${SM_CHANNEL_TRAINING}"
```
#### local machine
```yml
data:
  dataset_name: ExampleDataset
  common_params:
    data_folder: "/path/to/data"
```

### 2. Path to artifacts dir
#### sagemaker
```yml
log_dir: '/opt/ml/checkpoints'
```

#### local machine
```yml
log_dir: '/tmp/logs'
```
### 3. Restore path
`do_restore` is a special indicator which was designed to be used for SageMaker spot instances training. 
With this indicator you can debug your model locally and be free to leave the `restore_path` pointing to some
common directory like `/opt/ml/checkpoints`, where TorchOk will search the checkpoints for.
#### sagemaker
```yml
restore_path: '/opt/ml/checkpoints'
do_restore: '${SM_USER_ENTRY_POINT}'
```

#### local machine
```yml
restore_path: '/opt/ml/checkpoints'
do_restore: '${SM_USER_ENTRY_POINT}'
```

## Mlflow
To have more convenient logs it is recommended to name your experiment as ```project_name-developer_name```, so that all your experiments related to this project will be under one tag in mlflow
```yml
experiment_name: &experiment_name fips-roman
```
State all the model parameters in ```mlflow.runName``` in logger params
```yml
logger:
  logger: mlflow
  experiment_name: *experiment_name
  tags:
      mlflow.runName: "siloiz_contrastive_xbm_resnet50_512d"
  save_dir: "s3://sagemaker-mlflow-main/mlruns"
  secrets_manager:
      region: "eu-west-1"
      mlflow_secret: "acme/mlflow"
```
