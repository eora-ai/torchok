import argparse
import tarfile
import uuid
import time
import re
import boto3
import sagemaker
from sagemaker.pytorch.estimator import PyTorch

# project bucket where training codes will be uploaded to (CHANGE FOR YOUR PROJECT)
PROJECT_BUCKET = "sagemaker-mlflow-main"
# project Docker image
IMAGE_URI = "018518131176.dkr.ecr.eu-west-1.amazonaws.com/research/sagemaker:pt1.9.0-pl1.4.5-cu111-ubuntu18.04"
# role for Sagemaker: AmazonSageMakerFullAccess, SecretsManagerReadWrite, AmazonS3FullAccess
ROLE = "arn:aws:iam::018518131176:role/Sagemaker-Basic-Role"
REGION_NAME = "eu-west-1"  # region where Sagemaker training jobs will operate
SUBNETS = ['subnet-04d916f31d063c074']  # subnets to be able to connect to MLflow database and Internet
SECURITY_GROUPS = ['sg-055829cbf34ffaebb']  # security groups to be able to connect to MLflow database
KEY_PREFIX = "source_codes"  # used to specify location of the code uploaded inside S3 bucket
aws_name = boto3.client('sts').get_caller_identity()['Arn'].split('/')[1]


def get_job_name():
    # replace bc dot is not an available symbol
    jobname = f'{aws_name}-{time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())}'
    return re.sub(r'[^a-zA-Z0-9-]{1}', '-', jobname)


def run_locally(args, hyperparameters):
    hyperparameters['job_link'] = 'local'
    print("Initializing estimator...")
    estimator = PyTorch(
        image_uri=IMAGE_URI,
        role=ROLE,
        entry_point='train.py',
        source_dir='.',
        subnets=SUBNETS,
        security_group_ids=SECURITY_GROUPS,
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        hyperparameters=hyperparameters,
        max_run=args.max_run,
        volume_size=args.volume_size,
        debugger_hook_config=False
    )
    print("Estimator initialized.")
    print("Starting fit...")
    estimator.fit(args.input_path)
    print(f"Training Job started. ")


def run_on_cluster(args, hyperparameters):
    # generate random name for source code archive in order to don't mixup teammates' sources
    uuid_name = uuid.uuid4()
    checkpoint_s3_uri = f"s3://{PROJECT_BUCKET}/{uuid_name}"
    source_arch_name = f"/tmp/{uuid_name}.tar.gz"
    job_name = get_job_name()
    job_link = f'https://{REGION_NAME}.console.aws.amazon.com/sagemaker/home?region={REGION_NAME}#/jobs/{job_name}'
    hyperparameters['job_link'] = job_link

    print("Archiving source files...")
    with tarfile.open(source_arch_name, mode="w:gz") as tar_gz:
        tar_gz.add(".")
    print("Source files archived.")

    print("Connecting to Sagemaker session...")
    boto3_session = boto3.session.Session(region_name=REGION_NAME)
    sess = sagemaker.Session(boto3_session, default_bucket=PROJECT_BUCKET)
    source_dir = sess.upload_data(source_arch_name, key_prefix=KEY_PREFIX)
    print("Sagemaker session connected.")

    if args.input_path.startswith('fsx://'):
        input_path_split = args.input_path[6:].split('/')
        file_system_id = input_path_split[0]
        directory_path = '/' + '/'.join(input_path_split[1:])
        print(file_system_id, directory_path)
        inputs = sagemaker.inputs.FileSystemInput(file_system_id, "FSxLustre",
                                                  directory_path=directory_path,
                                                  file_system_access_mode="ro")
    else:
        inputs = args.input_path

    print("Initializing estimator...")
    estimator = PyTorch(
        image_uri=IMAGE_URI,
        role=ROLE,
        entry_point='train.py',
        source_dir=source_dir,
        subnets=SUBNETS,
        sagemaker_session=sess,
        security_group_ids=SECURITY_GROUPS,
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        hyperparameters=hyperparameters,
        max_run=args.max_run,
        volume_size=args.volume_size,
        debugger_hook_config=False,
        use_spot_instances=args.spot,
        max_wait=args.max_run if args.spot else None,
        checkpoint_s3_uri=checkpoint_s3_uri
    )
    print("Estimator initialized.")
    print("Starting fit...")
    estimator.fit(inputs, wait=False, job_name=job_name)
    print(f"Training Job started. "
          f"Track job status at {job_link}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config for training pipeline configuration relative to root of this project')
    parser.add_argument('--instance_type', type=str, default='local_gpu',
                        help='Instance type to start. Possible values are:\n\
                        1) local - docker-compose training will be started on the current host without GPU support;\n\
                        2) local_gpu - docker-compose training will be started on the current host with GPU support;\n\
                        3) <sagemaker_instance> - any instance available for training jobs in SageMaker, e.g. ml.g4dn.xlarge \
                        (for full list of available instance types follow the link: https://aws.amazon.com/sagemaker/pricing \
                        and https://aws.amazon.com/ec2/instance-types). Default: local_gpu')
    parser.add_argument('--instance_count', type=int, default=1,
                        help='Number of instances to start training on (make sure your code works in distributed mode). Default: 1')
    parser.add_argument('--input_path', type=str, required=True,
                        help='S3/FSx or local directory location of the input training data. Prefixes must correspond to file system type. \
                            For local training they need to be in form of file://<absolute_or_relative_path>, \
                            for training on a cluster with slow data loading from S3 bucket - s3://<path/to/data>, \
                            for training on a cluster with fast data access to FSx for Lustre - fsx://<file-system-id>/<mount-name>/<directory>')
    parser.add_argument('--volume_size', type=int, required=False, default=30,
                        help='Size in GB of the EBS volume to use for storing input data during training')
    parser.add_argument('--max_run', type=int, default=24 * 60 * 60, help='Timeout in seconds for training (default: 24 * 60 * 60). \
                        Instance will stop when this time is exceeded')
    parser.add_argument('--spot', action="store_true", required=False,
                        help='Use spot instances for training (70% costs are saved in average)')

    args, _ = parser.parse_known_args()

    if args.instance_type.startswith('local') and not args.input_path.startswith('file://'):
        raise ValueError(
            f'For local training there must be input_path specified with a prefix file://. Given: {args.input_path}')
    elif not args.instance_type.startswith('local') and \
            not args.input_path.startswith('s3://') and not args.input_path.startswith('fsx://'):
        raise ValueError(f'For training on a cluster there must be input_path specified with a '
                         f'prefix s3:// or fsx://. Given: {args.input_path}')

    hyperparameters = {'config': args.config}

    if args.instance_type.startswith('local'):
        run_locally(args, hyperparameters)
    else:
        run_on_cluster(args, hyperparameters)
