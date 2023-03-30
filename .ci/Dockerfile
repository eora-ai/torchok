# Approximately 10 min to build

# SageMaker PyTorch image
# All available images can be found here: 
# https://github.com/aws/deep-learning-containers/blob/master/available_images.md
FROM 763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-training:1.12.1-gpu-py38-cu113-ubuntu20.04-sagemaker

RUN pip install pymysql mlflow==2.2.1 boto3 --timeout 120
WORKDIR /app
COPY . .
RUN pip install .

ENTRYPOINT [ "python", "-m", "torchok" ]
