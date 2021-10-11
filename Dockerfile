# Approximately 10 min to build

FROM nvidia/cuda:11.1-devel-ubuntu18.04
# Python
ARG python_version=3.7
ARG SSH_PASSWORD=password

# https://docs.docker.com/engine/examples/running_ssh_service/
# Last is SSH login fix. Otherwise user is kicked off after login
RUN apt-get update && apt-get install -y openssh-server && \
    mkdir /var/run/sshd && echo "root:$SSH_PASSWORD" | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    echo "export VISIBLE=now" >> /etc/profile && \
    mkdir /root/.ssh && chmod 700 /root/.ssh && touch /root/.ssh/authorized_keys && \
    chmod 644 /root/.ssh/authorized_keys

ENV NOTVISIBLE "in users profile"
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

# Install Miniconda
RUN mkdir -p $CONDA_DIR && \
    apt-get update && \
    apt-get install -y wget git vim htop zip libhdf5-dev g++ graphviz libgtk2.0-dev libgl1-mesa-glx \
    openmpi-bin nano cmake libopenblas-dev liblapack-dev libx11-dev && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash /Miniconda3-latest-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-latest-Linux-x86_64.sh

RUN wget "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -O "awscliv2.zip" && \
    unzip awscliv2.zip && ./aws/install && rm -r aws && rm awscliv2.zip && \
    wget https://dvc.org/deb/dvc.list -O /etc/apt/sources.list.d/dvc.list && apt-get update && apt-get install dvc

COPY ./environment.yml /torchok/environment.yml

# Install Data Science essential
RUN conda config --set remote_read_timeout_secs 100000.0 && \
    conda init && \
    conda update -n base -c defaults conda && \
    conda env create -f torchok/environment.yml && \
    conda clean -yt && \
    echo "conda activate torchok" >> /root/.bashrc && \
    echo "cd /" >> /root/.bashrc

RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    /opt/conda/envs/torchok/bin/python -m pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && \
    cd /

ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV LIBRARY_PATH /usr/local/cuda/lib64:/lib/x86_64-linux-gnu:$LIBRARY_PATH
ENV CUDA_HOME /usr/local/cuda

# aws cli configuration
ENV AWS_ACCESS_KEY_ID ""
ENV AWS_SECRET_ACCESS_KEY ""
ENV AWS_DEFAULT_REGION "eu-west-1"
ENV AWS_DEFAULT_OUTPUT "json"

# To access the container from the outer world
ENV SSH_PUBLIC_KEY ""

# To be able to add SSH key on docker run --env ... and to get important environment variables in SSH's bash
# writing env variables to /etc/profile as mentioned here:
# https://docs.docker.com/engine/examples/running_ssh_service/#environment-variables
RUN echo '#!/bin/bash\n \
echo $SSH_PUBLIC_KEY >> /root/.ssh/authorized_keys\n \
echo "export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID" >> /etc/profile\n \
echo "export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY" >> /etc/profile\n \
echo "export AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION" >> /etc/profile\n \
echo "export AWS_DEFAULT_OUTPUT=$AWS_DEFAULT_OUTPUT" >> /etc/profile\n \
echo "export CONDA_DIR=$CONDA_DIR" >> /etc/profile\n \
echo "export PATH=$CONDA_DIR/bin:$PATH" >> /etc/profile\n \
echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH" >> /etc/profile\n \
echo "export LIBRARY_PATH=/usr/local/cuda/lib64:/lib/x86_64-linux-gnu:$LIBRARY_PATH" >> /etc/profile\n \
echo "export CUDA_HOME=/usr/local/cuda" >> /etc/profile\n \
echo "alias juplabstart=\"nohup jupyter lab --ip 0.0.0.0 --allow-root > jup.log 2>&1 &\"" >> /etc/profile\n \
echo "alias jupnotestart=\"nohup jupyter notebook --ip 0.0.0.0 --allow-root > jup.log 2>&1 &\"" >> /etc/profile\n \
echo "alias jupkill=\"kill -9 \$(pgrep -f jupyter)\"" >> /etc/profile\n \
echo "alias tbkill=\"kill -9 \$(pgrep -f tensorboard)\"" >> /etc/profile\n \
/usr/sbin/sshd -D' \
>> /bin/start.sh

RUN echo '#!/bin/bash\n \
nohup tensorboard --bind_all --logdir=$1 > tb.log 2>&1 & echo "see tb.log for address"' \
>> /bin/tbstart.sh && chmod +x /bin/tbstart.sh

COPY . /torchok

EXPOSE 8888 6006 22
ENTRYPOINT bash /bin/start.sh
