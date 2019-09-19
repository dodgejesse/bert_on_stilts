FROM python:3.6.8-stretch

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PATH /usr/local/nvidia/bin/:$PATH

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64

WORKDIR /stage

# Install base packages.
RUN apt-get update --fix-missing && apt-get install -y \
    bzip2 \
    ca-certificates \
    curl \
    gcc \
    git \
    libc-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    libevent-dev \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

#RUN pip install --no-cache-dir -q http://download.pytorch.org/whl/cu80/torch-0.4.0-cp36-cp36m-linux_x86_64.whl
#RUN pip install torchvision


COPY requirements.txt requirements.txt 
RUN pip install -r requirements.txt


COPY ./ bert_on_stilts/ 
RUN pip install --editable bert_on_stilts/

#CMD ["python", "beaker_pytorch/vgg_on_cifar.py"]
