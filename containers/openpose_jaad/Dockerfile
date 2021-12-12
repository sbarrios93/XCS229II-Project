FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04

ENV OPENPOSE="/root/app/openpose"
ENV PEDESTRIANS="/root/pedestrians"
# WORKDIR /root
# COPY ./app /root/app

# RUN mkdir output

# # Install dependencies required
RUN apt-get update && apt-get install -y wget git
# DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
# python3-dev python3-pip git g++ wget libssl-dev curl libprotobuf-dev protobuf-compiler libopencv-dev \
# libgoogle-glog-dev libboost-all-dev libhdf5-dev libatlas-base-dev \
# git vim

# WORKDIR /root/app
# RUN echo ":::Cloning openpose"
# RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git

# # Clone JAAD
# RUN echo ":::Cloning JAAD"
# RUN git clone https://github.com/ykotseruba/JAAD.git jaad && cd jaad

# WORKDIR /root/app/jaad
# RUN echo ":::Download JAAD Clips"
# RUN chmod +x download_clips.sh
# RUN ./download_clips.sh

# # Install external dependencies
# WORKDIR /root/deps

# # Install CMAKE
# RUN curl -fsSL -o cmake-3.22.1-linux-x86_64.sh https://github.com/Kitware/CMake/releases/download/v3.22.1/cmake-3.22.1-linux-x86_64.sh
# RUN chmod +x cmake-3.22.1-linux-x86_64.sh && ./cmake-3.22.1-linux-x86_64.sh --skip-license --prefix=/usr/local

# # Install OpenPose
# WORKDIR /root/app/openpose/build 
# RUN cmake -DBUILD_PYTHON=ON ..
# RUN make -j`nproc`
# WORKDIR /root/app/openpose

# # Install model_25b
# WORKDIR /root/app/openpose/experimental_models/1_25BSuperModel11FullVGG/body_25b
# RUN curl -fsSL -o pose_iter_XXXXXX.caffemodel posefs1.perception.cs.cmu.edu/OpenPose/models/pose/1_25BSuperModel11FullVGG/body_25b/pose_iter_XXXXXX.caffemodel
# WORKDIR /root/app/openpose
# RUN cp -r experimental_models/1_25BSuperModel11FullVGG/body_25b models/pose/
# RUN mv /root/app/pose_deploy.prototxt /root/app/openpose/models/pose/body_25b/pose_deploy.prototxt

# Install Conda
ENV CONDA_DIR $HOME/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh -O ~/miniconda.sh
RUN chmod +x ~/miniconda.sh
RUN ~/miniconda.sh -b -p $CONDA_DIR
RUN rm ~/miniconda.sh
# make non-activate conda commands available
ENV PATH=$CONDA_DIR/bin:$PATH
# make conda activate command available from /bin/bash --login shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile
# make conda activate command available from /bin/bash --interative shells
RUN conda init bash


# Set Conda Environment
RUN conda create -y -n pedestrians python=$PYTHON_VERSION pathlib pyyaml opencv

# Download git repo

RUN git clone https://github.com/sbarrios93/XCS229II-Project.git pedestrians
WORKDIR /root/pedestrians
RUN cp -r /root/app/jaad/ /root/pedestrians/data
RUN conda activate pedestrians
RUN python extract_jaad_images.py
