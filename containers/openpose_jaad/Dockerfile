FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04

WORKDIR /root
COPY ./app /root/app

RUN mkdir output

# Install dependencies required
RUN apt-get update && \
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
python3-dev python3-pip git g++ wget libssl-dev curl libprotobuf-dev protobuf-compiler libopencv-dev \
libgoogle-glog-dev libboost-all-dev libhdf5-dev libatlas-base-dev

# Install external dependencies
WORKDIR /root/deps

# Install CMAKE
RUN curl -fsSL -o cmake-3.22.1-linux-x86_64.sh https://github.com/Kitware/CMake/releases/download/v3.22.1/cmake-3.22.1-linux-x86_64.sh
RUN chmod +x cmake-3.22.1-linux-x86_64.sh && ./cmake-3.22.1-linux-x86_64.sh --skip-license --prefix=/usr/local

# Install OpenPose
WORKDIR /root/app/openpose/build 
RUN  && cmake -DBUILD_PYTHON=ON ..
RUN make -j`nproc`
WORKDIR /root/openpose

