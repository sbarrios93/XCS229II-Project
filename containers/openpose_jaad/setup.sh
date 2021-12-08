#!/bin/bash
SKIP_JAAD=false

while [ "$1" != "" ]; do
    case $1 in
        --skip-jaad)
            SKIP_JAAD=true
        ;;
        *)
            exit 1
        ;;
    esac
    shift
done

# check if docker is installed
echo ":::Checking... if Docker is installed"
if [ -x "$(command -v docker)" ]; then
    echo "Docker Installed (version $(docker --version))"
else
    echo "Not installed"
    echo ":::Installing Docker"
    curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh
    dock    rm get-docker.sh # delete file after use
fi

## Set up NVIDIA Docker Deps
# Set up stable repo
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# install nvidia-docker2
apt-get update && apt-get install -y --no-install-recommends nvidia-docker2

# restart docker
systemctl restart docker

# we will need ffmpeg
apt-get install -y --no-install-recommends ffmpeg

# make directory for app
mkdir app && cd app

# Clone openpose
echo ":::Cloning openpose"
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git

# Clone JAAD
echo ":::Cloning JAAD"
git clone https://github.com/ykotseruba/JAAD.git jaad

# if we are not skipping jaad clips, then we need to download them and split them
if [[ $SKIP_JAAD == false ]]; then
    # do not skip downloading and splitting jaad clips
    echo ":::Download JAAD Clips"
    chmod +x jaad/download_clips.sh
    ./jaad/download_clips.sh
    
    echo ":::Split Clips to Frames"
    chmod +x jaad/split_clips.sh
    ./jaad/split_clips.sh
fi