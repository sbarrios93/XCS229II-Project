#!/bin/bash
SKIP_NVIDIA_DOCKER=false

while [ "$1" != "" ]; do
    case $1 in
        --skip-nvidia-docker)
            SKIP_NVIDIA_DOCKER=true
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

if [[ $SKIP_NVIDIA_DOCKER == false ]]; then
    ## Set up NVIDIA Docker Deps
    # Set up stable repo
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
    && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
    && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

    # install nvidia-docker2
    apt-get update && apt-get install -y --no-install-recommends nvidia-docker2

    sudo tee /etc/docker/daemon.json <<EOF
    {
        "runtimes": {
            "nvidia": {
                "path": "/usr/bin/nvidia-container-runtime",
                "runtimeArgs": []
            }
        }
    }
EOF
    # restart docker
    sudo pkill -SIGHUP dockerd
    systemctl restart docker
fi

# make directory for app
mkdir -p app
cd app

