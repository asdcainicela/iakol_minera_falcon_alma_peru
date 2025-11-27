https://jetsonhacks.com/2023/05/30/jetson-orin-nano-tutorial-ssd-install-boot-and-jetpack-setup/

$ git clone https://github.com/jetsonhacks/bootFromExternalStorage.git

$ cd bootFromExternalStorage

$ ./get_jetson_files.sh

$ ./flash_jetson_external_storage.sh

$ sudo apt update

$ sudo apt upgrade

$ sudo apt install nvidia-jetpack

sudo pip3 install -U jetson-stats


## Configure VNC

    # https://forums.developer.nvidia.com/t/vnc-connection-to-jetson-orin-nano/262958/4
    1. Connect an external monitor on your board
    2. Settings -> Sharing -> Screen Sharing: Active
    3. Press "Screen Sharing" -> Access Options -> Require a password
    4. Settings -> Sharing -> Remote Login: On
    5. $ gsettings set org.gnome.Vino require-encryption false

## Docker Config

    sudo apt-get install docker.io
    sudo apt-get install curl

    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt-get update
    sudo apt-get install -y nvidia-docker2
    sudo systemctl restart docker


    # install the container tools
    git clone https://github.com/dusty-nv/jetson-containers
    bash jetson-containers/install.sh
    # https://github.com/dusty-nv/jetson-containers/blob/master/docs/setup.md

Modify default runtime

    sudo nano /etc/docker/daemon.json
    # add default realtime line
    {
        "runtimes": {
            "nvidia": {
                "path": "nvidia-container-runtime",
                "runtimeArgs": []
            }
        },
        "default-runtime": "nvidia"
    }

    # restart to take effect
    sudo systemctl restart docker
    # verify
    sudo docker info | grep 'Default Runtime'
    #  Default Runtime: nvidia

Modify execution permissions to docker command

    sudo usermod -aG docker $USER

    # reboot to take effect
    sudo reboot
    
    # verify
    groups
    # ... docker ...

    # download base image
    docker pull dustynv/l4t-ml:r35.3.1

    jetson-containers run dustynv/l4t-ml:r35.3.1
    # --ipc=host important to run
    docker run --runtime nvidia -it --ipc=host --rm --network=host dustynv/l4t-ml:r35.3.1
    # test with GPIO
    docker run --runtime nvidia --privileged -it --ipc=host --rm --network=host dustynv/l4t-ml:r35.3.1



Copy Dockerfile

docker build --no-cache -t l4tml_briq .
    docker build -t l4tml_briq .

    docker run --runtime nvidia --privileged -d --ipc=host \
        --device /dev/bus/usb:/dev/bus/usb \
        --device /dev/video0:/dev/video0 \
        --device /dev/video1:/dev/video1 \
        --network=host \
        -v $(pwd):/briq_system/src \
        --name briq_container \
        --restart unless-stopped l4tml_briq \
        /bin/bash -c "jupyter lab --ip 0.0.0.0 --port 8888 --allow-root --NotebookApp.token='' --NotebookApp.password='' &> /var/log/jupyter.log & tail -f /var/log/jupyter.log"