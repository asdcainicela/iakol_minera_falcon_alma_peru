
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

Copy Dockerfile

mkdir -p ~/alma/

git clone  ... git

cd [] /docker
```bash
docker build --no-cache -t l4tml_alma .
docker build -t l4tml_alma .
```
cd ..

desde la carpeta creada por git 
```bash
docker run --runtime nvidia --privileged -d --ipc=host \
  --device /dev/bus/usb:/dev/bus/usb \
  --device /dev/video0:/dev/video0 \
  --device /dev/video1:/dev/video1 \
  --network=host \
  -v $(pwd):/app/alma \
  --name cnm_container \
  --restart unless-stopped l4tml_alma \
  /bin/bash -c 'jupyter lab --ip 0.0.0.0 --port 8888 --allow-root --no-browser --ServerApp.token "alma" --ServerApp.password="" &> /var/log/jupyter.log & tail -f /var/log/jupyter.log'
```