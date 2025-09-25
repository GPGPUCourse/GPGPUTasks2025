#!/bin/bash

# exit script on failure
set -ev
CUDA_RUNFILE=cuda_12.9.0_575.51.03_linux.run
wget https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/${CUDA_RUNFILE}
sudo apt install linux-headers-$(uname -r)
chmod +x ${CUDA_RUNFILE}
sudo ./${CUDA_RUNFILE} --silent --toolkit
# permamently add NVCC to PATH:
echo "export CUDA_HOME=/usr/local/cuda" >> ~/.zshrc
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda/lib64" >> ~/.zshrc
echo "export PATH=\$PATH:\$CUDA_HOME/bin" >> ~/.zshrc
