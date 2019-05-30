#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/

cd src/cuda/
echo "Compiling SparsePSA layer kernels by nvcc..."
nvcc -c -o SparsePSA_layer.cu.o SparsePSA_cuda_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52
cd ../../
python build.py
