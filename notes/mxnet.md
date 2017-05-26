## README

Install mxnet for python2/3 on Ubuntu 16.04.

## Instructions

1. (Optional, for GPU support only) Install cuda_8.0.61_375.26_linux.run from Nvdia (Install driver:no, install cuda toolkit: yes).

2. (Optional, for GPU support only) Download and extract cudnn-8.0-linux-x64-v5.1.tgz from https://developer.nvidia.com/rdp/cudnn-download.

```
cd cuda/
sudo cp include/cudnn.h /usr/include
sudo cp lib64/libcudnn* /usr/lib/x86_64-linux-gnu/
sudo chmod a+r /usr/lib/x86_64-linux-gnu/libcudnn*
```

3. Install mxnet via pip:

```
sudo apt-get install python-pip
pip install --upgrade pip
pip install --user mxnet-cu80
pip3 install --user mxnet-cu80
```

To verify mxnet installation, test the following Python file:
```Python
import mxnet as mx

a = mx.nd.ones((2, 3), mx.gpu())
b = a * 2 + 1
print(b.asnumpy())
```