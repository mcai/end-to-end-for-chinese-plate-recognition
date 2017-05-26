## README

opencv.

## Instructions for Installing opencv 3.2 for Python 2/3 on Ubuntu 16.04.

Install prerequisites:

```
sudo apt-get install --assume-yes build-essential cmake git
sudo apt-get install --assume-yes build-essential pkg-config unzip ffmpeg qtbase5-dev python-dev python3-dev python-numpy python3-numpy
sudo apt-get install --assume-yes libopencv-dev libgtk2.0-dev libgtk-3-dev libdc1394-22 libdc1394-22-dev libjpeg-dev libpng12-dev libtiff5-dev libjasper-dev
sudo apt-get install --assume-yes libavcodec-dev libavformat-dev libswscale-dev libxine2-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev
sudo apt-get install --assume-yes libv4l-dev libtbb-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev
sudo apt-get install --assume-yes libvorbis-dev libxvidcore-dev v4l-utils
```

Download and extract opencv 3.2.0 from https://github.com/opencv/opencv/releases.
Download and extract opencv_contrib 3.2.0 from https://github.com/opencv/opencv_contrib/releases.

```
mkdir build
cd build/

cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_C_EXAMPLES=ON \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D OPENCV_EXTRA_MODULES_PATH=~/Tools/opencv_contrib-3.2.0/modules \
	-D BUILD_EXAMPLES=ON \
	-D WITH_OPENGL=ON ..

make -j $(($(nproc) + 1))

sudo make install
sudo /bin/bash -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'
sudo ldconfig
sudo apt-get update
```

and reboot.

## Instructions for Installing opencv 3.2 for Python 3 on Mac OS Sierra.

```
brew tap homebrew/science
brew install opencv3 --with-contrib --with-python3

brew edit opencv3, amd comment lines with errors.

echo /usr/local/opt/opencv3/lib/python3.6/site-packages >> /usr/local/lib/python3.6/site-packages/opencv3.pth
```

