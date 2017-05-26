# Introduction

Chinese Plate Number Recognition powered by OpenCV and MxNet written in Python 3.

# Dependencies

* Python 3

* OpenCV 3.2

* MxNet 0.9.5

Please see notes/opencv.md and notes/mxnet.md for installing OpenCV and MxNet, respectively.

The following combinations were tested successfully:

* Python 3.5 + OpenCV 3.2.1 + MxNet 0.9.5(with GPU) + Ubuntu 16.04.2

* Python 3.6 + OpenCV 3.2.1 + MxNet 0.9.5(without GPU) + Mac OS Sierra (10.12.5)

# File Structure

* fonts/ - fonts that are used to generate plate number images.

* images/ - background images.

* models/ - trained MxNet models.

* no_plates/ - noise images which contain no plate numbers.

* plates/ - plate number images for training.

* plates_to_test/ - plate number images for testing.

# Usage

1. Generate plate images for training:

`python3 ./generate_plates.py`

2. Training:

`python3 ./train.py`

3. Recognition:

`python3 ./recognize.py`
 
