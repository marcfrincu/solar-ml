# solar-ml
My repository containing code for identifying solar features in real-time or prerecorded videos. It uses machine learning.

For a full discussion behind the idea and how to train your own model see https://starscape-experiences.space/index.php/2023/01/05/detecting-solar-features-automatically-using-machine-learning/.

The code is based on [this article](https://medium.com/analytics-vidhya/detecting-custom-objects-on-video-stream-with-tensorflow-and-opencv-34406bd0ec9).

## Requirements
* Python 3.7.9 (works on 3.7.3 as well)
* Tensorflow 1.15.0 (machine learning platform): _pip install tensorflow==1.15.0_
* or Tf-slim: _pip install --upgrade tf_slim_
* Keras 2.2.4 (neural network library that contains the algorithms we will use): _pip install keras==2.2.4_
* [ASI ZWO SDK](https://astronomy-imaging-camera.com/software-drivers). Update line 78 in the detect.py script to reflect the path of your ZWO library.
* zwoasi library: _pip install zwoasi_
* matplotlib library: _pip install matplotlib_
* Other required dependencies. My [article](https://starscape-experiences.space/index.php/2023/01/05/detecting-solar-features-automatically-using-machine-learning/) explains the process.

## Using
Download entire git archive and unzip. 

Download [TensorFlow models](https://github.com/tensorflow/models) and copy the _object_detection_ folder in your _solar-ml_ unzipped folder.

Copy the _inference_graph_ and _training_ folders from the _solar_ml_ folder in the _object_detection_folder_.

Run python ./detect.py -help for information on the options.

Currently you can:
* Run the model on a prerecorded video: _python ./detect.py -fromrecording PATH_TO_RECORDING_
* Run the model on a live stream using an USB ZWO ASI camera: _python ./detect.py -gain 37 -exposure 5000_
* Run the model and save frames with detections: _python ./detect.py -saveframes_. This works on both recorded and live streams. It creates a folder called _saved_frames_with_detection_ and writes timestamped frames in there
* Update exposure and gain at runtime by editing the _config_ file
## Disclaimer
The code is distributed as is under no guarantee.

## Tests
The code has been tested on a Pi4 with 4GB and CPU overcloaked at 2GHz.

The code has been tested on an Intel Core i7 8th Gen laptop running Windows 10.

It consumes about 600 Mb RAM.

## License
GPL3
