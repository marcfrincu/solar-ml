# solar-ml
My repository containing code for identifying solar features in real-time or prerecorded videos. It uses machine learning.

For a full discussion behind the idea and how to train your own model see https://starscape-experiences.space/index.php/2023/01/05/detecting-solar-features-automatically-using-machine-learning/.

The code is based on [this article](https://medium.com/analytics-vidhya/detecting-custom-objects-on-video-stream-with-tensorflow-and-opencv-34406bd0ec9).

## Requirements
* Python 3.7.3
* Tensorflow 1.15.0 (machine learning platform): _pip install tensorflow==1.15.0_
* or Tf-slim: _pip install --upgrade tf_slim_
* Keras 2.2.4 (neural network library that contains the algorithms we will use): _pip install keras==2.2.4_
* [ASI ZWO SDK](https://astronomy-imaging-camera.com/software-drivers). Update line 78 in the detect.py script to reflect the path of your ZWO library.
* zwoasi library: _pip install zwoasi_
* Other required dependencies. My [article](https://starscape-experiences.space/index.php/2023/01/05/detecting-solar-features-automatically-using-machine-learning/) explains the process.

## Using
Download entire git archive and unzip. 
Run python ./detect.py -help for information on the options

Currently you can:
* Run the model on a prerecorded video: _python ./detect.py -fromrecording PATH_TO_RECORDING_
* Run the model on a live stream using an USB ZWO ASI camera: _python ./detect.py -gain 37 -exposure 5000_
* Run the model and save frames with detections: _python ./detect.py -saveframes_. This works on both recorded and live streams. It creates a folder called _saved_frames_with_detection_ and writes timestamped frames in there

## Disclaimer
The code is distributed as is under no guarantee.

## License
GPL3
