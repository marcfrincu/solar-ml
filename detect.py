import os
import pathlib

#if "models-master" in pathlib.Path.cwd().parts:
#    while "models-master" in pathlib.Path.cwd().parts:
#        os.chdir('..')
#elif not pathlib.Path('models-master').exists():
#    !git clone --depth 1 https://github.com/tensorflow/models

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

MODEL_NAME = 'C:/Users/fmarc/Downloads/models-master/models-master/research/object_detection/inference_graph' # change this to your own path
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb' # change this to your own path
PATH_TO_LABELS = 'C:/Users/fmarc/Downloads/models-master/models-master/research/object_detection/training/labelmap.pbtxt' # change this to your own path
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def load_model():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

def run_inference_for_single_image(image, graph, tensor_dict, sess):
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                            feed_dict={image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

import zwoasi as asi
import cv2

ASI_LIBRARY_PATH = 'C:/Users/fmarc/Downloads/ASI_Windows_SDK_V1.27/ASI SDK/lib/x64/ASICamera2.dll' # change this to your own path
asi.init(ASI_LIBRARY_PATH) 

from datetime import datetime

def run_inference_for_stream(camera, detection_graph, save_frames, livestream):
    try:
        image_index = 0
        if (save_frames):
            try:
                os.mkdir('saved_frames_with_detection')
            except Exception as e:
                print ('Image folder already exists. Skipping folder creation.')
        with detection_graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                'num_detections', 'detection_boxes', 'detection_scores',

                'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)

                df_last = True
                while True:
                    if (livestream):
                        # print some stats
                        settings = camera.get_control_values()
                        df = camera.get_dropped_frames()
                        gain = settings['Gain']
                        exposure = settings['Exposure']
                        if df != df_last:
                            print('   Gain {gain:d}  Exposure: {exposure:f} Dropped frames: {df:d}'
                                .format(gain=settings['Gain'], exposure=settings['Exposure'], df=df))
                            df_last = df
                        image_np = camera.capture_video_frame()
                    else:
                        ret, image_np = camera.read()
                        if not ret:
                            print("Can't receive frame (stream end?). Exiting ...")
                            break

                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    # Actual detection.
                    output_dict = run_inference_for_single_image(image_np, detection_graph, tensor_dict, sess)
                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        output_dict['detection_boxes'],
                        output_dict['detection_classes'],
                        output_dict['detection_scores'],
                        category_index,
                        instance_masks=output_dict.get('detection_masks'),
                        use_normalized_coordinates=True,
                        line_thickness=4)
                    cv2.imshow('object_detection', cv2.resize(image_np, (800, 600)))
                    if (save_frames and len(output_dict) > 0):
                        current_dateTime = datetime.now()
                        timestamp = str(current_dateTime.year) + str(current_dateTime.month) + str(current_dateTime.day) + "-" + str(current_dateTime.hour) + str(current_dateTime.minute) + str(current_dateTime.second)
                        filename = 'image_' + timestamp + '_' + str(image_index) + '.png'
                        print ('Saving image with detections to ' + filename)
                        cv2.imwrite('saved_frames_with_detection/' + filename, image_np)
                        image_index += 1
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        if livestream:
                            camera.stop_video_capture()
                        else:
                            camera.release()
                        cv2.destroyAllWindows()
                        break
    except Exception as e:
        print(e)
        if livestream:
            camera.stop_video_capture()
        else:
            camera.release()

# usage ./detect.py -gain 37 -exposure 5000 -fromrecording RECORDING_PATH -saveframes
# usage ./detect.py -help
def main():
    args = sys.argv[1:]

    if ('-help' in args):
        print('Usage ./detect.py -gain 37 -exposure 5000 -fromrecording RECORDING_PATH -saveframes\nAll arguments are optional.\n-gain positive integer (0-100) sets the gain value.\n-exposure positive integer sets exposure time in microseconds.\n-fromrecording specifies an existing recording. Gain and exposure parameters have no effect in this case.\n-saveframes saves frames with detections inside the saved_frames_with_detection folder.')
        return

    save_frames = False
    if ('-saveframes' in args):
        save_frames = True

    livestream = True
    if ('-fromrecording' in args):
        video_index = args.index('-fromrecording')
        if (len(args) > video_index + 1):
            print(args[video_index + 1])
            camera = cv2.VideoCapture(args[video_index + 1])
            livestream = False
    else:
        num_cameras = asi.get_num_cameras()
        if num_cameras == 0:
            raise ValueError('No ZWO ASI cameras found')
        else:
            print('Detected ' + str(num_cameras))

        camera_id = 0  # use first camera from list
        cameras_found = asi.list_cameras()
        print(cameras_found)

        camera = asi.Camera(0)
        camera_info = camera.get_camera_property()
        print(camera_info)

        # Use minimum USB bandwidth permitted
        camera.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, camera.get_controls()['BandWidth']['MinValue'])

        gain_value = 37
        if ('-gain' in args):
            gain_index = args.index('-gain')
            if (len(args) > gain_index + 1):
                gain_value = int(args[gain_index + 1])
        camera.set_control_value(asi.ASI_GAIN, gain_value)

        exposure_value = 5000  # in microseconds
        if ('-exposure' in args):
            exposure_index = args.index('-exposure')
            if (len(args) > gain_index + 1):
                exposure_value = int(args[exposure_index + 1])
        camera.set_control_value(asi.ASI_EXPOSURE, exposure_value)

        #camera.set_control_value(asi.ASI_WB_B, 99)
        #camera.set_control_value(asi.ASI_WB_R, 75)
        #camera.set_control_value(asi.ASI_GAMMA, 50)
        camera.set_control_value(asi.ASI_BRIGHTNESS, 9)
        camera.set_control_value(asi.ASI_FLIP, 1) # flip horizontally

        camera.set_image_type(asi.ASI_IMG_RGB24)

        # Set the timeout, units are ms
        timeout = (camera.get_control_value(asi.ASI_EXPOSURE)[0] / 1000) * 2 + 500
        camera.default_timeout = timeout

        camera.start_video_capture()     

    detection_graph = load_model()
    run_inference_for_stream(camera, detection_graph, save_frames, livestream)

if __name__ == "__main__":
    main()
