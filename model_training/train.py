# -*- coding: utf-8 -*-
"""Copy of detection-training-colab.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VabWCrQ8005iYQsIV-EEosnTPHvvScqj

# [How to train an object detection model easy for free](https://www.dlology.com/blog/how-to-train-an-object-detection-model-easy-for-free/) | DLology Blog

## Configs and Hyperparameters

Support a variety of models, you can find more pretrained model from [Tensorflow detection model zoo: COCO-trained models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models), as well as their pipline config files in [object_detection/samples/configs/](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs).
"""

# Number of training steps.
#num_steps = 1000  # 200000
num_steps = 20000

# Number of evaluation steps.
num_eval_steps = 1000  # 50

# batch size, originally 12, apparently fits to GPU memory better
# the config files actually fix this to 24?
batch_size = 12

MODELS_CONFIG = {
    'ssd_mobilenet_v2': {
        'model_name': 'ssd_mobilenet_v2_coco_2018_03_29',
        'pipeline_file': 'ssd_mobilenet_v2_coco.config',
        'batch_size': batch_size
    },
    'ssd_mobilenetv2_oidv4': {
        'model_name': 'ssd_mobilenet_v2_oid_v4_2018_12_12',
        'pipeline_file': 'ssd_mobilenet_v2_oid_v4.config',
        'batch_size': batch_size
    },
    'ssd_mobilenet_v2_quantized_coco': {
        'model_name': 'ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03',
        'pipeline_file': 'ssd_mobilenet_v2_quantized_300x300_coco.config',
        'batch_size': batch_size
    },
    'ssdlite_mobilenet_v2_coco': {
        'model_name': 'ssdlite_mobilenet_v2_coco_2018_05_09',
        'pipeline_file': 'ssd_mobilenet_v2_coco.config',
        'batch_size': batch_size
    },

    # faster rcnn models
    'faster_rcnn_inception_resnet_v2_atrous_oid_v4': {
        'model_name': 'faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12',
        'pipeline_file': 'faster_rcnn_inception_resnet_v2_atrous_oid_v4.config',
        'batch_size': batch_size
    },
    'faster_rcnn_inception_v2_coco': {
        'model_name': 'faster_rcnn_inception_v2_coco_2018_01_28',
        'pipeline_file': 'faster_rcnn_inception_v2_coco.config',
        'batch_size': batch_size
    }
}

# Pick the model you want to use
# Select a model in `MODELS_CONFIG`.
# selected_model = 'ssd_mobilenet_v2'
selected_model = 'ssd_mobilenet_v2'

# Name of the object detection model to use.
MODEL = MODELS_CONFIG[selected_model]['model_name']

# Name of the pipline file in tensorflow object detection API.
pipeline_file = MODELS_CONFIG[selected_model]['pipeline_file']

# Training batch size fits in Colabe's Tesla K80 GPU memory for selected model.
batch_size = MODELS_CONFIG[selected_model]['batch_size']


replace___ = False
if replace___:
    if selected_model == 'ssd_mobilenetv2_oidv4':
        # replace pipeline config file...
        !wget https://gist.githubusercontent.com/bgalvao/d9db76d7fe9aa02fd930f4c8a11d500c/raw/e3b1849ceb555bb2239404e6e032c57096d4d6ca/ssd_mobilenet_v2_oid_v4.config -O /content/models/research/object_detection/samples/configs/ssd_mobilenet_v2_oid_v4.config
        !tail /content/models/research/object_detection/samples/configs/ssd_mobilenet_v2_oid_v4.config -n 15

    if selected_model == 'faster_rcnn_inception_resnet_v2_atrous_oid_v4':
        !wget https://bitbucket.org/!api/2.0/snippets/burnpnk/gn8p4K/66f377f0679715e7de5fee5f3359e93466977f23/files/faster_rcnn_inception_resnet_v2_atrous_oid_v4.config \
        -O /content/models/research/object_detection/samples/configs/faster_rcnn_inception_resnet_v2_atrous_oid_v4.config
        !tail /content/models/research/object_detection/samples/configs/faster_rcnn_inception_resnet_v2_atrous_oid_v4.config -n 15

    if selected_model == 'faster_rcnn_inception_v2_coco':
        !wget https://bitbucket.org/!api/2.0/snippets/burnpnk/pnB4Mp/5d0b913ef706d5c98b7e6cbb4090375a22151504/files/faster_rcnn_inception_v2_coco.config \
        -O /content/models/research/object_detection/samples/configs/faster_rcnn_inception_v2_coco.config
        !head /content/models/research/object_detection/samples/configs/faster_rcnn_inception_v2_coco.config -n 22

selected_model


test_record_fname = './dataset/test.record'
train_record_fname = './dataset/train.record'
label_map_pbtxt_fname = './dataset/label_map.pbtxt'

"""## Download base model"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/models/research

import os
import shutil
import glob
import urllib.request
import tarfile

MODEL_FILE = MODEL + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
DEST_DIR = '/content/models/research/pretrained_model'


fine_tune_checkpoint = os.path.join(DEST_DIR, "model.ckpt")
fine_tune_checkpoint

"""## Configuring a Training Pipeline"""

import os
pipeline_fname = os.path.join('/content/models/research/object_detection/samples/configs/', pipeline_file)

assert os.path.isfile(pipeline_fname), '`{}` not exist'.format(pipeline_fname)
pipeline_fname

def get_num_classes(pbtxt_fname):
    from object_detection.utils import label_map_util
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return len(category_index.keys())

import re

num_classes = get_num_classes(label_map_pbtxt_fname)
with open(pipeline_fname) as f:
    s = f.read()
with open(pipeline_fname, 'w') as f:
    
    # fine_tune_checkpoint
    s = re.sub('fine_tune_checkpoint: ".*?"',
               'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), s)
    
    # tfrecord files train and test.
    s = re.sub(
        '(input_path: ".*?)(train.record)(.*?")', 'input_path: "{}"'.format(train_record_fname), s)
    s = re.sub(
        '(input_path: ".*?)(val.record)(.*?")', 'input_path: "{}"'.format(test_record_fname), s)

    # label_map_path
    s = re.sub(
        'label_map_path: ".*?"', 'label_map_path: "{}"'.format(label_map_pbtxt_fname), s)

    # Set training batch_size.
    s = re.sub('batch_size: [0-9]+',
               'batch_size: {}'.format(batch_size), s)

    # Set training steps, num_steps
    s = re.sub('num_steps: [0-9]+',
               'num_steps: {}'.format(num_steps), s)
    
    # Set number of classes num_classes.
    s = re.sub('num_classes: [0-9]+',
               'num_classes: {}'.format(num_classes), s)
    f.write(s)

#!cat {pipeline_fname}

model_dir = 'training/'
# Optionally remove content in output model directory to fresh start.
!rm -rf {model_dir}
os.makedirs(model_dir, exist_ok=True)


"""## Train the model"""
#%%capture
!python /content/models/research/object_detection/model_main.py \
    --pipeline_config_path={pipeline_fname} \
    --model_dir={model_dir} \
    --alsologtostderr \
    --num_train_steps={num_steps} \
    --num_eval_steps={num_eval_steps}

# memory footprint support libraries/code


"""## Exporting a Trained Inference Graph
Once your training job is complete, you need to extract the newly trained 
inference graph, which will be later used to perform the object detection.
This can be done as follows:

### export to frozen inference graph
"""

# AssertionError: Export directory already exists, and isn't empty. Please choose a different export directory, or delete all the contents of the specified directory: ./fine_tuned_model/saved_model
#!rm ./fine_tuned_model/saved_model/* -r

import re
import numpy as np

output_directory = './fine_tuned_model'

lst = os.listdir(model_dir)
lst = [l for l in lst if 'model.ckpt-' in l and '.meta' in l]
steps=np.array([int(re.findall('\d+', l)[0]) for l in lst])
last_model = lst[steps.argmax()].replace('.meta', '')

last_model_path = os.path.join(model_dir, last_model)

print(last_model_path)
!python /content/models/research/object_detection/export_inference_graph.py \
    --input_type=image_tensor \
    --pipeline_config_path={pipeline_fname} \
    --output_directory={output_directory} \
    --trained_checkpoint_prefix={last_model_path}

!pwd
#!ls /content/models/research/training  # model_dir
#!ls /content/models/research/pretrained_model  # pretrained model duh
#!ls /content/models/research/fine_tuned_model  # gen directory from exporting a 'trained inference graph'
!ls -rlh /content/models/research/fine_tuned_model/saved_model/

f = '/content/models/research/fine_tuned_model/saved_model/saved_model.pb'
uploaded = drive.CreateFile({'title': os.path.basename(f)})
uploaded.SetContentFile(f)
uploaded.Upload()
print('Uploaded file\n- {}\n- with ID {}\n'.format(f, uploaded.get('id')))

"""### export to a tflite-conversion-compatible frozen graph (for mobile devices)"""

# to export to a TFLITE conversion compatible file you have to use a different script
# note that this only works with ssd models
!python /content/models/research/object_detection/export_tflite_ssd_graph.py \
    --pipeline_config_path={pipeline_fname} \
    --output_directory={output_directory} \
    --trained_checkpoint_prefix={last_model_path} \
    --add-post-processing-op=true

"""### convert tflite compatible frozen graph to `*.tflite` file"""

!if [ -d /content/tensorflow/bazel-bin/tensorflow/lite/toco/ ]; then echo "toco is installed :)"; else echo "toco is not installed :("; fi

#INPUT_FILE=../../../releases/v3.0/tflite_graph.pb
#OUTPUT_FILE=../../../releases/v3.0/detect.tflite

#INPUT_ARRAYS=normalized_input_image_tensor
#OUTPUT_ARRAYS='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1',\
#'TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'

"""### see the output files"""

!ls ./fine_tuned_model/

"""## Download the model output files

All output files (+ input config file) are downloaded one-by-one. I tried to compress several times using `*.tar.xz`, `*.tar.gz` and `*.zip`. These compressed archives were somehow adulterated either from upload to Google Drive or direct download. As such, it is opted to upload 1-by-1 to Drive, as it is the simpler, painless-in-the-ass and quickest option available.
"""
# Commented out IPython magic to ensure Python compatibility.
# collect file paths

# %cd /content/models/research
# make list of files to download
output_dir = os.path.abspath('./fine_tuned_model')
dn_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir)]
dn_files = dn_files + [pipeline_fname] + [label_map_pbtxt_fname]
#dn_files

# uses PyDrive
# Create & upload
for f in dn_files:
    if os.path.isdir(f):
        continue
    uploaded = drive.CreateFile({'title': os.path.basename(f)})
    uploaded.SetContentFile(f)
    uploaded.Upload()
    print('Uploaded file\n- {}\n- with ID {}\n'.format(f, uploaded.get('id')))
