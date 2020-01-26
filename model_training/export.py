"""## Exporting a Trained Inference Graph
Once your training job is complete, you need to extract the newly trained 
inference graph, which will be later used to perform the object detection.
This can be done as follows:

### export to frozen inference graph
"""

import re
import numpy as np
from subprocess import run
from os.path import join, exists
from shutil import rmtree
import os
import argparse

def find_model_last_ckpt_prefix(model_name):
    """
    Returns model.ckpt prefix of last found checkpoint
    """
    training_dir = join('output', model_name, 'training')
    lst = os.listdir(training_dir)
    lst = [l for l in lst if 'model.ckpt-' in l and '.meta' in l]
    if len(lst) == 0:
        return 'model.ckpt' # default hacky hack
    steps=np.array([int(re.findall('\d+', l)[0]) for l in lst])
    last_model_prefix = lst[steps.argmax()].replace('.meta', '')
    return last_model_prefix


# recommended reading
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/exporting_models.md
# this export should work for any model architecture
def export_frozen_inference_graph(pipeline_fname, output_dir, ckpt_path):
    
    if exists(output_dir):
        print('deleting {} for re-export to be executed'.format(output_dir))
        rmtree(output_dir)

    run([
        'python',
        '../models/research/object_detection/export_inference_graph.py',
        '--input_type=image_tensor',
        '--pipeline_config_path={}'.format(pipeline_fname),
        '--output_directory={}'.format(output_dir),
        '--trained_checkpoint_prefix={}'.format(ckpt_path)
    ])


# to export to a TFLITE conversion compatible file you have to use a different script
# note that this only works with ssd models
def export_tflite_ssd_graph(pipeline_fname, output_dir, ckpt_path):
    
    if exists(output_dir):
        print('deleting {} for re-export to be executed'.format(output_dir))
        rmtree(output_dir)

    run([
        'python',
        '../models/research/object_detection/export_tflite_ssd_graph.py',
        '--pipeline_config_path={}'.format(pipeline_fname),
        '--output_directory={}'.format(output_dir),
        '--trained_checkpoint_prefix={}'.format(ckpt_path),
        '--add-post-processing-op=true'
    ])



def export(model_name, ckpt_prefix=None):
    
    training_dir = join('output', model_name, 'training')
    # training_config = join(training_dir, 'pipeline.config')  # dev purposes
    training_config = join(training_dir, 'training.config')
    output_dir = join('output', model_name, 'export')

    if ckpt_prefix is None:
        ckpt_prefix = find_model_last_ckpt_prefix(model_name)
    ckpt_path = join(training_dir, ckpt_prefix)

    export_frozen_inference_graph(
        training_config,
        join(output_dir, 'frozen_graph'),
        ckpt_path
    )
    
    export_tflite_ssd_graph(
        training_config,
        join(output_dir, 'tflite_graph'),
        ckpt_path
    )





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_name',
        help='Model name in ./output.'
    )
    args = parser.parse_args()


    """
    tensorflow.python.framework.errors_impl.InvalidArgumentError: 
    Restoring from checkpoint failed. This is most likely due to a mismatch 
    between the current graph and the graph from the checkpoint. 
    Please ensure that you have not altered the graph expected based on the 
    checkpoint.
    """
    # i.e. training config with 5 classes is going to mismatch with
    # some model.ckpt with 90 classes
    export(args.model_name)
