"""
Configuring a Training Pipeline

This script provides utilities to:
- load a default pipeline.config file and overwrite key values with
  user input-ted hyperparam values
- load an already written pipeline.config.
"""

import re
import os
from os.path import join, exists
from export import find_model_last_ckpt_prefix


def get_num_classes(pbtxt_fname):
    from object_detection.utils import label_map_util
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return len(category_index.keys())


def set_training_config(
    base_pipeline_config, model_name, train_record, test_record,
    labelmap_pbtxt_path, batch_size, num_steps, num_eval_steps, checkpoint=None
    ):

    """
    writes a pipeline config that is going to be used for training, and returns
    the path to it. (The baseline pipeline.config remains thus intact).

    - base_pipeline_config : str
        path to pipeline.config from pretrained model downloaded from 
        download.tensorflow.org
    - model_name : str
        the name to the model you want to give in order to save to 
        ./output/model_name/training/training.config.
    - checkpoint : str
        If you want to resume training, then pass a different checkpoint template.
        E.g. 'model.ckpt-400'. This function will always assert that
        model.ckpt*{.index, .meta} exist. Default is to resume from the last
        checkpoint..
    """
    # creates an edited copy of the base_pipeline_config
    with open(base_pipeline_config) as f:
        s = f.read()

        # checkpoint
        if checkpoint is None:
            # find last of it
            last_model = find_model_last_ckpt_prefix(model_name)
            ftc = join('output', model_name, 'training', last_model)
        else:
            ftc = join(
                'output', model_name, 'training', checkpoint
            )
        assert exists(ftc+'.index') and exists(ftc+'.meta')
        s = re.sub('fine_tune_checkpoint: ".*?"',
                'fine_tune_checkpoint: "{}"'.format(ftc), s
        )

        # tfrecord files train and test.
        s = re.sub(
            '(input_path: ".*?)(train.record)(.*?")', 'input_path: "{}"'.format(
                train_record), s
            )
        s = re.sub(
            '(input_path: ".*?)(val.record)(.*?")', 'input_path: "{}"'.format(
                test_record), s
            )

        # label_map_path
        s = re.sub(
            'label_map_path: ".*?"', 'label_map_path: "{}"'.format(
                labelmap_pbtxt_path), s
            )

        # Set training batch_size.
        s = re.sub('batch_size: [0-9]+',
                'batch_size: {}'.format(batch_size), s)

        # Set training steps, num_steps
        s = re.sub('num_steps: [0-9]+',
                'num_steps: {}'.format(num_steps), s)

        # Set number of classes num_classes.
        s = re.sub('num_classes: [0-9]+',
                'num_classes: {}'.format(
                    get_num_classes(labelmap_pbtxt_path)),
                    s
                )
        
        s = re.sub(
            'max_evals: [0-9]+', 'max_evals: {}'.format(num_eval_steps),
            s
        )
    
    # writes the edit to a new file
    out_dir = join('./output', model_name, 'training')
    pipeline_fname = join(out_dir, 'training.config')

    with open(pipeline_fname, 'w') as f:
        f.write(s)
    return pipeline_fname
