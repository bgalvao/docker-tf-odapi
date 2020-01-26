from os.path import join, exists
from os import makedirs
from sys import exit
import shutil
from subprocess import run
import argparse
from time import sleep

# utility to load model, either a fresh one or pre-trained
from load import load_model_paths
# # utility to set training pipeline (hyperparams and more)
from pipeline import set_training_config
from export import find_model_last_ckpt_prefix, export


def train(training_config_file, training_dir, num_steps, num_eval_steps):
    run([
        'python',
        '../models/research/object_detection/model_main.py',
        '--pipeline_config_path={}'.format(training_config_file),
        '--model_dir={}'.format(training_dir),
        '--alsologtostderr',
        '--num_train_steps={}'.format(num_steps),
        '--num_eval_steps={}'.format(num_eval_steps)
    ],
        # stderr=open(join(training_dir, 'training.log'), 'w')
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model2resume',
        help='Model that is already downloaded which training you want to resume.\
            If --ckpt2resume is not passed, training will resume from last checkpoint.',
        type=str
    )
    parser.add_argument(
        '--model2download',
        help='No model to resume from? Pick a model specified in \
            ./model/tf_org_models.py to download from\
            tensorflow.org. This arg is conflictive with --model2resume.',
        type=str
    )
    parser.add_argument(
        '--alt_model_name',
        help='If downloading a model --model2download, you can pick an alternative\
            name',
        type=str
    )
    parser.add_argument(
        '--num_steps',
        help='Number of training epochs to train a model for.',
        type=int,
        default=200000
    )
    parser.add_argument(
        '--num_eval_steps',
        help='Number of times to run evaluation on test set during training.',
        type=int,
        default=50
    )
    parser.add_argument(
        '--batch_size',
        help='Number of samples to feed in a step of a training epoch.\
            Larger is better, but constrained to GPU memory.',
        type=int,
        default=12
    )
    parser.add_argument(
        '--ckpt2resume',
        help='Checkpoint PREFIX to resume training from. If non-existent, crash.',
        type=str
    )
    parser.add_argument(
        '--training_config',
        help='Specify a pipeline config file to use for training. This is to\
        adjust other hyperparams not present on this CLI, such as the\
        learning rate.',
        type=str
    )
    args = parser.parse_args()


    # defaults :: current input dataset converted to tfrecords
    test_record_fname = './input/tf_records/test.record'
    train_record_fname = './input/tf_records/train.record'
    label_map_pbtxt_fname = './input/tf_csv/label_map.pbtxt'

    training_dir, export_dir, base_pipeline_config_path = load_model_paths(
        model2resume=args.model2resume,
        model2download=args.model2download,
        alt_model_name=args.alt_model_name
    )

    print(args)

    if args.training_config is None:
        print('Configuring a new training.config from scratch...')
        training_config_file = set_training_config(
            base_pipeline_config=base_pipeline_config_path,
            model_name='ssd_mobilenet_v2',
            train_record=train_record_fname, 
            test_record=test_record_fname,
            labelmap_pbtxt_path=label_map_pbtxt_fname,  # later copied to export dir
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            checkpoint=args.ckpt2resume
        )
    else:
        training_config_file = args.training_config
        if not exists(training_config_file):
            raise FileNotFoundError(training_config_file)

    print("TRAINING DIR :: {}\nEXPORT DIR :: {}\n ".format(
        training_dir, export_dir
    ))
    print("TRAINING CONFIG :: {}\n".format(training_config_file))
    print("make sure these values match the values in TRAINING_CONFIG\
        \nnum_steps, num_eval_steps, batch_size = {}, {}, {}\n\n".format(
        args.num_steps, args.num_eval_steps, args.batch_size
    ))

    print('training will now commence')
    print('I\'ll see you on the other side\n')
    print("̿̿'̿'\̵͇̿̿\з= ( ▀ ͜͞ʖ▀) =ε/̵͇̿̿/’̿’̿\n\n")
    
    
    train(
        training_config_file, training_dir, args.num_steps, args.num_eval_steps
    )


