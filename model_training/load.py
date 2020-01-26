import argparse
import tarfile
from urllib.request import urlretrieve
import shutil
from os import system, makedirs, listdir
from os.path import join, exists, isdir

from tf_org_models import downloadable_models


def download_model(model_arch, alt_model_name=None):
    """
    Use this if a model does not already exist on disk
    or you're trying a new one.
    This function downloads a pre-trained model from
    http://download.tensorflow.org/models/object_detection/
    directory of models that is listed in the model zoo
    https://github.com/tensorflow/models/blob/maste

    - tf_org_model_name : a model name to download, specced in tf_org_models.py
    - output_model_name : name of model to store on local disk.
                          if None, just use tf_org_model_name
    """

    download_base_url = 'http://download.tensorflow.org/models/object_detection/'
    
    tf_org_model_name = downloadable_models[model_arch]['tf.org.model_name']
    base_pipeline_config = downloadable_models[model_arch]['pipeline_file']

    model_archive_basename = tf_org_model_name + '.tar.gz'
    dst_dir = './downloaded_pretrained_models'
    model_archive_path = join(dst_dir, model_archive_basename)

    if not isdir(dst_dir):
        makedirs(dst_dir)

    # "cache"
    if not exists(model_archive_path):
        print("downloading ::", download_base_url + model_archive_basename)
        urlretrieve(
            url=download_base_url + model_archive_basename,
            filename=model_archive_path
        )
    else:
        print("model archive was already downloaded yay :D")

    # saves to ./output/$model_name/training
    dst = join('./output', alt_model_name) \
        if alt_model_name is not None else join(
            './output', model_arch
        )
    
    print("extracting download.tensorflow.org model archive...")
    tar = tarfile.open(model_archive_path)
    tar.extractall(path=dst)
    tar.close()

    if exists(join(dst, 'training')):
        raise FileExistsError(
            "\n\nModel directory already exists. \
            \n- Overwriting models that were potentially trained is forbidden.\
            \n- If you want to train a new model from scratch, \
            \n     pass a non-existent alt_model_name.\
            \n\n>> These are the model names already in use:\
            \n {}".format("\n".join(listdir('./output')))
        )

    # not elegant, but whatever works man
    shutil.move(
        src=join(dst, tf_org_model_name),
        dst=join(dst, 'training')
    )
    # store model_arch for future ref
    system("echo {} > {}".format(model_arch, join(
            dst,
            'training/model.arch')
        )
    )
    # make sure its integrity checks out
    assert_ckpt(join(dst, 'training'))
    
    # make export directory for inference executables
    model_name = model_arch if alt_model_name is None else alt_model_name
    export_dir = join('./output', model_name, 'export')
    if not exists(export_dir):
        makedirs(export_dir)

    # rename `checkpoint` file so that training can actually take off...
    # don't ask, but... 
    # https://github.com/tensorflow/models/issues/5053#issuecomment-441423962
    shutil.move(join(dst, 'training/checkpoint'), join(dst, 'training/old_checkpoint'))
    
    # path to training dir and path to base training pipeline config
    return join(dst, 'training'), export_dir, base_pipeline_config


def load_model(model2resume):
    """
    Use this if you want to load a (semi-)trained model stored on disk.
    model_name : model_name under ./output
    """
    # verify it is exists
    model2resume_path = join('output', model2resume)
    if not exists(model2resume_path):
        raise FileNotFoundError('model {} does not exist'.format(model2resume))
    training_dir = join(model2resume_path, 'training')
    assert_ckpt(training_dir)

    # go remember the model architecture of this model
    # to fetch back the pipeline file
    with open(join(training_dir, 'model.arch'), 'r') as f:
        s = f.read().strip("\n")
        base_pipeline_config_path = downloadable_models[s]['pipeline_file']
    
    # fetch export dir
    export_dir = join('output', model2resume, 'export')
    assert isdir(export_dir)

    return training_dir, export_dir, base_pipeline_config_path


def assert_ckpt(model_traning_dir):
    """
    Confirms whether model.ckpt* files exist in specced model_dirpath.
    If not, everything crashes (not possible to train).
    If they do exist, it returns a pipeline.config friendly path
    to model.ckpt* files.
    """
    index = join(model_traning_dir, 'model.ckpt.index')
    meta = join(model_traning_dir,'model.ckpt.meta')
    assert exists(index) and exists(meta), \
        "crucial model.ckpt* files not found in {}".format(model_traning_dir)


def load_model_paths(model2resume=None, model2download=None,
        alt_model_name=None):

    """
    # Pick the model you want to download to train from scratch
    model2download = None  # check tf_org_models.py for keys
    alt_model_name = None  # alternative model name to save in ./output

    # pick the downloaded model (and probably trained)
    # that you want to resume training
    model2resume = 'ssd_mobilenet_v2'
    """

    if model2download is not None and model2resume is not None:
        raise Exception(
            "\n\nYou can either download a new model and train from scratch:\
            \ni.e.: spec model2download\
            \n\nOr resume training on an existing model\
            \ni.e.: spec model2resume\
            \n\nbut not both."
        )
    elif model2download is None and model2resume is None:
        raise Exception(
            "You have to specify either model2resume or model2download"
        )

    if model2download is not None:
        training_dir, export_dir, base_pipeline_config_path = download_model(
            model_arch=model2download,
            alt_model_name=alt_model_name
        )
    else:  # model 2 resume is specified
        training_dir, export_dir, base_pipeline_config_path = load_model(
            model2resume
        )
    return training_dir, export_dir, base_pipeline_config_path
