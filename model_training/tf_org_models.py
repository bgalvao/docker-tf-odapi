# add models as you need to experiment...
# fetch the tf.org.model_name the archive link from the tensorflow model zoo.
# pipeline file is fetched from the models repo in the docker container.
# the path to configs in this repo is 
# models/research/object_detection/samples/configs
# prefixed with ../ in the docker image

from os.path import join, exists


def get_pipeline_path(ppname):
    full_path = join(
        '../models/research/object_detection/samples/configs',
        ppname
    )
    assert exists(full_path), "Config file full path not found :: {}".format(
        full_path
    )
    return full_path


downloadable_models = {
    # single shot detectors
    'ssd_mobilenet_v2': {
        # tf.org.model_name is the model name expected to be found on
        # http://download.tensorflow.org/models/object_detection/
        'tf.org.model_name': 'ssd_mobilenet_v2_coco_2018_03_29',
        'pipeline_file': get_pipeline_path('ssd_mobilenet_v2_coco.config')
    },
    'ssd_mobilenet_v2_quantized_coco': {
        'tf.org.model_name': 'ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03',
        'pipeline_file': get_pipeline_path('ssd_mobilenet_v2_quantized_300x300_coco.config')
    },
    # faster rccn's
    'faster_rcnn_inception_v2_coco': {
        'tf.org.model_name': 'faster_rcnn_inception_v2_coco_2018_01_28',
        'pipeline_file': get_pipeline_path('faster_rcnn_inception_v2_coco.config')
    }
}
