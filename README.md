# Dockerized [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)

Drivers and Cuda versions are a pain to deal with. Praise our lord and savior Docker.

- [Dockerized TensorFlow Object Detection API](#dockerized-tensorflow-object-detection-api)
    - [Set up](#set-up)
        - [test host machine](#test-host-machine)
        - [prerequisites](#prerequisites)
        - [steps to install an image](#steps-to-install-an-image)
        - [running a container](#running-a-container)
    - [process](#process)
        - [part 1 :: dataset conversions](#part-1--dataset-conversions)
        - [part 2 :: training the model](#part-2--training-the-model)
        - [part 3 :: converting between model formats](#part-3--converting-between-model-formats)
            - [converting to a .tflite format](#converting-to-a-tflite-format)
            - [converting to a TensorFlowJS format](#converting-to-a-tensorflowjs-format)
    - [other notes](#other-notes)
        - [running on a Coral TPU](#running-on-a-coral-tpu)
  

## Set up

### test host machine

This was tested with a host machine with the following characteristics.

```console
user@machine:~/work/docker/tf_object_detection_api$ nvidia-smi
Wed Jan  8 13:04:47 2020       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 440.44       Driver Version: 440.44       CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce MX250       Off  | 00000000:01:00.0 Off |                  N/A |
| N/A   44C    P0    N/A /  N/A |    190MiB /  2002MiB |      2%      Default |
+-------------------------------+----------------------+----------------------+
```

However you should not need CUDA, only a driver for your GPU.


### prerequisites

These are the prerequisites are only for the TensorFlow Object Detection API.

- an nvidia driver
- `docker`
- `nvidia-docker`

### steps to install an image

```bash
# build the object detection API image
docker build -t odapi -f ./dockerfiles/odapi.dockerfile ./dockerfiles

# build the image for tensorflow js conversions
docker build -t tfjs -f ./dockerfiles/tfjs.dockerfile ./dockerfiles

# build the image for tensorflow lite conversions
docker build -t toco -f ./dockerfiles/toco.dockerfile ./dockerfiles
```

You can check that you built the 3 images by running `docker images`. Here is a summary though:
- **`odapi`**, runs the TensorFlow **O**bject **D**etection **API**, which at the date of writing, only runs on TensorFlow 1.x, so it pulls from the `tensorflow/tensorflow:1.15.0-gpu-py3` image as its base.
- **`tfjs`**, encapsulates the `tensorflowjs` suite and its function is to provide a TensorFlow JS converter. Since it does not require GPU, and its latest version requires TensorFlow 2.x, it pulls from `tensorflow/tensorflow:2.1.0-py3` as its base image.
- finally, **`toco`** comes with TOCO (i.e. [TensorFlow Lite Converter](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/toco)) built with [bazel](bazel.build). It pulls from the same base image as `tfjs`, which is cached.

### running a container

A docker-compose.yml is in the works to handle port, volumes, launch programs etc.
In the meantime, here are a few options.

TensorFlow Object Detection API with GPU support

```shell
docker run --rm -it --gpus all -v $(pwd):/home/oceanus odapi
```

to run this container without a gpu, remove the gpu flag:

```shell
docker run --rm -it -v $(pwd):/home/oceanus odapi
```

to start a container for TensorFlow JS conversions.

```shell
docker run --rm -it -v $(pwd):/home/oceanus tfjs
```


## process

This section describes the scripts you have to run in order to get a model, in relevant order.

### part 1 :: dataset conversions

It goes from a supervisely-formatted dataset (placed in `./input/supervisely`) » to `./input/tf_csv` » lastly, to `./input/tf_records`.

For these steps, you only need to run a python installation with `tensorflow`, or use the odapi container in tty mode.

```shell
python ./input/supervisely2tf_csv.py  # will use defaults
```

After you have a tf_csv formatted dataset, run the following to generate TFRecords.

```shell
# Generate `train.record`
python input/generate_tfrecord.py \
--csv_input=input/tf_csv/train.csv \
--output_path=input/tf_records/train.record \
--img_path=input/tf_csv/images/train \
--label_map=input/tf_csv/label_map.pbtxt

# Generate `test.record`
python ./input/generate_tfrecord.py \
--csv_input=./input/tf_csv/test.csv \
--output_path=./input/tf_records/test.record \
--img_path=./input/tf_csv/images/test \
--label_map=./input/tf_csv/label_map.pbtxt
```

This should have you set up with the dataset.


### part 2 :: training the model

### part 3 :: converting between model formats

The resulting model of the previous part comes in three formats.

- [FrozenGraph](), deprecated in TensorFlow 2.
- [SavedModel](https://www.tensorflow.org/guide/saved_model), which will be used to convert to TensorFlowJS
- tflite-compatible inference graph
    - saved in files `tflite_graph.pb` and `tflite_graph.pbtxt`


#### converting to a .tflite format

This is takes from the [official guide to getting a tflite model](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md) that you may want to read.

Boot the purpose-specific container:

```shell
docker run -it --rm -v $(pwd):/home/oceanus toco
# this is the only container that will run as root btw
```

Edit the [tflite_graph2tflite.sh](./container_scripts/tflite_graph2tflite.sh) script so that `MODEL_NAME` matches that of your target output model. After that is ensured, you just have to run
the script from the container.

```shell
./container_scripts/tflite_graph2tflite.sh
```

The files for TensorFlow Lite will be placed in the `./output/$MODEL_NAME/tflite` directory.


#### converting to a TensorFlowJS format

Boot the corresponding container

```shell
docker run -it --rm -v $(pwd):/home/oceanus tfjs 
```

Edit the [saved_model2tfjs.sh](./container_scripts/saved_model2tfjs.sh) script so that `MODEL_NAME` matches that of your target output model. After that is ensured, just execute the script from the container.

```shell
./container_scripts/saved_model2tfjs.sh
```

The files for TensorFlow JS will be placed in the `./output/$MODEL_NAME/tensorflow_js` directory.


## other notes

### running on a Coral TPU
If you want to run a model on a Coral TPU, one way to train a model
that supports TPU training - any marked with an [asterisk (☆) in the Model Zoo list](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models). (not tested)

Another option is to train an `ssd_mobilenet_v2_`**`quantized`**`_coco` in order to easily convert to an uint8-tflite-formatted model - as per section [tflite conversion section](#converting-to-a-tflite-format). (tested)


