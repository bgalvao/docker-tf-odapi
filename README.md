# Dockerized TensorFlow Object Detection API

Drivers and Cuda versions are a pain to deal with.

- [Dockerized TensorFlow Object Detection API](#dockerized-tensorflow-object-detection-api)
  - [Set up](#set-up)
    - [test host machine](#test-host-machine)
    - [prerequisites](#prerequisites)
    - [steps to install an image](#steps-to-install-an-image)
    - [running a container](#running-a-container)
  - [process](#process)
    - [part 1 :: dataset conversions](#part-1--dataset-conversions)
    - [part 2 :: uuhh](#part-2--uuhh)
  

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
docker build -t odapi -f ./dockerfiles/odapi  # build the object detection API image
docker build -t tfjs -f ./dockerfiles/tfjs  # build the image for tensorflow js conversions
```

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

This section describes the scripts you have to run in order to get a model.

### part 1 :: dataset conversions

It goes from a supervisely-formatted dataset (placed in `./input/supervisely`) » `./input/tf_csv` » `./input/tf_records`.

For these steps, you only need to run a python installation with `tensorflow`.

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

That should set you up with the dataset.


### part 2 :: uuhh
