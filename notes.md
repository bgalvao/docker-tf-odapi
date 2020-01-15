# Notes

toc

## test host machine

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



## prerequisites

- nvidia-driver
- docker
- nvidia-docker

## steps to install the image

```bash
docker build -t image_name -f path/to/Dockerfile
```

## running the image

Should you want to mount a volume, look it up yourself.. A docker-compose.yml is in the works to handle port, volumes, launch programs etc.

with gpu:

```bash
docker run --rm -it --gpus all image_name
```

without a gpu, remove the gpu flag:

```bash
docker run --rm -it image_name
```

