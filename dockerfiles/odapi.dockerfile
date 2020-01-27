FROM tensorflow/tensorflow:1.15.0-gpu-py3

# pip dependencies of the object detection api
RUN pip install Cython contextlib2 pillow lxml jupyterlab matplotlib pandas \
    tqdm bullet

# install git
RUN apt update
RUN apt install git -y

# clone tensorflow models repo
WORKDIR /home
RUN git clone https://github.com/tensorflow/models.git -v
WORKDIR /home/models
# if this build ever fails in the future... reset to specific version if you're lazy
# RUN git reset 7124ed1227aa3e7a3c38cd412e63994b942f3c63 --hard

# coco api clone
WORKDIR /home
RUN git clone https://github.com/cocodataset/cocoapi.git --depth 1 -v
# build cocoapi tools
WORKDIR /home/cocoapi/PythonAPI
RUN make

# install pycocotools
RUN cp -r pycocotools /home/models/research/

# install protobuf-compiler
# ** from tensorflow/models/research/ **
WORKDIR /home/models/research
RUN apt install unzip wget -y
RUN wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
RUN unzip protobuf.zip

# compile protos
# ** from tensorflow/models/research/ **
RUN ./bin/protoc object_detection/protos/*.proto --python_out=.

# add object detection API libraries to python path
# ** from tensorflow/models/research/ **
ENV PYTHONPATH /home/models/research:/home/models/research/slim

# test installation
RUN echo 'testing installation of Object Detection API' && \
    python /home/models/research/object_detection/builders/model_builder_test.py

# return home
RUN mkdir -p /home/oceanus
ENV HOME /home/oceanus
USER 1000:1000
WORKDIR $HOME
