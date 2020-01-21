FROM tensorflow/tensorflow:1.15.0-gpu-py3

# make a "user" directory, later stored to HOME
RUN mkdir -p /home/oceanus
ENV HOME /home/oceanus
WORKDIR /home/oceanus
RUN apt update

# pip dependencies of the object detection api
RUN pip install Cython contextlib2 pillow lxml jupyterlab matplotlib pandas

# install git
RUN apt install git -y

# clone tensorflow models repo
RUN git clone https://github.com/tensorflow/models.git --depth 1 -v
WORKDIR models
# reset to specific version...
RUN git reset 7124ed1227aa3e7a3c38cd412e63994b942f3c63 --hard
WORKDIR $HOME

# coco api clone
RUN git clone https://github.com/cocodataset/cocoapi.git --depth 1 -v
# build cocoapi tools
WORKDIR cocoapi/PythonAPI
RUN make

# install pycocotools
RUN cp -r pycocotools $HOME/models/research/

# install protobuf-compiler
# ** from tensorflow/models/research/ **
WORKDIR $HOME/models/research
RUN apt install unzip wget -y
RUN wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
RUN unzip protobuf.zip

# compile protos
# ** from tensorflow/models/research/ **
RUN ./bin/protoc object_detection/protos/*.proto --python_out=.

# add object detection API libraries to python path
# ** from tensorflow/models/research/ **
ENV PYTHONPATH $HOME/models/research:$HOME/models/research/slim

# test installation
RUN echo 'testing installation of Object Detection API' && \
    python $HOME/models/research/object_detection/builders/model_builder_test.py

# return home
USER 1000:1000
WORKDIR $HOME