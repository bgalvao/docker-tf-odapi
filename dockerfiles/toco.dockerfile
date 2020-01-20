FROM tensorflow/tensorflow:2.1.0-py3

# make a "user" directory
RUN mkdir -p /home/oceanus
#ENV HOME /home/oceanus
WORKDIR /home/oceanus


RUN apt update


# install bazel
RUN apt install curl -y
RUN curl https://bazel.build/bazel-release.pub.gpg | apt-key add -
RUN echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" \
    | tee /etc/apt/sources.list.d/bazel.list
RUN apt update && apt install bazel -y

# clone tensorflow repo
RUN apt install git -y
RUN git clone --depth 1 https://github.com/tensorflow/tensorflow.git

# install toco
WORKDIR /home/oceanus/tensorflow
RUN touch BUILD
RUN bazel build --config=opt 
# RUN bazel build 
# RUN



# return home
# USER 1000:1000
# WORKDIR $HOME
