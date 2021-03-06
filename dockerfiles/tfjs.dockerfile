FROM tensorflow/tensorflow:2.1.0-py3

# make a "user" directory, later stored to HOME
RUN mkdir -p /home/oceanus
ENV HOME /home/oceanus
WORKDIR /home/oceanus
RUN apt update

# install git
RUN apt install git -y

# install tensorflowjs
# at the time of writing this Dockerfile
# pip index is not updated with tensorflowjs 1.5.1
# RUN pip install tensorflowjs
# so I am building it manually
WORKDIR $HOME
RUN git clone --depth 1 https://github.com/tensorflow/tfjs.git
WORKDIR $HOME/tfjs/tfjs-converter/python/
# who still uses python 2 in this day and age omg
RUN sed s/"python2 "/""/ build-pip-package.sh > build-pip-package-3.sh
RUN chmod +x build-pip-package-3.sh
RUN ./build-pip-package-3.sh ./wheels --test
RUN python -m pip install ./wheels/tensorflowjs-1.5.1-py3-none-any.whl

# return home
USER 1000:1000
WORKDIR $HOME
