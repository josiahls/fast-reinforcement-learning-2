FROM pytorch/pytorch

ARG BUILD=dev

ENV CONTAINER_USER fastrl
ENV CONTAINER_GROUP fastrl_group
ENV CONTAINER_UID 1000
# Add user to conda
RUN addgroup --gid $CONTAINER_UID $CONTAINER_GROUP && \
    adduser --uid $CONTAINER_UID --gid $CONTAINER_UID $CONTAINER_USER --disabled-password  && \
    mkdir -p /opt/conda && chown $CONTAINER_USER /opt/conda

RUN chown $CONTAINER_USER:$CONTAINER_GROUP -R /opt/conda/lib/python3.7/site-packages
RUN chown $CONTAINER_USER:$CONTAINER_GROUP -R /home/fastrl


RUN apt-get update && apt-get install -y software-properties-common rsync
RUN add-apt-repository -y ppa:git-core/ppa && apt-get update && apt-get install -y git libglib2.0-dev graphviz && apt-get update
RUN pip install albumentations \
    catalyst \
    captum \
    "fastprogress>=0.1.22" \
    graphviz \
    jupyter \
    kornia \
    matplotlib \
    "nbconvert<6"\
    nbdev \
    neptune-client \
    opencv-python \
    pandas \
    pillow \
    pyarrow \
    pydicom \
    pyyaml \
    scikit-learn \
    scikit-image \
    scipy \
    "sentencepiece<0.1.90" \
    spacy \
    tensorboard \
    wandb \
    jupyterlab \
    watchdog[watchmedo]

USER $CONTAINER_USER
WORKDIR /home/$CONTAINER_USER
ENV PATH="/home/$CONTAINER_USER/.local/bin:${PATH}"

RUN git clone https://github.com/fastai/fastai.git --depth 1 \
        && git clone https://github.com/fastai/fastcore.git --depth 1 \
        && git clone https://github.com/josiahls/fastrl.git --depth 1
RUN /bin/bash -c "if [[ $BUILD == 'prod' ]] ; then echo \"Production Build\" && cd fastai && pip install . && cd ../fastcore && pip install . && cd ../fastrl && pip install .; fi"
# Note that we are not installing the .dev dependencies
RUN /bin/bash -c "if [[ $BUILD == 'dev' ]] ; then echo \"Development Build\" && cd fastai && pip install -e . && cd ../fastcore && pip install -e . cd ../fastrl && pip install -e \".[dev]\"; fi"

RUN /bin/bash -c "pip install jupyterlab"
RUN echo '#!/bin/bash\njupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser' >> run_jupyter.sh

USER $CONTAINER_USER
RUN /bin/bash -c "cd fastrl && pip install -e ."

RUN chmod u+x run_jupyter.sh