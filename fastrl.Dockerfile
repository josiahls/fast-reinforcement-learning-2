FROM pytorch/pytorch

ARG BUILD=dev

ENV CONTAINER_USER fastrl_user
ENV CONTAINER_GROUP fastrl_group
ENV CONTAINER_UID 1000
# Add user to conda
RUN addgroup --gid $CONTAINER_UID $CONTAINER_GROUP && \
    adduser --uid $CONTAINER_UID --gid $CONTAINER_UID $CONTAINER_USER --disabled-password  && \
    mkdir -p /opt/conda && chown $CONTAINER_USER /opt/conda

RUN apt-get update && apt-get install -y software-properties-common rsync
RUN add-apt-repository -y ppa:git-core/ppa && apt-get update && apt-get install -y git libglib2.0-dev graphviz libxext6 libsm6 libxrender1 python-opengl xvfb nano && apt-get update
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

RUN chown $CONTAINER_USER:$CONTAINER_GROUP -R /opt/conda/bin
RUN chown $CONTAINER_USER:$CONTAINER_GROUP -R /opt/conda/lib/python3.7/site-packages
RUN chown $CONTAINER_USER:$CONTAINER_GROUP -R /home/$CONTAINER_USER

COPY --chown=$CONTAINER_USER:$CONTAINER_GROUP extra/themes.jupyterlab-settings /home/$CONTAINER_USER/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/
COPY --chown=$CONTAINER_USER:$CONTAINER_GROUP extra/shortcuts.jupyterlab-settings /home/$CONTAINER_USER/.jupyter/lab/user-settings/@jupyterlab/shortcuts-extension/
COPY --chown=$CONTAINER_USER:$CONTAINER_GROUP extra/tracker.jupyterlab-settings /home/$CONTAINER_USER/.jupyter/lab/user-settings/@jupyterlab/notebook-extension/

USER $CONTAINER_USER
WORKDIR /home/$CONTAINER_USER
ENV PATH="/home/$CONTAINER_USER/.local/bin:${PATH}"

RUN git clone https://github.com/fastai/fastai.git --depth 1 \
        && git clone https://github.com/fastai/fastcore.git --depth 1 \
        && git clone https://github.com/josiahls/fastrl.git --depth 1
RUN /bin/bash -c "if [[ $BUILD == 'prod' ]] ; then echo \"Production Build\" && cd fastai && pip install . && cd ../fastcore && pip install . && cd ../fastrl && pip install .; fi"
# Note that we are not installing the .dev dependencies for fastai or fastcore
RUN /bin/bash -c "if [[ $BUILD == 'dev' ]] ; then echo \"Development Build\" && cd fastai && pip install -e . && cd ../fastcore && pip install -e . cd ../fastrl && pip install -e \".[dev]\"; fi"

RUN echo '#!/bin/bash\npip install fastrl -e \".[dev]\" && xvfb-run -s "-screen 0 1400x900x24" jupyter lab --ip=0.0.0.0 --port=8080 --allow-root --no-browser  --NotebookApp.token='' --NotebookApp.password=''' >> run_jupyter.sh

USER $CONTAINER_USER
RUN /bin/bash -c "cd fastrl && pip install -e ."
RUN chmod u+x run_jupyter.sh