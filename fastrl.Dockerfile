FROM pytorch/pytorch

ARG BUILD=dev

RUN apt-get update && apt-get install -y software-properties-common rsync
RUN add-apt-repository -y ppa:git-core/ppa && apt-get update && apt-get install -y git libglib2.0-dev graphviz && apt-get update
RUN pip install "fastprogress>=0.1.22" \
    graphviz \
    jupyter \
    matplotlib \
    "nbconvert<6"\
    nbdev \
    opencv-python \
    pandas \
    pillow \
    pyyaml \
    scikit-learn \
    scikit-image \
    scipy \
    spacy

RUN git clone https://github.com/fastai/fastai.git --depth 1  && git clone https://github.com/fastai/fastcore.git --depth 1 \
        && git clone https://github.com/josiahls/fastrl.git --depth 1
RUN /bin/bash -c "if [[ $BUILD == 'prod' ]] ; then echo \"Production Build\" && cd fastai && pip install . && cd ../fastcore && pip install . && cd ../fastrl && pip install .; fi"
RUN /bin/bash -c "if [[ $BUILD == 'dev' ]] ; then echo \"Development Build\" && cd fastai && pip install -e \".[dev]\" && cd ../fastcore && pip install -e \".[dev]\" cd ../fastrl && pip install -e \".[dev]\"; fi"
RUN echo '#!/bin/bash\njupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser' >> run_jupyter.sh
RUN chmod u+x run_jupyter.sh