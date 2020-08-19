FROM nvidia/cuda:10.0-base-ubuntu18.04
# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8
LABEL com.nvidia.volumes.needed="nvidia_driver"
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates nano curl \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

ENV CONTAINER_USER fastrl
ENV CONTAINER_GROUP fastrl_group
ENV CONTAINER_UID 1000
# Add user to conda
RUN addgroup --gid $CONTAINER_UID $CONTAINER_GROUP && \
    adduser --uid $CONTAINER_UID --gid $CONTAINER_UID $CONTAINER_USER --disabled-password  && \
    mkdir -p /opt/conda && \
    chown $CONTAINER_USER /opt/conda

USER $CONTAINER_USER
USER root

# Add local jekk serving
RUN apt-get update && apt-get install -y --fix-missing ruby-full build-essential zlib1g-dev
RUN echo '# Install Ruby Gems to ~/.gems' >> ~/.bashrc
RUN echo 'export GEM_HOME="$HOME/.gems"' >> ~/.bashrc
RUN echo 'export PATH="$HOME/.gems/bin:$PATH"' >> ~/.bashrc
RUN /bin/bash -c "source ~/.bashrc"
RUN gem install jekyll bundler

WORKDIR /opt/project/fastrl/docs
RUN chown $CONTAINER_USER /opt/project/
COPY --chown=$CONTAINER_USER:$CONTAINER_GROUP docs/Gemfile Gemfile
COPY --chown=$CONTAINER_USER:$CONTAINER_GROUP docs/Gemfile.lock Gemfile.lock
RUN gem i bundler -v 2.0.2
RUN bundle install
WORKDIR /opt/project/fastrl/

# Create Conda env for fastrl from environment.yaml
COPY --chown=$CONTAINER_USER:$CONTAINER_GROUP environment.yaml environment.yaml
RUN conda env create -f environment.yaml
RUN chown -R $CONTAINER_USER /opt/conda/envs/fastrl/ && chmod -R 777 /opt/conda/envs/fastrl/
RUN /bin/bash -c "source activate fastrl && conda install -c conda-forge nodejs ptvsd"
RUN apt-get update && apt-get install -y python-opengl xvfb

COPY --chown=$CONTAINER_USER:$CONTAINER_GROUP . .
RUN ["chmod", "+x", "entrypoint.sh"]
RUN echo 'source activate fastrl' >> ~/.bashrc
RUN /bin/bash -c "source activate fastrl && python setup.py develop"
USER $CONTAINER_USER
ENTRYPOINT ["./entrypoint.sh"]
CMD ["/bin/bash","-c"]


