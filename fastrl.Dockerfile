FROM continuumio/miniconda:4.7.12

RUN apt-get update -y && apt-get install -y nano make curl jq

ENV DEBIAN_FRONTEND noninteractive
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

# Create Conda env for fastrl from environment.yaml
WORKDIR /opt/project/fastrl
COPY --chown=$CONTAINER_USER:$CONTAINER_GROUP environment.yaml environment.yaml
RUN conda env create -f environment.yaml
RUN chown -R $CONTAINER_USER /opt/conda/envs/fastrl/ && chmod -R 777 /opt/conda/envs/fastrl/
RUN /bin/bash -c "source activate fastrl && conda install -c conda-forge nodejs ptvsd"
#RUN /bin/bash -c "source activate fastrl && jupyter labextension install @jupyterlab/debugger"

# Add local jekk serving
RUN apt-get update && apt-get install -y --fix-missing ruby-full build-essential zlib1g-dev
RUN echo '# Install Ruby Gems to ~/.gems' >> ~/.bashrc
RUN echo 'export GEM_HOME="$HOME/.gems"' >> ~/.bashrc
RUN echo 'export PATH="$HOME/.gems/bin:$PATH"' >> ~/.bashrc
RUN /bin/bash -c "source ~/.bashrc"
RUN gem install jekyll
RUN gem i bundler -v 2.0.2

RUN chown $CONTAINER_USER /opt/project/
WORKDIR /opt/project/fastrl/docs
COPY --chown=$CONTAINER_USER:$CONTAINER_GROUP docs/Gemfile Gemfile
COPY --chown=$CONTAINER_USER:$CONTAINER_GROUP docs/Gemfile.lock Gemfile.lock
RUN bundle install
WORKDIR /opt/project/fastrl/

EXPOSE 8888
EXPOSE 4000

ENV PATH /opt/conda/envs/fastrl/bin:$PATH
ENV CONDA_DEFAULT_ENV fastrl
ENV FONTCONFIG_FILE $CONDA_PREFIX/etc/fonts/fonts.conf
ENV FONTCONFIG_PATH $CONDA_PREFIX/etc/fonts/
WORKDIR /opt/project/fastrl

USER root
COPY --chown=$CONTAINER_USER:$CONTAINER_GROUP fastrl fastrl
COPY --chown=$CONTAINER_USER:$CONTAINER_GROUP settings.ini settings.ini
COPY --chown=$CONTAINER_USER:$CONTAINER_GROUP README.md README.md
RUN /bin/bash -c "source activate fastrl && jt -t grade3 -cellw=90% -fs=20 -tfs=20 -ofs=20"
#COPY --chown=$CONTAINER_USER:$CONTAINER_GROUP entrypoint.sh /usr/local/bin/
COPY --chown=$CONTAINER_USER:$CONTAINER_GROUP entrypoint.sh entrypoint.sh
#RUN chmod +x /usr/local/bin/entrypoint.sh
RUN chmod +x entrypoint.sh
RUN mkdir /home/fastrl/.gem && \
        chown $CONTAINER_USER -R /home/$CONTAINER_USER/.gem && \
        chown $CONTAINER_USER -R /opt/project

#WORKDIR /opt/project/
1
#RUN chown -R $CONTAINER_USER /opt/conda/envs/fastrl && chmod -R 777 /opt/conda/envs/fastrl

RUN apt-get install -y python-opengl
RUN apt-get install -y xvfb
USER $CONTAINER_USER
RUN conda init bash
COPY --chown=$CONTAINER_USER:$CONTAINER_GROUP . .
# Install the deploy package for system access
ENTRYPOINT "entrypoint.sh"
RUN echo 'source activate fastrl' >> ~/.bashrc
CMD ["/bin/bash","-c"]
RUN /bin/bash -c "source activate fastrl && pip install ptan --no-dependencies && python setup.py develop"