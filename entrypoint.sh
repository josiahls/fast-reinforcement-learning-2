#!/bin/bash

source activate fastrl && nbdev_build_docs
cd docs
bundle exec jekyll serve --host=0.0.0.0 --port=4000 &

cd ../
source activate fastrl && python setup.py develop
xvfb-run -s "-screen 0 1400x900x24" jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --notebook-dir=/opt/project/fastrl --allow-root &

sleep 2
bash
exec "$@"