# Fastrl2
> This is a temporary location for fastrl version 2.


# Overview

Fastai for computer vision and tabular learning has been amazing. One would wish that this would be the same for RL. The purpose of this repo is to have a framework that is as easy as possible to start, but also designed for testing new agents.

Documentation is being served  at https://josiahls.github.io/fast-reinforcement-learning-2/ from documentation directly generated via `nbdev` in this repo.

# Current Issues of Interest

## Data Issues
- [ ] data and async_data are still buggy. We need to verify that the order that the data being returned is the best it can be for our models. We need to make sure that "dones" are returned and that there are new duplicate (unless intended)
- [ ] Better data debugging. Do environments skips steps correctly? Do n_steps work correct?

# Whats new?

As we have learned how to support as many RL agents as possible, we found that `fastrl==1.*` was vastly limited in the models that it can support. `fastrl==2.*` will leverage the `nbdev` library for better documentation and more relevant testing. We also will be building on the work of the `ptan`<sup>1</sup> library as a close reference for pytorch based reinforcement learning APIs. 


<sup>1</sup> "Shmuma/Ptan". Github, 2020, https://github.com/Shmuma/ptan. Accessed 13 June 2020.

## Install

## PyPI (Not implemented yet)
Placeholder here, there is no pypi package yet. It is recommended to do traditional forking.

(For future, currently there is no pypi persion)`pip install fastrl==2.0.0 --pre`

## Conda

`conda env create -f environment.yaml`

`source activate fastrl && python setup.py develop`

## Docker (highly recommend)

For cpu execution
```bash
docker build -f fastrl.Dockerfile -t fastrl:latest .
docker run --rm -it -p 8888:8888 -p 4000:4000 --user "$(id -u):$(id -g)" -v $(pwd):/opt/project/fastrl fastrl /bin/bash
```

```bash
docker build -f fastrl_cuda.Dockerfile -t fastrl_cuda:latest .
docker run --rm -it -p 8888:8888 -p 4000:4000 --user "$(id -u):$(id -g)" -v $(pwd):/opt/project/fastrl fastrl_cuda /bin/bash
```

## Contributing

After you clone this repository, please run `nbdev_install_git_hooks` in your terminal. This sets up git hooks, which clean up the notebooks to remove the extraneous stuff stored in the notebooks (e.g. which cells you ran) which causes unnecessary merge conflicts.

Before submitting a PR, check that the local library and notebooks match. The script `nbdev_diff_nbs` can let you know if there is a difference between the local library and the notebooks.
* If you made a change to the notebooks in one of the exported cells, you can export it to the library with `nbdev_build_lib` or `make fastai2`.
* If you made a change to the library, you can export it back to the notebooks with `nbdev_update_lib`.
