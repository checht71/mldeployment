# mldeployment
Repo with code used in the "[How to Deploy ML Models in Production with BentoML](https://www.youtube.com/watch?v=HHkmfI_yncc)" video tutorial on The Sound of AI channel.

This code has been repurposed to work with a PyTorch model.

A few notes on bugs I encountered:
1. You must use an environment with python 3.10
2. You need to install the official version of docker to containerize this model. Here's the commands to do it on linux:
   `curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh`
3. If containerizing doesn't work, run it with the `--debug` tag
4. Make sure the model accepts Numpy arrays as input. BentoML does not accept torch tensors as input. [Here's a list of other i/o options.](https://docs.bentoml.com/en/latest/guides/iotypes.html)
