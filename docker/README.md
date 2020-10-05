# Containerised Version

Under some circumstances, a containerised version of the utility is useful.
In order to make this easier, the repository contains a Dockerfile for building
a docker container.

The container uses the official Python 3 base image and installs the latest
version found on github.

## Building the Docker Image

You can build the image by simply excuting the following command line:

```sh
wget -O Dockerfile https://raw.githubusercontent.com/patlevin/tfjs-to-tf/master/docker/Dockerfile
docker build -t tfjs_graph_converter .
```

You don't have to have git or Python 3 installed on your system to build
the container - just docker.

## Using the Docker Image

You can run the docker image using `docker run`, e.g.

```sh
docker run --rm tfjs_graph_converter --version
```

Command line arguments are the same as documented in the [README.md](../README.md).
