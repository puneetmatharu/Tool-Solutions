#!/bin/bash

# SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and affiliates.
#
# SPDX-License-Identifier: Apache-2.0

source ./versions.sh

set -eux -o pipefail

help_str="dockerize.sh takes a PyTorch wheel as its one and only argument. It installs
the wheel inside a Docker container with examples and requirements. The docker image
will then be run unless you pass in the optional --build-only argument"

if [ "$#" -lt 1 ]; then
    echo $help_str
    exit 1
fi

if ! [ -e "$1" ]; then
    echo "I couldn't find wheel at $1"
    echo $help_str
    exit 1
fi

docker buildx \
    build --load \
    -t toolsolutions-pytorch:latest  \
    --build-context rootdir=../.. \
    --build-arg DOCKER_IMAGE_MIRROR \
    --build-arg TORCH_WHEEL=$1 \
    --build-arg TORCHAO_NIGHTLY="${TORCHAO_NIGHTLY}" \
    --build-arg TORCHVISION_NIGHTLY="${TORCHVISION_NIGHTLY}" \
    .

[[ $* == *--build-only* ]] && exit 0
docker run --rm -it toolsolutions-pytorch:latest
