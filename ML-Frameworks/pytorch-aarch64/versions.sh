#!/bin/bash

# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and affiliates.
#
# SPDX-License-Identifier: Apache-2.0

# Source-of-truth versions and hashes for this repo

# For information on how to update the versions below, read the README.md.

# get-source.sh deps
PYTORCH_HASH=fb69aa6b76c7ddbcac13ac3dd8b14625f4352ffd   # 2.12.0.dev20260328 from viable/strict, Mar 28th
IDEEP_HASH=cbbfd4ad7c5ac6d7683af571055e95c948d8cf54     # From ideep_pytorch, Mar 17th
ONEDNN_HASH=abc14842394f985313191bc1e3c69bb7f8cecd23    # From main, Mar 27th
KLEIDIAI_HASH=6c544373d1731d2b74435fc02f8bad5f4631b0b1  # v1.23.0 from main, Mar 25th

# build-wheel.sh deps
ACL_VERSION="v52.8.0"   # Jan 23rd
OPENBLAS_VERSION="d26960a21ec5da7f77377f28bd6e230060841ae0"  # Mar 27th

# Dockerfile deps
TORCHVISION_NIGHTLY="0.26.0.dev20260329"
TORCHAO_NIGHTLY="0.18.0.dev20260424"
