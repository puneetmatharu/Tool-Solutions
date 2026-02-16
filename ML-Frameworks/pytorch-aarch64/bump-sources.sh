#!/bin/bash

# SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and affiliates.
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

cat <<EOF

Update references in versions.sh

Commit hashes:
  PyTorch
    URL:  https://github.com/pytorch/pytorch/tree/viable/strict
    Rule: Use the latest commit on this branch (record the commit hash). Build the wheel
          to determine the version used in the trailing comment.

  ideep
    URL:  https://github.com/intel/ideep/tree/ideep_pytorch
    Rule: Use the latest commit on this branch (record the commit hash).

  oneDNN
    URL:  https://github.com/uxlfoundation/oneDNN/tree/main
    Rule: Use the latest commit on this branch (record the commit hash).

  TorchAO
    URL:  https://github.com/pytorch/ao/tree/main
    Rule: Use the latest commit on this branch (record the commit hash).

  KleidiAI
    URL:  https://github.com/ARM-software/kleidiai/tree/main
    Rule: Use the latest commit on this branch (record the commit hash). Use the project
          version from the CMakeLists.txt to set the version in the trailing comment.

Tags:
  ACL
    URL:  https://github.com/ARM-software/ComputeLibrary/tags
    Rule: Pick the newest release tag (record the tag name).

  OpenBLAS
    URL:  https://github.com/OpenMathLib/OpenBLAS/tags
    Rule: Pick the newest release tag (record the tag name).

Nightly wheels:
  Torchvision:
    URL:  https://download.pytorch.org/whl/nightly/torchvision/
    Rule: Pick the newest wheel matching: +cpu, the Python ABI (e.g. cp312 for Python 3.12),
          manylinux_2_28_aarch64.whl.
          Record the dev version from the filename.
    Example: torchvision-0.26.0.dev20260215+cpu-cp312-cp312-manylinux_2_28_aarch64.whl -> 0.26.0.dev20260215

EOF
