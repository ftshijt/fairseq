#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import sys

from setuptools import Extension, find_packages, setup

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python >= 3.6 is required for fairseq.")


def write_version_py():
    with open(os.path.join("fairseq", "version.txt")) as f:
        version = f.read().strip()

    # write version info to fairseq/version.py
    with open(os.path.join("fairseq", "version.py"), "w") as f:
        f.write('__version__ = "{}"\n'.format(version))
    return version


version = write_version_py()


with open("README.md") as f:
    readme = f.read()


if sys.platform == "darwin":
    extra_compile_args = ["-stdlib=libc++", "-O3"]
else:
    extra_compile_args = ["-std=c++11", "-O3"]


class NumpyExtension(Extension):
    """Source: https://stackoverflow.com/a/54128391"""

    def __init__(self, *args, **kwargs):
        self.__include_dirs = []
        super().__init__(*args, **kwargs)

    @property
    def include_dirs(self):
        import numpy

        return self.__include_dirs + [numpy.get_include()]

    @include_dirs.setter
    def include_dirs(self, dirs):
        self.__include_dirs = dirs




def do_setup(package_data):
    setup(
        name="fairseq",
        version=version,
        description="Facebook AI Research Sequence-to-Sequence Toolkit",
        url="https://github.com/pytorch/fairseq",
        classifiers=[
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        long_description=readme,
        long_description_content_type="text/markdown",
        install_requires=[
            "cffi",
            "cython",
            "hydra-core",
            "omegaconf",
            "numpy>=1.21.3",
            "regex",
            "sacrebleu>=1.4.12",
            "torch>=1.13",
            "tqdm",
            "bitarray",
            "torchaudio",
            "scikit-learn",
            "packaging",
        ],
        extras_require={
            "dev": ["flake8", "pytest", "black==22.3.0"],
            "docs": ["sphinx", "sphinx-argparse"],
        },
        packages=find_packages(
            exclude=[
                "examples",
                "examples.*",
                "scripts",
                "scripts.*",
                "tests",
                "tests.*",
            ]
        ),
        package_data=package_data,
        test_suite="tests",
        entry_points={
            "console_scripts": [
                "fairseq-eval-lm = fairseq_cli.eval_lm:cli_main",
                "fairseq-generate = fairseq_cli.generate:cli_main",
                "fairseq-hydra-train = fairseq_cli.hydra_train:cli_main",
                "fairseq-interactive = fairseq_cli.interactive:cli_main",
                "fairseq-preprocess = fairseq_cli.preprocess:cli_main",
                "fairseq-score = fairseq_cli.score:cli_main",
                "fairseq-train = fairseq_cli.train:cli_main",
                "fairseq-validate = fairseq_cli.validate:cli_main",
            ],
        },
        zip_safe=False,
    )


def get_files(path, relative_to="fairseq"):
    all_files = []
    for root, _dirs, files in os.walk(path, followlinks=True):
        root = os.path.relpath(root, relative_to)
        for file in files:
            if file.endswith(".pyc"):
                continue
            all_files.append(os.path.join(root, file))
    return all_files


if __name__ == "__main__":
    try:
        # symlink examples into fairseq package so package_data accepts them
        fairseq_examples = os.path.join("fairseq", "examples")
        if "build_ext" not in sys.argv[1:] and not os.path.exists(fairseq_examples):
            os.symlink(os.path.join("..", "examples"), fairseq_examples)

        package_data = {
            "fairseq": (
                get_files(fairseq_examples)
                + get_files(os.path.join("fairseq", "config"))
            )
        }
        do_setup(package_data)
    finally:
        if "build_ext" not in sys.argv[1:] and os.path.islink(fairseq_examples):
            os.unlink(fairseq_examples)
