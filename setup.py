# =============================================================================
# Based on ActionCLIP:
#   "ActionCLIP: A New Paradigm for Action Recognition"
#   Mengmeng Wang, Jiazheng Xing, Yong Liu
#   arXiv:2109.08472
#   https://github.com/sallymmx/ActionCLIP (MIT License)
#
# This repository contains substantial modifications and extensions
# made by Yoonseon Oh (2025), including changes to model architecture,
# training/evaluation pipeline, and efficiency logging.
#
# License: MIT (see LICENSE file in the repository root)
# =============================================================================

import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="clip",
    py_modules=["clip"],
    version="1.0",
    description="",
    author="OpenAI",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True,
    extras_require={'dev': ['pytest']},
)
