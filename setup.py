"""
Copyright 2020 The Microsoft DeepSpeed Team

DeepSpeed library

Create a new wheel via the following command: python setup.py bdist_wheel

The wheel will be located at: dist/*.whl
"""

import os
import torch
import shutil
import subprocess
import warnings
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CppExtension

import op_builder

VERSION = "0.3.0"


def fetch_requirements(path):
    with open(path, 'r') as fd:
        return [r.strip() for r in fd.readlines()]


install_requires = fetch_requirements('requirements/requirements.txt')
dev_requires = fetch_requirements('requirements/requirements-dev.txt')
sparse_attn_requires = fetch_requirements('requirements/requirements-sparse-attn.txt')

# Add development requirements
install_requires += dev_requires

# If MPI is available add 1bit-adam requirements
if torch.cuda.is_available():
    if shutil.which('ompi_info') or shutil.which('mpiname'):
        onebit_adam_requires = fetch_requirements(
            'requirements/requirements-1bit-adam.txt')
        onebit_adam_requires.append(f"cupy-cuda{torch.version.cuda.replace('.','')[:3]}")
        install_requires += onebit_adam_requires

cmdclass = {}
cmdclass['build_ext'] = BuildExtension.with_options(use_ninja=False)

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

if not torch.cuda.is_available():
    # Fix to allow docker buils, similar to https://github.com/NVIDIA/apex/issues/486
    print(
        "[WARNING] Torch did not find cuda available, if cross-compling or running with cpu only "
        "you can ignore this message. Adding compute capability for Pascal, Volta, and Turing "
        "(compute capabilities 6.0, 6.1, 6.2)")
    if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5"

ext_modules = []

from op_builder import ALL_OPS

# Default to pre-install kernels to false so we rely on JIT
OP_DEFAULT = int(os.environ.get('DS_BUILD_CUDA', 0))
print(f"DS_BUILD_CUDA={OP_DEFAULT}")


def command_exists(cmd):
    result = subprocess.Popen('type ninja', stdout=subprocess.PIPE, shell=True)
    return result.wait() == 0


if not OP_DEFAULT and not command_exists('ninja'):
    raise Exception("Delaying DeepSpeed op installation will result in ops being JIT "
                    "compiled, this requires ninja to be installed. Please install "
                    "ninja e.g., apt-get install build-ninja")


def op_enabled(op_name):
    assert hasattr(ALL_OPS[op_name], 'BUILD_VAR'), \
        f"{op_name} is missing BUILD_VAR field"
    env_var = ALL_OPS[op_name].BUILD_VAR
    return int(os.environ.get(env_var, OP_DEFAULT))


install_ops = dict.fromkeys(ALL_OPS.keys(), False)
for op_name in ALL_OPS.keys():
    # Is op disabled from environment variable
    if op_enabled(op_name):
        builder = ALL_OPS[op_name]
        # Is op compatible with machine arch/deps
        if builder.is_compatible():
            install_ops[op_name] = op_enabled(op_name)
            ext_modules.append(builder.builder())

compatible_ops = {op_name: op.is_compatible() for (op_name, op) in ALL_OPS.items()}

print(f'Install Ops={install_ops}')

# Write out version/git info
git_hash_cmd = "git rev-parse --short HEAD"
git_branch_cmd = "git rev-parse --abbrev-ref HEAD"
if command_exists('git'):
    result = subprocess.check_output(git_hash_cmd, shell=True)
    git_hash = result.decode('utf-8').strip()
    result = subprocess.check_output(git_branch_cmd, shell=True)
    git_branch = result.decode('utf-8').strip()
else:
    git_hash = "unknown"
    git_branch = "unknown"
print(f"version={VERSION}+{git_hash}, git_hash={git_hash}, git_branch={git_branch}")
with open('deepspeed/git_version_info_installed.py', 'w') as fd:
    fd.write(f"version='{VERSION}+{git_hash}'\n")
    fd.write(f"git_hash='{git_hash}'\n")
    fd.write(f"git_branch='{git_branch}'\n")
    fd.write(f"installed_ops={install_ops}\n")
    fd.write(f"compatible_ops={compatible_ops}\n")

print(f'install_requires={install_requires}')
print(f'compatible_ops={compatible_ops}')
print(f'ext_modules={ext_modules}')

setup(name='deepspeed',
      version=f"{VERSION}+{git_hash}",
      description='DeepSpeed library',
      author='DeepSpeed Team',
      author_email='deepspeed@microsoft.com',
      url='http://deepspeed.ai',
      install_requires=install_requires,
      packages=find_packages(exclude=["docker",
                                      "third_party"]),
      include_package_data=True,
      scripts=[
          'bin/deepspeed',
          'bin/deepspeed.pt',
          'bin/ds',
          'bin/ds_ssh',
          'bin/ds_report'
      ],
      classifiers=[
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8'
      ],
      license='MIT',
      ext_modules=ext_modules,
      cmdclass=cmdclass)
