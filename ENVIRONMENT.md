# SparseAttention conda environment

Created on 2026-09-09 for the current SharePrefill, probability-mean and
Shared6144 + Residual4096 code. This is a reconstructed compatible environment,
not an exact export of the historical AWS environment.

## Use the installed environment

```bash
source /project/yl258/jz648/miniconda3/etc/profile.d/conda.sh
conda activate sparseattention
cd /project/yl258/jz648/sparseAttention
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

Interpreter for the IDE:
`/project/yl258/jz648/miniconda3/envs/sparseattention/bin/python`.
The environment sets `PYTHONNOUSERSITE=1` to exclude packages from `~/.local`.

## Dependencies and provenance

| Component | Version |
| --- | --- |
| Python | 3.10 |
| PyTorch | 2.4.0+cu124 |
| Triton | 3.0.0 |
| Transformers | 4.44.0 |
| FlashAttention | 2.6.3, official Python 3.10 / torch2.4 / CUDA12 wheel |
| NumPy | 1.26.4 |
| Accelerate | 0.33.0 |
| lm-eval | commit `a4987bba6e9e9b3f22bd3a6c1ecf0abd04fd5622` |
| FlexPrefill | commit `a24b27bf117fd6ad4ba34b0db82c598e7786fd59` |

The repository's `experiments/requirements.txt` is only a minimal, unpinned
list. The current runtime additionally imports Triton, lm-eval, FlashAttention
and FlexPrefill. The versions above follow the
[FlexPrefill installer](https://github.com/ByteDance-Seed/FlexPrefill/blob/a24b27bf117fd6ad4ba34b0db82c598e7786fd59/install.sh),
its `extra_requirements.txt` and `setup.py`.
The CUDA wheel comes from the
[official PyTorch CUDA 12.4 index](https://download.pytorch.org/whl/cu124/torch/);
FlashAttention comes from its
[official release](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.6.3).

`requirements-sparseattention.txt` pins the main Python dependencies, including
analysis libraries and InfiniteBench text metrics.
`requirements-sparseattention.lock.txt` records all installed pip packages and
the FlashAttention wheel hash. To replay that snapshot after installing PyTorch
from the CUDA index, use it instead of `requirements-sparseattention.txt`.
Legacy optional Graph2Vec
(`karateclub`/`gensim`), MInference, MoA, DAM, and DuoAttention are outside this
environment's scope. Their historical baseline environments are separate.

## Recreate on Linux x86_64

```bash
conda env create -f environment.yml
conda activate sparseattention
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-sparseattention.txt
```

The FlashAttention requirement is specific to Python 3.10 and Linux x86_64.
Its official wheel is labeled `cu123torch2.4cxx11abiFALSE`.
No local CUDA compiler is needed to install this precompiled extension.

The benchmark needs the FlexPrefill source tree as well as its installed module.
This checkout already has it at `FlexPrefill/`, excluded by `.gitignore`.
For a fresh checkout:

```bash
git clone https://github.com/ByteDance-Seed/FlexPrefill.git FlexPrefill
git -C FlexPrefill checkout a24b27bf117fd6ad4ba34b0db82c598e7786fd59
```

Pass `--flexprefill_root /project/yl258/jz648/sparseAttention/FlexPrefill`
to supported runners. The historical scripts still contain `/home/ubuntu/work`
and `/local` constants. Restore model weights, data, AE head groups and reference
metrics, and adapt those paths before launching experiments; see
`experiments/residual6040_20260908/DEPLOYMENT.md`.
The original `instance2_env.sh` requires the AWS `/local` mount and is not an
activation script for this cluster.

## Validation scope

Installation checks passed on 2026-09-09: `pip check` reported no broken
requirements; all 19 dependency/project-module imports passed, including
FlashAttention, FlexPrefill, `selector6040` and the InfiniteBench runner.
Normal `conda activate sparseattention` selected the correct interpreter with
the user site disabled. Details are in `environment-validation.json`.

The installation node (`n0026`) has no usable NVIDIA driver exposed:
`nvidia-smi` cannot communicate with the driver. CPU imports and dependency
checks can be run here; they do not validate GPU kernels or full inference.
On an allocated GPU node, first check `nvidia-smi` and
`python -c 'import torch; print(torch.cuda.is_available())'`.
Run CUDA correctness checks in a separate output directory so the checked-in
historical `correctness.json` remains an archive.
