# Notice

This repository is a modified copy of **BindCraft** by Martin Pacesa and colleagues
(https://github.com/martinpacesa/BindCraft), distributed under the MIT License. The
original copyright notice and license text are kept unchanged in `LICENSE`.

If you use this code, please cite BindCraft:
Pacesa, M. et al. One-shot design of functional protein binders with BindCraft. Nature (2025).
and ProteinMPNN: Dauparas, J. et al. Robust deep learning-based protein sequence design using ProteinMPNN. Science (2022).

## What was changed

The binder-sequence redesign step (originally ProteinMPNN through ColabDesign/JAX) now uses **MoMPNN**:
the official ProteinMPNN code run with a solubility-tuned checkpoint (PyTorch). The upstream
implementation of that step was removed from this fork.

| File | Change |
|---|---|
| `functions/mompnn_backend.py` | New. Runs MoMPNN through the official ProteinMPNN script, fetches the script and checkpoint on first use, and returns results in the shape BindCraft expects. |
| `functions/colabdesign_utils.py` | Body of `mpnn_gen_sequence()` replaced by a call to the backend; the now-unused `from colabdesign.mpnn import mk_mpnn_model` removed. |

Everything else (trajectory design, AF2 re-prediction of every MPNN sequence, filters, ranking, CSVs) is upstream code, unchanged.
The settings `mpnn_weights` and `model_path` are no longer read; they remain in the profiles only so existing settings files still load.
`MPNN_score` / `MPNN_seq_recovery` come from a different model than upstream's and are not comparable with upstream runs.

## Requirements added

PyTorch must be available to the Python that runs ProteinMPNN. BindCraft's conda environment does not include it.
Install it there, or point `ORYQEVA_MPNN_PYTHON` at another interpreter that has it. If the GPU is newer than the
installed PyTorch supports, the step is retried on CPU automatically (fast enough for a model this small).
Cache location: `~/.oryqeva` (override with `ORYQEVA_CACHE_DIR`).

## Third-party components

- **ProteinMPNN** (MIT License, https://github.com/dauparas/ProteinMPNN) is used unmodified.
- **PyRosetta** is a separate dependency with its own license; commercial use requires a license from the University of Washington. It is not distributed here.

## Status

Tested: the backend end to end with the real ProteinMPNN script and the real MoMPNN checkpoint on CPU, on a synthetic
idealised-helix backbone (plumbing only: 20 sequences returned, interface residues kept, omitted residues respected).
Not tested: inside a full BindCraft run on a GPU, and nothing about binder quality.
Whether MoMPNN gives better binders than BindCraft's built-in step is **not measured**; BindCraft's default already uses
solubility-trained ProteinMPNN weights (`"mpnn_weights": "soluble"`). Do not claim an improvement without a head-to-head comparison.

## Provenance

This directory is a vendored copy inside the Oryqeva repository (`third_party/bindcraft_mompnn/`), not a GitHub fork.
It is based on upstream BindCraft commit `7713aa0d0d351e4117a8befeb8541f3a8ebd3368`
(https://github.com/martinpacesa/BindCraft). Omitted from the copy: the upstream `notebooks/` folder
(Colab notebooks that install and run upstream BindCraft, not this version). Upstream `LICENSE` is kept unchanged here;
it applies to this directory. Oryqeva's own MIT license (repository root) applies to the rest of the repository.
