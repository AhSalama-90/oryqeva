# Oryqeva

AI-driven pipeline for designing and validating custom binder proteins for use as low-cost diagnostic detection reagents (ELISA, Western blot, immunoassays) — a domestically producible alternative to expensive imported antibodies.

## Overview

Oryqeva combines protein backbone generation (BindCraft), AI-based sequence design (MoMPNN, a fine-tuned ProteinMPNN model), and a machine-learning developability classifier built on ESM2 protein language model embeddings to predict which candidate binders are experimentally viable and structurally ready for chemical conjugation to a detection label.

## Validated results

- Applied to 1,062 protein designs across 14 disease-relevant protein targets (using open BindCraft-generated backbones released by Anthropic, CC-BY-4.0)
- Developability classifier trained and benchmarked on 2,505 real experimentally characterized designs from [Proteinbase](https://proteinbase.com)
- **AUC-ROC = 0.763**, substantially outperforming standard physicochemical scoring (GRAVY, net charge, hydrophilic fraction combined: AUC-ROC = 0.670)

## Pipeline stages

1. Backbone design (BindCraft integration)
2. Sequence design (MoMPNN)
3. Physicochemical property scoring (GRAVY, net charge, hydrophilic fraction)
4. Conjugation-site scoring (PROPKA)
5. Binding-affinity pre-filtering (PRODIGY)
6. Engineered single-cysteine conjugation site (when no natural site is safely usable)
7. Sortase-tag terminus accessibility checking
8. ESM2-based developability scoring (`score_developability_esm2`)
9. Unified decision function (`get_final_conjugation_strategy`) — combines all of the above into one final ready-to-order sequence + confidence score

## Status

Computational prototype. No wet-lab synthesis or experimental validation of an Oryqeva-designed candidate has been completed yet — this is the current top priority.

## Requirements

```bash
pip install biopython propka prodigy-prot fair-esm scikit-learn
```

## Usage

```python
from oryqeva_pipeline import get_final_conjugation_strategy

result = get_final_conjugation_strategy(
    pdb_path="your_design.pdb",
    sequence="YOUR_PROTEIN_SEQUENCE",
    binder_chain="B",
    target_chain="A",
    esm2_model=model, esm2_alphabet=alphabet, esm2_batch_converter=batch_converter,
)

print(result["strategy"])        # conjugation strategy chosen
print(result["oryqeva_score"])   # developability probability (0-1)
print(result["final_sequence"])  # ready-to-synthesize sequence
```

## Third-party code

`third_party/bindcraft_mompnn/` is a modified copy of [BindCraft](https://github.com/martinpacesa/BindCraft)
(Pacesa et al., MIT License) in which the binder-sequence redesign step uses MoMPNN instead of the built-in
ProteinMPNN/JAX implementation. It keeps its own `LICENSE` and has a `NOTICE.md` listing every change.
MoMPNN is the ProteinMPNN architecture (Dauparas et al., MIT) with a solubility-tuned checkpoint, so ProteinMPNN's
code is still used.

**Status: untested inside a full GPU run, and whether it improves binder quality is not measured.** The default
`run_bindcraft()` still uses a standard BindCraft install; nothing changes unless you point `bindcraft_root` at this copy.

To try it (keep your working BindCraft install untouched):

```bash
ln -s ~/BindCraft/params ~/Oryqeva/third_party/bindcraft_mompnn/params   # reuse the AF2 weights
python3 -m venv ~/mpnn_env && ~/mpnn_env/bin/pip install torch numpy       # PyTorch for MoMPNN
export ORYQEVA_MPNN_PYTHON=~/mpnn_env/bin/python
# then: run_bindcraft(..., bindcraft_root="~/Oryqeva/third_party/bindcraft_mompnn")
```

## Author

Ahmed Salama — Final-year Biotechnology and Genetic Engineering student, Helwan National University, Egypt. Research Assistant at AGERI.

## Data attribution

Backbone structures used in validation originate from Anthropic's open protein-binder-design dataset (CC-BY-4.0). Developability classifier trained on data from [Proteinbase](https://proteinbase.com) (ODC-BY license).
