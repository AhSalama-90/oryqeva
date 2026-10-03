"""
Oryqeva — end-to-end binder-design pipeline

Status of each stage (2026-09):
  STAGE 1  Input handling           -> DONE
  STAGE 1.5 IARA hotspot triage    -> NEW (recommends binding-viable regions
                                        BEFORE BindCraft runs, to avoid
                                        wasting GPU hours on dead targets —
                                        would have flagged SC2RBD/BHRF1 early)
  STAGE 2  Backbone design          -> DONE (BindCraft, run separately in Colab;
                                        this module reads its output folder)
  STAGE 3  Sequence design (MoMPNN) -> DONE (validated: p<0.001 on 9 proteins)
  STAGE 4  Property scoring         -> DONE (GRAVY, net charge, hydrophilic
                                        fraction; SASA needs freesasa installed)
  STAGE 5  Conjugation scoring      -> DONE (PROPKA, real API, binder-chain only)
  STAGE 5.5 PRODIGY pre-filter      -> NEW (fast structure-based affinity guess,
                                        runs before the expensive AF2 step)
  STAGE 6  Active-site shielding    -> TODO (EditorForge, not started; likely
                                        not needed since binders have no
                                        catalytic active site)
  STAGE 7  Ensemble w/ DGIF         -> TODO (not started)
  STAGE 8  Final AF2 confirmation   -> TODO (not started, needs GPU)
  STAGE 9  Ranking / classification -> DONE (thresholds tuned on real PD-L1 data)

Run this in the same Colab session where BindCraft and MoMPNN are already
set up. Two installs are needed the first time per session:
    !pip install propka --quiet
    !pip install prodigy-prot --quiet
"""

import os
import re
import glob
import json
import subprocess
import csv
import sys
import time
import datetime
from dataclasses import dataclass, field
from typing import Optional

# ── project paths ────────────────────────────────────────────────────
# DRIVE_ROOT only matters for Colab (where BindCraft results live on Drive);
# it is overridden by run_bindcraft()/the notebook on any other machine.
DRIVE_ROOT = "/content/drive/MyDrive"
BINDCRAFT_ROOT = f"{DRIVE_ROOT}/BindCraft"
ORYQEVA_ROOT = f"{DRIVE_ROOT}/Oryqeva"

# MoMPNN and ProteinMPNN are fetched on demand into a stable, OS-independent
# cache under the user's home directory -- this works the same on Colab, a
# local Linux/WSL machine, or any other environment, with no manual copying.
ORYQEVA_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".oryqeva")
MOMPNN_CKPT = os.path.join(ORYQEVA_CACHE_DIR, "checkpoints", "mompnn_sol.pt")
MOMPNN_CKPT_URL = "https://raw.githubusercontent.com/AhSalama-90/oryqeva/main/checkpoints/mompnn_run/mompnn_sol.pt"
PROTEINMPNN_REPO_URL = "https://github.com/dauparas/ProteinMPNN.git"
PROTEINMPNN_DIR = os.path.join(ORYQEVA_CACHE_DIR, "ProteinMPNN")


@dataclass
class DesignRequest:
    """What the 'Run design' button in the UI mockup sends."""
    target_name: str                 # e.g. "PDL1"
    starting_pdb: str                 # path to target .pdb
    chains: str = "A"                 # target chain (BindCraft convention)
    binder_chain: str = "B"           # binder chain (BindCraft convention)
    hotspot_residues: str = ""       # e.g. "56,115,123"
    assay_type: str = "western_blot"  # western_blot | elisa | immunoassay
    num_candidates: int = 5


@dataclass
class CandidateResult:
    design_id: str
    sequence: str
    stability: Optional[float] = None
    solubility_gravy: Optional[float] = None
    net_charge: Optional[float] = None
    hydrophilic_fraction: Optional[float] = None
    surface_sasa_fraction: Optional[float] = None
    conjugation_sites: Optional[int] = None
    conjugation_total_lys_cys: Optional[int] = None
    conjugation_excluded_disulfide: Optional[int] = None
    prodigy_dg: Optional[float] = None            # PRODIGY dG (kcal/mol)
    prodigy_kd: Optional[float] = None             # PRODIGY predicted Kd (M)
    binding_confidence: Optional[float] = None      # TODO: needs Stage 8 (AF2)
    classification: str = "PENDING"


# ── STAGE 1.5: IARA hotspot triage (NEW, runs BEFORE BindCraft) ─────────
def suggest_hotspots_iara(
    target_pdb_path: str,
    target_chain: str = "A",
    top_n: int = 5,
    score_threshold: float = 0.5,
) -> dict:
    """
    Runs IARA (github.com/leodeals/IARA) on a target structure and returns
    the residues it predicts are most "designable" — i.e. most likely to
    support a successful de novo binder, based on a GNN trained on real
    BindCraft trajectory outcomes.

    Use this BEFORE spending GPU hours on BindCraft: if IARA's top score
    is below `score_threshold`, that is an early warning the target may
    behave like SC2RBD/BHRF1 did today (backbones hallucinate fine, but
    every downstream sequence fails AF2 filters) — worth trying a
    different trim/hotspot set, or a different target, before committing
    a full BindCraft run.

    NOTE: this wraps IARA's command-line script. The exact CLI flag names
    below are based on the project's documented usage pattern, not a
    verified run — check `python iara_predict.py --help` in the installed
    repo once and adjust the `cmd` list if the flags differ.

    Install once per session (outside this function):
        !git clone https://github.com/leodeals/IARA /content/iara
        # then follow the repo's conda env setup instructions
    """
    import subprocess
    import json as _json

    result = {"top_residues": [], "top_score": None, "error": None}
    try:
        cmd = [
            "python", "/content/iara/iara_predict.py",
            "--pdb", target_pdb_path,
            "--chain", target_chain,
            "--output_format", "json",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

        if proc.returncode != 0:
            result["error"] = proc.stderr[:300]
            return result

        scores = _json.loads(proc.stdout)
        # Expect a list of {"residue": int, "score": float} — sort and trim.
        ranked = sorted(scores, key=lambda r: r["score"], reverse=True)
        result["top_residues"] = ranked[:top_n]
        result["top_score"] = ranked[0]["score"] if ranked else None
    except Exception as e:
        result["error"] = str(e)
    return result


# ── STAGE 2: read BindCraft output (already run manually per target) ────
def load_bindcraft_designs(target_name: str) -> list[str]:
    """
    Returns paths to accepted design PDBs from a finished BindCraft run.
    Assumes the standard BindCraft output layout under BINDCRAFT_ROOT/<target_name>/.
    These PDBs are the FULL COMPLEX (target + binder chains together),
    which is what both PROPKA and PRODIGY need as input.
    """
    pattern = f"{DRIVE_ROOT}/BindCraft/{target_name}/Accepted/*.pdb"
    designs = sorted(glob.glob(pattern))
    if not designs:
        raise FileNotFoundError(
            f"No accepted BindCraft designs found for '{target_name}'. "
            f"Run BindCraft on this target first."
        )
    return designs


# ── STAGE 3: MoMPNN sequence design (reuses the exact call we validated) ─
def ensure_mompnn_checkpoint(checkpoint_path: str = None, url: str = None) -> str:
    """
    Returns a local path to the solubility-tuned MoMPNN checkpoint,
    downloading it from the Oryqeva GitHub repo the first time it is
    needed on this machine. Safe to call every run -- it is a no-op once
    the file is cached.
    """
    checkpoint_path = checkpoint_path or MOMPNN_CKPT
    url = url or MOMPNN_CKPT_URL
    if os.path.exists(checkpoint_path):
        return checkpoint_path

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    try:
        import requests
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        tmp_path = checkpoint_path + ".part"
        with open(tmp_path, "wb") as f:
            f.write(resp.content)
        os.replace(tmp_path, checkpoint_path)
    except Exception as e:
        raise InputFailure(
            f"Could not download the MoMPNN checkpoint from {url}: {e}. "
            f"Place it manually at {checkpoint_path} and re-run."
        )
    return checkpoint_path


def ensure_proteinmpnn_script(repo_dir: str = None) -> str:
    """
    Returns the local path to ProteinMPNN's protein_mpnn_run.py, cloning
    the official ProteinMPNN repository the first time it is needed on
    this machine (it ships only the base-model architecture code; the
    actual weights used are the MoMPNN checkpoint from
    ensure_mompnn_checkpoint(), passed in separately).
    """
    repo_dir = repo_dir or PROTEINMPNN_DIR
    script_path = os.path.join(repo_dir, "protein_mpnn_run.py")
    if os.path.exists(script_path):
        return script_path

    os.makedirs(os.path.dirname(repo_dir), exist_ok=True)
    try:
        proc = subprocess.run(
            ["git", "clone", "--quiet", "--depth", "1", PROTEINMPNN_REPO_URL, repo_dir],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0 or not os.path.exists(script_path):
            raise RuntimeError(proc.stderr[:300])
    except Exception as e:
        raise InputFailure(
            f"Could not clone ProteinMPNN from {PROTEINMPNN_REPO_URL}: {e}. "
            f"Clone it manually to {repo_dir} and re-run."
        )
    return script_path


def run_mompnn(
    pdb_path: str,
    out_folder: str,
    chains: str = "A",
    python_exe: str = None,
) -> str:
    """
    Redesigns the given chain(s) with the solubility-tuned MoMPNN
    checkpoint, fetching the checkpoint and the ProteinMPNN script
    automatically on first use (see ensure_mompnn_checkpoint and
    ensure_proteinmpnn_script -- no manual setup required on any machine).
    Returns the path to the resulting .fa file.
    """
    checkpoint_path = ensure_mompnn_checkpoint()
    script_path = ensure_proteinmpnn_script()
    # Lets MPNN run in its own env (e.g. one whose PyTorch build supports newer GPUs)
    py = python_exe or os.environ.get("ORYQEVA_MPNN_PYTHON") or sys.executable

    pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]
    os.makedirs(out_folder, exist_ok=True)

    cmd = [
        py, script_path,
        "--pdb_path", pdb_path,
        "--pdb_path_chains", chains,
        "--out_folder", out_folder,
        "--path_to_model_weights", os.path.dirname(checkpoint_path),
        "--model_name", os.path.splitext(os.path.basename(checkpoint_path))[0],
        "--num_seq_per_target", "8",
        "--sampling_temp", "0.1",
        "--seed", "37",
        "--batch_size", "1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 and "CUDA" in proc.stderr:
        # Newer GPUs (e.g. RTX 50-series) can be unsupported by the installed PyTorch
        # build ("no kernel image is available"). MPNN is small, so retry on CPU.
        print("MoMPNN: GPU run failed with a CUDA error; retrying on CPU.")
        cpu_env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=cpu_env)
    if proc.returncode != 0:
        raise BindCraftFailure(f"MoMPNN run failed: {proc.stderr[-800:]}")

    return f"{out_folder}/seqs/{pdb_id}.fa"


def extract_sequences(fasta_path: str) -> list[str]:
    with open(fasta_path) as f:
        lines = f.read().strip().split("\n")
    seqs = [l.strip() for l in lines if not l.startswith(">")]
    return seqs[1:]  # drop the reference sequence


def extract_binder_sequences(fasta_path: str, pdb_path: str, binder_chain: str = "B") -> list[str]:
    """
    ProteinMPNN writes every chain of the complex to the .fa file, joined by
    "/" in alphabetical chain order. Returns only the binder-chain sequence
    of each redesigned variant (the native record is dropped).
    """
    from Bio.PDB import PDBParser
    chain_ids = sorted(c.id for c in PDBParser(QUIET=True).get_structure("s", pdb_path)[0])
    idx = chain_ids.index(binder_chain)
    out = []
    for s in extract_sequences(fasta_path):
        parts = s.split("/")
        out.append(parts[idx] if len(parts) == len(chain_ids) else s)
    return out


# ── STAGE 4: property scoring (the four metrics validated on 9 proteins) ─
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def score_gravy(seq: str) -> Optional[float]:
    if not set(seq).issubset(STANDARD_AA):
        return None
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    return ProteinAnalysis(seq).gravy()


def score_net_charge(seq: str) -> float:
    pos = seq.count("K") + seq.count("R")
    neg = seq.count("D") + seq.count("E")
    return (pos - neg) / len(seq) * 100


def score_hydrophilic_fraction(seq: str) -> float:
    hydrophilic = set("RKDENQST")
    return sum(1 for aa in seq if aa in hydrophilic) / len(seq)


# ── STAGE 5: conjugation scoring (PROPKA — real API) ─────────────────────
def score_conjugation_sites(
    pdb_path: str,
    binder_chain: str = "B",
    pka_min: float = 6.0,
    pka_max: float = 11.0,
) -> Optional[dict]:
    """
    Scores LYS/CYS residues on the binder chain for suitability as
    bioconjugation sites (NHS-ester on LYS, maleimide on CYS), using
    PROPKA's predicted pKa values.

    binder_chain: which chain in the complex PDB is the BINDER (BindCraft
                  convention: target = chain A, binder = chain B).

    Excludes:
      - CYS involved in a disulfide bond (PROPKA reports pKa ~99.99 for
        these — they're chemically inert, not free thiols).
      - Residues whose predicted pKa falls outside [pka_min, pka_max]:
        too high means the side chain is buried/H-bonded and won't be
        reactive at standard conjugation buffer pH (~8.5-9 for NHS-ester,
        ~7.0-7.5 for maleimide).

    Returns a dict with counts, or None if PROPKA isn't available/fails.
    Install once per session: !pip install propka --quiet
    """
    try:
        import propka.run as pk
    except ImportError:
        return None

    try:
        mol = pk.single(pdb_path, optargs=["--quiet"])
    except Exception:
        return None

    total_lys_cys = 0
    valid_sites = 0
    excluded_disulfide = 0
    excluded_buried = 0

    for conf_name, conformation in mol.conformations.items():
        for group in conformation.groups:
            res_type = getattr(group, "residue_type", None)
            if res_type not in ("LYS", "CYS"):
                continue

            atom = group.atom
            if atom.chain_id != binder_chain:
                continue  # only score the binder, not the target

            total_lys_cys += 1
            pka = group.pka_value

            if res_type == "CYS" and pka > 90:  # PROPKA disulfide convention
                excluded_disulfide += 1
                continue

            if not (pka_min <= pka <= pka_max):
                excluded_buried += 1
                continue

            valid_sites += 1

        break  # single-conformation structures only have one entry; stop after it

    return {
        "total_lys_cys": total_lys_cys,
        "valid_conjugation_sites": valid_sites,
        "excluded_disulfide": excluded_disulfide,
        "excluded_out_of_pka_range": excluded_buried,
    }


# ── STAGE 5.5: PRODIGY binding-affinity pre-filter ───────────────────────
def score_prodigy_affinity(
    complex_pdb_path: str,
    temperature_celsius: float = 25.0,
) -> dict:
    """
    Runs PRODIGY on a binder-target complex PDB and parses predicted
    binding free energy (dG) and dissociation constant (Kd).

    complex_pdb_path: path to the PDB of the FULL COMPLEX (binder + target
                       chains together) — this is exactly what
                       load_bindcraft_designs() returns, so no extra
                       preparation is needed.

    Install once per session: !pip install prodigy-prot --quiet
    """
    result = {"prodigy_dg": None, "prodigy_kd": None, "prodigy_error": None}
    try:
        cmd = [
            "prodigy",
            complex_pdb_path,
            "--temperature", str(temperature_celsius),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = proc.stdout

        dg_match = re.search(
            r"Predicted binding affinity \(kcal\.mol-1\):\s*(-?\d+\.?\d*)", output
        )
        kd_match = re.search(
            r"Predicted dissociation constant.*?:\s*([\d.eE+-]+)", output
        )

        if dg_match:
            result["prodigy_dg"] = float(dg_match.group(1))
        if kd_match:
            result["prodigy_kd"] = float(kd_match.group(1))
        if not dg_match and not kd_match:
            result["prodigy_error"] = f"Could not parse PRODIGY output: {output[:200]}"
    except Exception as e:
        result["prodigy_error"] = str(e)
    return result


# ── STAGE 8: AF2 confirmation (not built yet) ────────────────────────────
def confirm_binding(pdb_path: str, seq: str) -> Optional[float]:
    """TODO: AlphaFold2 Multimer re-scoring. Returns None until built."""
    return None


# ── STAGE 9: ranking (tuned on real PD-L1 GRAVY + conjugation data) ─────
GRAVY_P33 = -0.3535
GRAVY_P67 = -0.2931
CONJ_P33 = 9.0
CONJ_P67 = 17.0


def classify(result: CandidateResult) -> str:
    if result.solubility_gravy is None:
        return "PENDING"
    conj = result.conjugation_sites if result.conjugation_sites is not None else CONJ_P33
    if result.solubility_gravy <= GRAVY_P33 and conj >= CONJ_P67:
        return "PRIORITIZE"
    if result.solubility_gravy >= GRAVY_P67 and conj <= CONJ_P33:
        return "ABSTAIN"
    return "REVIEW"


# ── Orchestrator: what the "Run design" button actually calls ───────────
def run_pipeline(
    request: DesignRequest,
    esm2_model=None,
    esm2_alphabet=None,
    esm2_batch_converter=None,
) -> list[CandidateResult]:
    """
    End-to-end pipeline: BindCraft designs -> MoMPNN -> property scoring
    -> conjugation strategy -> classify_v2.

    Pass esm2_model/esm2_alphabet/esm2_batch_converter (loaded once per
    session) to get the binding_prior_unvalidated score. Omit to skip.

    IMPORTANT: PROPKA/PRODIGY values are unreliable on backbone-only
    structures (structures without side chains). They are computed here
    for completeness but should be treated as approximate until AF2
    structures are available.
    """
    design_pdbs = load_bindcraft_designs(request.target_name)[: request.num_candidates]

    results = []
    for pdb_path in design_pdbs:
        out_folder = f"{ORYQEVA_ROOT}/results/{request.target_name}"
        fasta_path = run_mompnn(pdb_path, out_folder, request.binder_chain)
        sequences = extract_binder_sequences(fasta_path, pdb_path, request.binder_chain)

        # structure-based scores run once per backbone (same for all sequences)
        prodigy = score_prodigy_affinity(pdb_path)

        for i, seq in enumerate(sequences):
            # get_final_conjugation_strategy handles PROPKA + terminus check
            # + engineered Cys + ESM2 score in one call
            conj_result = get_final_conjugation_strategy(
                pdb_path=pdb_path,
                sequence=seq,
                binder_chain=request.binder_chain,
                target_chain=request.chains,
                esm2_model=esm2_model,
                esm2_alphabet=esm2_alphabet,
                esm2_batch_converter=esm2_batch_converter,
            )
            propka = conj_result.get("propka_result") or {}

            result = CandidateResult(
                design_id=f"{os.path.basename(pdb_path)}_seq{i+1}",
                sequence=conj_result["final_sequence"],
                solubility_gravy=score_gravy(seq),
                net_charge=score_net_charge(seq),
                hydrophilic_fraction=score_hydrophilic_fraction(seq),
                conjugation_sites=propka.get("valid_conjugation_sites"),
                conjugation_total_lys_cys=propka.get("total_lys_cys"),
                conjugation_excluded_disulfide=propka.get("excluded_disulfide"),
                prodigy_dg=prodigy["prodigy_dg"],
                prodigy_kd=prodigy["prodigy_kd"],
                binding_confidence=None,  # populated after AF2 (Stage 8)
            )
            oryqeva_score = conj_result.get("oryqeva_score")
            result.classification = classify_v2(oryqeva_score, conj_result["strategy"])
            results.append(result)

    return results


# __main__ block moved to end of file


def add_sortase_tag(sequence: str, position: str = "C-terminus") -> str:
    """
    Adds an LPXTG-family sortase recognition tag to a binder sequence,
    enabling reliable site-specific conjugation via Sortase A —
    independent of whether the sequence happens to contain LYS/CYS
    residues in accessible positions (as scored by score_conjugation_sites).

    This is a simple sequence-level addition (5-6 extra residues), not a
    structural redesign. LPETGG is the most commonly used variant.

    position: "C-terminus" (most common, matches standard nanobody/scFv
              sortagging protocols) or "N-terminus".
    """
    tag = "LPETGG"
    if position == "C-terminus":
        return sequence + tag
    elif position == "N-terminus":
        return tag + sequence
    else:
        raise ValueError("position must be \'C-terminus\' or \'N-terminus\'")


def check_terminus_accessibility(
    pdb_path: str,
    binder_chain: str = "B",
    target_chain: str = "A",
    min_distance_angstrom: float = 15.0,
) -> dict:
    """
    Checks whether the binder's N-terminus and C-terminus are physically
    accessible (far from the target interface) before relying on either
    end for Sortase-tag-based conjugation.

    Adding an LPETGG tag as text is not enough — if that terminus is
    buried against the target surface in the actual 3D structure, the
    Sortase A enzyme won't be able to reach it in practice.

    Returns distances for both termini and recommends which one (if any)
    is safe to use for the tag, or whether a redesign is needed if both
    termini are too close to the target.
    """
    from Bio.PDB import PDBParser
    import numpy as np

    parser = PDBParser(QUIET=True)
    result = {
        "n_terminus_distance": None,
        "c_terminus_distance": None,
        "recommended_terminus": None,
        "error": None,
    }

    try:
        structure = parser.get_structure("s", pdb_path)
        model = structure[0]

        binder_residues = list(model[binder_chain].get_residues())
        target_atoms = [atom.coord for atom in model[target_chain].get_atoms()]
        target_atoms = np.array(target_atoms)

        def min_dist_to_target(residue):
            res_coords = np.array([atom.coord for atom in residue.get_atoms()])
            dists = np.linalg.norm(target_atoms[:, None, :] - res_coords[None, :, :], axis=2)
            return dists.min()

        n_term_res = binder_residues[0]
        c_term_res = binder_residues[-1]

        n_dist = min_dist_to_target(n_term_res)
        c_dist = min_dist_to_target(c_term_res)

        result["n_terminus_distance"] = float(n_dist)
        result["c_terminus_distance"] = float(c_dist)

        if c_dist >= min_distance_angstrom and c_dist >= n_dist:
            result["recommended_terminus"] = "C-terminus"
        elif n_dist >= min_distance_angstrom:
            result["recommended_terminus"] = "N-terminus"
        else:
            result["recommended_terminus"] = None  # both termini too close to interface — manual review required

    except Exception as e:
        result["error"] = str(e)

    return result


def add_sortase_tag_v2(sequence: str, position: str = "C-terminus", linker_length: int = 10) -> str:
    """
    Adds a flexible GGGGS-repeat linker + LPETGG sortase tag.
    The linker gives the tag enough flexibility to reach the Sortase A
    enzyme even when the terminus itself sits close to the target
    interface (addresses the "needs review" cases where termini were
    too close to be safely used as-is).
    """
    linker_unit = "GGGGS"
    repeats = (linker_length // 5) + 1
    linker = (linker_unit * repeats)[:linker_length]
    tag = "LPETGG"
    if position == "C-terminus":
        return sequence + linker + tag
    else:
        return tag + linker + sequence


def esm2_developability_score(sequence: str, model=None, alphabet=None, batch_converter=None) -> dict:
    """
    Scores a binder sequence for developability using ESM2 protein language
    model embeddings combined with a logistic regression classifier trained
    on 2,505 experimentally characterized designs from Proteinbase.

    Achieves AUC-ROC = 0.756 on random split, 0.523 by-target split.
    Does not generalise to unseen targets — use as a soft signal only.

    This is a heavier-weight, GPU-preferred scoring layer intended to
    complement (not replace) the fast physicochemical checks
    (score_gravy, score_net_charge, score_conjugation_sites) used
    elsewhere in the pipeline for quick, interpretable filtering.

    Requires: pip install fair-esm
    model, alphabet, batch_converter should be loaded once via:
        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval()
    and passed in to avoid reloading the model on every call.

    Returns {"embedding": list, "note": str} — the embedding itself,
    ready to be scored by a separately trained/loaded classifier
    (this function does not ship the trained classifier weights).
    """
    import torch

    if model is None or alphabet is None or batch_converter is None:
        return {"embedding": None, "note": "ESM2 model not loaded — pass model, alphabet, batch_converter"}

    try:
        seq_trunc = sequence[:1022]
        data = [("query", seq_trunc)]
        _, _, batch_tokens = batch_converter(data)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[12])
            embedding = results["representations"][12][0, 1:len(seq_trunc)+1].mean(0)
        return {"embedding": embedding.tolist(), "note": "ok"}
    except Exception as e:
        return {"embedding": None, "note": f"error: {e}"}


def score_developability_esm2(sequence: str, model=None, alphabet=None, batch_converter=None,
                                classifier_path: str = "/content/drive/MyDrive/Oryqeva/results/esm2_binder_classifier.pkl") -> dict:
    """
    Full ESM2-based developability score: extracts embedding, then scores
    it with the trained logistic regression classifier in one call.

    Returns {"oryqeva_score": float (0-1) or None, "note": str}
    oryqeva_score is the predicted probability of experimental success,
    based on a model trained on 2,379 real Proteinbase designs
    (AUC-ROC = 0.756 random split, 0.523 by-target split).

    Requires the ESM2 model/alphabet/batch_converter to be pre-loaded
    (see esm2_developability_score docstring for loading code), and the
    trained classifier .pkl to exist at classifier_path.
    """
    import torch
    import pickle
    import numpy as np
    import os

    if model is None or alphabet is None or batch_converter is None:
        return {"oryqeva_score": None, "note": "ESM2 model not loaded"}

    if not os.path.exists(classifier_path):
        return {"oryqeva_score": None, "note": f"classifier not found at {classifier_path}"}

    try:
        seq_trunc = sequence[:1022]
        data = [("query", seq_trunc)]
        _, _, batch_tokens = batch_converter(data)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[12])
            embedding = results["representations"][12][0, 1:len(seq_trunc)+1].mean(0).numpy()

        with open(classifier_path, "rb") as f:
            classifier = pickle.load(f)

        score = classifier.predict_proba(embedding.reshape(1, -1))[0, 1]
        return {"oryqeva_score": float(score), "note": "ok"}
    except Exception as e:
        return {"oryqeva_score": None, "note": f"error: {e}"}


def engineer_single_conjugation_site(
    pdb_path: str,
    binder_chain: str = "B",
    target_chain: str = "A",
    min_distance_from_interface: float = 12.0,
    exclude_positions: list = None,
) -> dict:
    """
    Proposes ONE engineered cysteine mutation site for site-specific
    conjugation, instead of relying on whichever LYS/CYS residues happen
    to already exist on the binder surface.
    """
    from Bio.PDB import PDBParser
    import numpy as np

    parser = PDBParser(QUIET=True)
    result = {
        "position": None,
        "original_residue": None,
        "distance_to_interface": None,
        "error": None,
    }

    exclude_positions = exclude_positions or []

    try:
        structure = parser.get_structure("s", pdb_path)
        model = structure[0]

        binder_residues = list(model[binder_chain].get_residues())
        target_atoms = np.array([atom.coord for atom in model[target_chain].get_atoms()])

        centroid = np.mean(
            [atom.coord for res in binder_residues for atom in res.get_atoms()], axis=0
        )

        candidates = []
        for i, res in enumerate(binder_residues):
            resnum = res.id[1]
            resname = res.resname

            if resnum in exclude_positions:
                continue
            if resname in ("CYS", "PRO", "GLY"):
                continue

            res_coords = np.array([atom.coord for atom in res.get_atoms()])
            dists_to_target = np.linalg.norm(
                target_atoms[:, None, :] - res_coords[None, :, :], axis=2
            )
            min_dist_to_target = dists_to_target.min()

            dist_from_centroid = np.linalg.norm(res["CA"].coord - centroid)

            if min_dist_to_target >= min_distance_from_interface:
                candidates.append({
                    "resnum": resnum,
                    "resname": resname,
                    "dist_to_target": float(min_dist_to_target),
                    "dist_from_centroid": float(dist_from_centroid),
                })

        if not candidates:
            result["error"] = (
                f"No residue on chain {binder_chain} is >= "
                f"{min_distance_from_interface} Å from the target interface"
            )
            return result

        best = max(candidates, key=lambda c: (c["dist_to_target"], c["dist_from_centroid"]))

        result["position"] = best["resnum"]
        result["original_residue"] = best["resname"]
        result["distance_to_interface"] = best["dist_to_target"]

    except Exception as e:
        result["error"] = str(e)

    return result


def apply_cysteine_mutation(sequence: str, position: int) -> str:
    """
    Applies the point mutation proposed by engineer_single_conjugation_site()
    to a sequence, substituting the residue at position (1-indexed) with Cysteine.
    """
    idx = position - 1
    if idx < 0 or idx >= len(sequence):
        raise ValueError(f"Position {position} is out of range for a sequence of length {len(sequence)}")
    return sequence[:idx] + "C" + sequence[idx+1:]


def get_final_conjugation_strategy(
    pdb_path: str,
    sequence: str,
    binder_chain: str = "B",
    target_chain: str = "A",
    esm2_model=None,
    esm2_alphabet=None,
    esm2_batch_converter=None,
) -> dict:
    """
    Single entry point that returns everything a client needs to move a
    design forward: the chosen conjugation strategy, the final ready-to-
    order sequence, AND the Oryqeva developability score.

    Combines PROPKA natural-site detection, terminus-based Sortase
    tagging, engineered single-cysteine fallback (see
    engineer_single_conjugation_site), and the ESM2-based developability
    classifier (see score_developability_esm2) in one call.

    Decision order for conjugation strategy:
      1. Natural LYS/CYS sites (PROPKA) AND a safe terminus for Sortase
         tagging -> "natural_plus_tag" (highest confidence).
      2. Safe terminus only -> "tag_only".
      3. Neither -> attempt engineer_single_conjugation_site; if it finds
         a safe position -> "engineered_cysteine". If that also fails
         (binder too small/compact) -> "needs_manual_review".

    The Oryqeva-Score (probability of experimental success, 0-1) is
    computed on final_sequence — i.e. AFTER the tag/mutation is applied,
    since that is the sequence that would actually be ordered/expressed.
    Pass esm2_model/esm2_alphabet/esm2_batch_converter (pre-loaded once
    per session, see score_developability_esm2 docstring) to get this
    score; omit them to skip it (oryqeva_score will be None).
    """
    conj = score_conjugation_sites(pdb_path, binder_chain=binder_chain)
    termini = check_terminus_accessibility(pdb_path, binder_chain=binder_chain, target_chain=target_chain)

    has_natural = conj is not None and conj.get("valid_conjugation_sites", 0) > 0
    terminus_ok = termini.get("recommended_terminus") is not None

    if has_natural and terminus_ok:
        strategy = "natural_plus_tag"
        final_sequence = add_sortase_tag(sequence, position=termini["recommended_terminus"])
    elif terminus_ok:
        strategy = "tag_only"
        final_sequence = add_sortase_tag(sequence, position=termini["recommended_terminus"])
    else:
        engineered = engineer_single_conjugation_site(pdb_path, binder_chain=binder_chain, target_chain=target_chain)
        if engineered.get("position") is not None:
            strategy = "engineered_cysteine"
            final_sequence = apply_cysteine_mutation(sequence, engineered["position"])
        else:
            strategy = "needs_manual_review"
            final_sequence = sequence

    esm2_result = score_developability_esm2(
        final_sequence,
        model=esm2_model, alphabet=esm2_alphabet, batch_converter=esm2_batch_converter,
    )

    return {
        "strategy": strategy,
        "final_sequence": final_sequence,
        "oryqeva_score": esm2_result.get("oryqeva_score"),
        "propka_result": conj,
        "termini_result": termini,
    }


# ── BindCraft backend (validation, settings builder, environment preflight, run + output parsing) ──
BACKEND_VERSION = "0.1.0"
DEFAULT_BINDCRAFT_ROOT = "/content/BindCraft"

FILTER_FILES = {
    "default": "default_filters.json",
    "relaxed": "relaxed_filters.json",
    "peptide": "peptide_filters.json",
    "peptide_relaxed": "peptide_relaxed_filters.json",
    "none": "no_filters.json",
}

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


# -- failure classification ------------------------------------------------
class OryqevaError(RuntimeError):
    category = "error"

    def __str__(self):
        return f"[{self.category}] {super().__str__()}"


class EnvironmentFailure(OryqevaError):
    category = "environment"


class InputFailure(OryqevaError):
    category = "input"


class SettingsFailure(OryqevaError):
    category = "settings"


class BindCraftFailure(OryqevaError):
    category = "bindcraft"


class OutputFailure(OryqevaError):
    category = "output"


# -- result objects --------------------------------------------------------
@dataclass
class BindCraftDesign:
    design_id: str
    pdb_path: str
    sequence: Optional[str]
    length: Optional[int]
    metrics: dict = field(default_factory=dict)


@dataclass
class BindCraftRun:
    design_path: str
    settings_path: str
    filters_path: str
    advanced_path: str
    log_path: str
    runtime_seconds: float
    designs: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def pdb_paths(self):
        return [d.pdb_path for d in self.designs]


# -- input validation ------------------------------------------------------
_HOTSPOT_TOKEN = re.compile(r"^([A-Za-z]?)(\d+)(?:-(\d+))?$")


def parse_hotspots(text: str, default_chain: str) -> list:
    """
    Parses BindCraft hotspot syntax: "56,115,123", "1,2-10", "A1-10,B20"
    or a whole chain such as "A". Returns (chain, start, end) tuples;
    end is None for a single residue, start is None for a whole chain.
    """
    items = []
    for tok in [t.strip() for t in (text or "").split(",") if t.strip()]:
        if re.fullmatch(r"[A-Za-z]", tok):
            items.append((tok, None, None))
            continue
        m = _HOTSPOT_TOKEN.match(tok)
        if not m:
            raise InputFailure(
                f"Invalid hotspot '{tok}'. Use BindCraft syntax such as "
                f"'56,115', '1,2-10' or 'A1-10,B20' (no colons or residue letters)."
            )
        chain = m.group(1) or default_chain
        start = int(m.group(2))
        end = int(m.group(3)) if m.group(3) else None
        if end is not None and end < start:
            raise InputFailure(f"Invalid hotspot range '{tok}': end is before start.")
        items.append((chain, start, end))
    return items


def validate_target(pdb_path: str, chains: str, hotspots: str = "") -> dict:
    """Checks the target PDB, its chains and the hotspot residues."""
    if not os.path.exists(pdb_path):
        raise InputFailure(f"Target PDB not found: {pdb_path}")
    try:
        from Bio.PDB import PDBParser
        structure = PDBParser(QUIET=True).get_structure("target", pdb_path)
        model = structure[0]
    except Exception as e:
        raise InputFailure(f"Target PDB could not be parsed: {e}")

    chain_ids = [c.id for c in model.get_chains()]
    wanted = [c.strip() for c in chains.split(",") if c.strip()]
    if not wanted:
        raise InputFailure("No target chain given.")
    info = {"chains": {}, "warnings": []}
    for ch in wanted:
        if ch not in chain_ids:
            raise InputFailure(f"Chain '{ch}' not found in {pdb_path}. Available: {chain_ids}")
        resnums = {r.id[1] for r in model[ch].get_residues()
                   if r.id[0] == " " and "CA" in r}
        if not resnums:
            raise InputFailure(f"Chain '{ch}' has no standard residues with CA atoms.")
        info["chains"][ch] = resnums

    n_res = sum(len(v) for v in info["chains"].values())
    info["n_residues"] = n_res
    if n_res > 400:
        info["warnings"].append(
            f"Target has {n_res} residues. BindCraft needs a lot of GPU memory for "
            f"large targets; trim the PDB to the region you want to bind."
        )

    for chain, start, end in parse_hotspots(hotspots, wanted[0]):
        if chain not in info["chains"]:
            raise InputFailure(f"Hotspot refers to chain '{chain}', which is not a target chain.")
        if start is None:
            continue
        have = info["chains"][chain]
        hi = end if end is not None else start
        if not any(r in have for r in range(start, hi + 1)):
            raise InputFailure(
                f"Invalid hotspot: chain {chain} residues {start}"
                f"{'-' + str(end) if end else ''} were not found in the target structure."
            )
        if end is None and start not in have:
            raise InputFailure(
                f"Invalid hotspot: chain {chain} residue {start} was not found in the target structure."
            )
    return info


# -- settings --------------------------------------------------------------
def prepare_target_pdb(
    target_input_path: str,
    chain_id: str = "A",
    cache_dir: Optional[str] = None,
    timeout_seconds: int = 120,
) -> str:
    """
    Accepts either a .pdb file (returned unchanged) or a .fasta/.fa file
    (single sequence, structure predicted with the ESMFold API and saved
    as a PDB). Returns the path to a PDB file ready for validate_target()
    and build_bindcraft_settings().

    ESMFold is used only to get a starting structure for a target described
    by sequence alone — it is not part of binder design itself.
    """
    ext = os.path.splitext(target_input_path)[1].lower()

    if ext == ".pdb":
        if not os.path.exists(target_input_path):
            raise InputFailure(f"Target PDB not found: {target_input_path}")
        return target_input_path

    if ext not in (".fasta", ".fa"):
        raise InputFailure(
            f"Unsupported target file type '{ext}'. Provide a .pdb, or a "
            f".fasta/.fa file with one sequence to fold with ESMFold."
        )

    if not os.path.exists(target_input_path):
        raise InputFailure(f"Target FASTA not found: {target_input_path}")

    with open(target_input_path) as f:
        lines = [l.rstrip() for l in f if l.strip()]
    records = []
    for line in lines:
        if line.startswith(">"):
            records.append("")
        elif records:
            records[-1] += line.strip()
        else:
            raise InputFailure(f"'{target_input_path}' is not valid FASTA (no header line).")
    records = [r for r in records if r]
    if not records:
        raise InputFailure(f"No sequence found in '{target_input_path}'.")
    if len(records) > 1:
        raise InputFailure(
            f"'{target_input_path}' has {len(records)} sequences. "
            f"prepare_target_pdb only folds a single-chain target; split multi-chain "
            f"targets and provide a PDB instead."
        )
    seq = records[0].upper()
    if not set(seq).issubset(set("ACDEFGHIKLMNPQRSTVWY")):
        raise InputFailure(f"Sequence in '{target_input_path}' has non-standard amino acids.")
    if len(seq) > 400:
        raise InputFailure(
            f"Sequence is {len(seq)} aa. ESMFold folding here is limited to 400 aa; "
            f"provide an experimental/AlphaFold PDB for larger targets instead."
        )

    import requests
    try:
        resp = requests.post(
            "https://api.esmatlas.com/foldSequence/v1/pdb/",
            data=seq, timeout=timeout_seconds,
        )
    except requests.RequestException as e:
        raise InputFailure(f"ESMFold request failed: {e}")
    if resp.status_code != 200 or not resp.text.strip().startswith(("HEADER", "ATOM")):
        raise InputFailure(f"ESMFold folding failed (status {resp.status_code}): {resp.text[:200]}")

    pdb_text = resp.text
    if chain_id != "A":
        pdb_text = re.sub(r"^(ATOM  .{16})A", lambda m: m.group(1) + chain_id, pdb_text, flags=re.M)
        pdb_text = re.sub(r"^(TER   .{16})A", lambda m: m.group(1) + chain_id, pdb_text, flags=re.M)

    out_dir = cache_dir or os.path.join(os.path.dirname(target_input_path) or ".", "esmfold_cache")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(target_input_path))[0]
    out_path = os.path.join(out_dir, f"{stem}_esmfold.pdb")
    with open(out_path, "w") as f:
        f.write(pdb_text)
    return out_path


def detect_gpu_memory_mb() -> Optional[int]:
    """
    Detects free GPU memory in MB via `nvidia-smi`. Returns None if nvidia-smi
    is unavailable or fails to parse, so callers can fall back to a
    conservative default (e.g. for CI/testing environments without a GPU).

    This replaces the previous hardcoded 15000 MB (Colab T4) default in
    check_target_size() / auto_trim_target(), so the memory guard works
    correctly on any machine -- Colab, a local 8 GB laptop GPU, or a
    multi-GPU workstation (the minimum across GPUs is used, conservatively).
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        values = [int(v.strip()) for v in result.stdout.strip().split("\n") if v.strip()]
        return min(values) if values else None
    except Exception:
        return None


def estimate_bindcraft_memory_mb(target_pdb_path: str, target_chain: str, max_binder_length: int) -> dict:
    """
    Rough VRAM estimate for a BindCraft/ColabDesign hallucination run, based
    on total sequence length (target + binder) passed through AlphaFold2.
    AF2/ColabDesign memory scales roughly with the square of sequence length
    (due to the pair representation), so this is quadratic, not linear.
    """
    from Bio.PDB import PDBParser
    model = PDBParser(QUIET=True).get_structure("t", target_pdb_path)[0]
    chain_ids = [c.strip() for c in target_chain.split(",") if c.strip()]
    target_len = sum(1 for cid in chain_ids for r in model[cid] if r.id[0] == " ")
    total_len = target_len + max_binder_length
    est_mb = round((total_len ** 2) * 0.043)
    return {"target_length": target_len, "max_binder_length": max_binder_length,
            "total_length": total_len, "estimated_mb": est_mb, "estimated_gb": round(est_mb / 1024, 1)}


def check_target_size(target_pdb_path: str, target_chain: str, max_binder_length: int,
                       available_gpu_mb: Optional[int] = None) -> dict:
    """
    Warns before a BindCraft run if the target is large enough to likely
    exhaust GPU memory, and suggests a fix. Prevents wasting GPU time on a
    run that would fail with RESOURCE_EXHAUSTED partway through.
    """
    est = estimate_bindcraft_memory_mb(target_pdb_path, target_chain, max_binder_length)
    budget_mb = available_gpu_mb or detect_gpu_memory_mb() or 15000
    est["gpu_budget_mb"] = budget_mb
    est["fits"] = est["estimated_mb"] < budget_mb * 0.85
    if not est["fits"]:
        import math
        max_total_len = math.isqrt(int(budget_mb * 0.85 / 0.043))
        est["suggested_max_target_length"] = max(max_total_len - max_binder_length, 50)
        est["message"] = (
            f"Target chain '{target_chain}' is {est['target_length']} residues. "
            f"Combined with a {max_binder_length}-residue binder, this is estimated "
            f"to need ~{est['estimated_gb']} GB of GPU memory, which is likely to "
            f"exceed the available ~{budget_mb/1024:.0f} GB and fail with an "
            f"out-of-memory error partway through the run. "
            f"BindCraft works on a defined epitope, not a whole receptor -- trim "
            f"the target PDB to the region around your hotspot residues "
            f"(roughly {est['suggested_max_target_length']} residues or fewer) "
            f"before running."
        )
    else:
        est["message"] = (
            f"Target chain '{target_chain}' is {est['target_length']} residues "
            f"(~{est['estimated_gb']} GB estimated) -- should fit in "
            f"~{budget_mb/1024:.0f} GB of GPU memory."
        )
    return est


def auto_trim_target(
    target_pdb_path: str,
    target_chain: str,
    hotspot_residues: str,
    max_binder_length: int,
    available_gpu_mb: Optional[int] = None,
    padding: int = 40,
    output_dir: Optional[str] = None,
) -> dict:
    """
    If the target is too large to fit in GPU memory (per check_target_size),
    trims it down to a contiguous window centred on the hotspot residues
    (or on the chain midpoint, if no hotspots were given), with `padding`
    extra residues on each side for structural context.

    This mirrors BindCraft's own documented guidance: target a functional
    epitope/domain rather than a whole receptor. The person is always told
    exactly what was trimmed and why -- this never trims silently.

    Returns a dict: {"pdb_path": ..., "trimmed": bool, "message": str,
    "kept_range": (start, end) or None}.
    """
    from Bio.PDB import PDBParser, PDBIO, Select
    import os as _os

    size_check = check_target_size(target_pdb_path, target_chain, max_binder_length, available_gpu_mb)
    if size_check["fits"]:
        return {"pdb_path": target_pdb_path, "trimmed": False,
                "message": size_check["message"], "kept_range": None}

    chain_ids = [c.strip() for c in target_chain.split(",") if c.strip()]
    if len(chain_ids) != 1:
        raise InputFailure(
            f"Target is too large ({size_check['estimated_gb']} GB estimated) and "
            f"spans multiple chains ('{target_chain}'), so it cannot be auto-trimmed "
            f"to a single window. Trim the target PDB to your epitope of interest "
            f"manually before running."
        )
    chain_id = chain_ids[0]

    structure = PDBParser(QUIET=True).get_structure("t", target_pdb_path)
    model = structure[0]
    if chain_id not in [c.id for c in model.get_chains()]:
        raise InputFailure(f"Chain '{chain_id}' not found in {target_pdb_path}.")

    resnums = sorted(r.id[1] for r in model[chain_id] if r.id[0] == " ")
    if not resnums:
        raise InputFailure(f"Chain '{chain_id}' has no standard residues.")

    hotspot_positions = []
    for chain, start, end in parse_hotspots(hotspot_residues, chain_id):
        if chain != chain_id or start is None:
            continue
        hotspot_positions.append(start)
        if end is not None:
            hotspot_positions.append(end)

    if hotspot_positions:
        center_lo, center_hi = min(hotspot_positions), max(hotspot_positions)
    else:
        mid = resnums[len(resnums) // 2]
        center_lo = center_hi = mid

    max_window = max(size_check["suggested_max_target_length"], 50)

    window_start = center_lo - padding
    window_end = center_hi + padding
    if window_end - window_start + 1 > max_window:
        mid = (center_lo + center_hi) // 2
        window_start = mid - max_window // 2
        window_end = window_start + max_window - 1

    window_start = max(window_start, resnums[0])
    window_end = min(window_end, resnums[-1])

    class _RangeSelect(Select):
        def __init__(self, chain, lo, hi):
            self.chain, self.lo, self.hi = chain, lo, hi
        def accept_chain(self, c):
            return c.id == self.chain
        def accept_residue(self, r):
            return r.id[0] == " " and self.lo <= r.id[1] <= self.hi

    out_dir = output_dir or _os.path.join(_os.path.dirname(target_pdb_path) or ".", "auto_trimmed")
    _os.makedirs(out_dir, exist_ok=True)
    stem = _os.path.splitext(_os.path.basename(target_pdb_path))[0]
    out_path = _os.path.join(out_dir, f"{stem}_trim{window_start}-{window_end}.pdb")

    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path, _RangeSelect(chain_id, window_start, window_end))

    kept_n = sum(1 for rn in resnums if window_start <= rn <= window_end)
    hotspot_note = (
        f"centred on hotspot residues {min(hotspot_positions)}-{max(hotspot_positions)}"
        if hotspot_positions else "centred on the chain midpoint (no hotspots given)"
    )
    message = (
        f"Target chain '{chain_id}' was {size_check['target_length']} residues "
        f"(~{size_check['estimated_gb']} GB estimated), too large for the available "
        f"GPU memory. Auto-trimmed to residues {window_start}-{window_end} "
        f"({kept_n} residues, {hotspot_note} with {padding}-residue padding). "
        f"Saved to: {out_path}. Review the trimmed structure before a large run -- "
        f"this is a heuristic window, not a verified functional domain boundary."
    )

    return {"pdb_path": out_path, "trimmed": True, "message": message,
            "kept_range": (window_start, window_end)}


def build_bindcraft_settings(
    target_pdb_path: str,
    target_name: str,
    target_chain: str = "A",
    hotspot_residues: str = "",
    binder_lengths: Optional[list] = None,
    num_designs: int = 5,
    design_root: Optional[str] = None,
) -> tuple:
    """
    Writes the BindCraft target settings JSON for one target and returns
    (settings_path, design_path, validation_info). Keys follow the schema of
    BindCraft's own target settings files.
    """
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", target_name or ""):
        raise SettingsFailure("target_name may only contain letters, digits, '_' and '-'.")
    lengths = list(binder_lengths or [50, 120])
    if len(lengths) != 2 or not all(isinstance(x, int) for x in lengths) or not (5 <= lengths[0] <= lengths[1]):
        raise SettingsFailure(f"binder_lengths must be [min, max] integers, got {lengths}.")
    if not isinstance(num_designs, int) or num_designs < 1:
        raise SettingsFailure("num_designs must be a positive integer.")

    target_pdb_path = prepare_target_pdb(target_pdb_path, chain_id=target_chain)
    info = validate_target(target_pdb_path, target_chain, hotspot_residues)

    design_path = os.path.join(design_root or f"{DRIVE_ROOT}/BindCraft", target_name) + "/"
    os.makedirs(design_path, exist_ok=True)
    settings = {
        "design_path": design_path,
        "binder_name": target_name,
        "starting_pdb": target_pdb_path,
        "chains": target_chain,
        "target_hotspot_residues": (hotspot_residues or "").strip(),
        "lengths": lengths,
        "number_of_final_designs": num_designs,
    }
    settings_path = os.path.join(design_path, f"{target_name}.json")
    try:
        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=2)
    except OSError as e:
        raise SettingsFailure(f"Could not write settings file: {e}")
    return settings_path, design_path, info


def find_config(root: str, filename: str) -> str:
    """Locates a BindCraft filter/advanced JSON anywhere under the install."""
    hits = glob.glob(os.path.join(root, "**", filename), recursive=True)
    if not hits:
        available = sorted({os.path.basename(p) for p in
                            glob.glob(os.path.join(root, "settings_*", "*.json"))})
        raise SettingsFailure(f"Config '{filename}' not found under {root}. Available: {available[:40]}")
    return sorted(hits, key=len)[0]


def _advanced_with_overrides(advanced_path: str, overrides: dict, design_path: str) -> str:
    with open(advanced_path) as f:
        adv = json.load(f)
    unknown = [k for k in overrides if k not in adv]
    if unknown:
        raise SettingsFailure(f"Unknown advanced settings keys: {unknown}")
    adv.update(overrides)
    out = os.path.join(design_path, "oryqeva_advanced.json")
    with open(out, "w") as f:
        json.dump(adv, f, indent=2)
    return out


# -- environment -----------------------------------------------------------
_PROBE = r"""
import json, os, sys
r = {}
for m in ("pyrosetta", "colabdesign", "jax", "haiku"):
    try:
        __import__(m); r[m] = True
    except Exception as e:
        r[m] = False; r[m + "_error"] = str(e)[:200]
try:
    import jax; r["jax_backend"] = jax.default_backend()
except Exception:
    r["jax_backend"] = None
sys.path.insert(0, os.getcwd())
try:
    import functions; r["bindcraft_functions"] = True
except Exception as e:
    r["bindcraft_functions"] = False; r["bindcraft_functions_error"] = str(e)[:200]
print("ORYQEVA_PROBE" + json.dumps(r))
"""


def check_environment(bindcraft_root: Optional[str] = None, python_exe: Optional[str] = None) -> dict:
    """Probes BindCraft, its Python packages, the AF2 weights and the GPU."""
    root = bindcraft_root or DEFAULT_BINDCRAFT_ROOT
    py = python_exe or sys.executable
    report = {"python": py, "root": root,
              "bindcraft_script": os.path.exists(os.path.join(root, "bindcraft.py"))}
    params = os.path.join(root, "params")
    report["af2_params"] = os.path.isdir(params) and bool(os.listdir(params))
    if not report["bindcraft_script"]:
        return report
    try:
        p = subprocess.run([py, "-c", _PROBE], capture_output=True, text=True, cwd=root, timeout=300)
        line = [l for l in p.stdout.splitlines() if l.startswith("ORYQEVA_PROBE")]
        report.update(json.loads(line[0][len("ORYQEVA_PROBE"):]) if line else {"probe_error": p.stderr[-300:]})
    except Exception as e:
        report["probe_error"] = str(e)
    return report


def assert_environment(report: dict) -> None:
    problems = []
    if not report.get("bindcraft_script"):
        problems.append(f"bindcraft.py not found under {report.get('root')}")
    if not report.get("af2_params"):
        problems.append(f"AlphaFold2 weights missing at {report.get('root')}/params")
    for m in ("pyrosetta", "colabdesign", "jax", "haiku"):
        if report.get(m) is False:
            problems.append(f"{m} unavailable ({report.get(m + '_error', '')})")
    if report.get("bindcraft_functions") is False:
        problems.append(f"BindCraft 'functions' import failed ({report.get('bindcraft_functions_error', '')})")
    if "probe_error" in report:
        problems.append(f"environment probe failed: {report['probe_error']}")
    if report.get("jax_backend") not in (None, "gpu", "cuda"):
        problems.append(f"JAX is using '{report.get('jax_backend')}', not a GPU. Switch the runtime to a GPU.")
    if problems:
        raise EnvironmentFailure("BindCraft environment incomplete: " + "; ".join(problems))


# -- output parsing --------------------------------------------------------
def sequence_from_pdb(pdb_path: str, chain: str) -> Optional[str]:
    try:
        from Bio.PDB import PDBParser
        model = PDBParser(QUIET=True).get_structure("s", pdb_path)[0]
        return "".join(THREE_TO_ONE.get(r.resname, "X") for r in model[chain] if r.id[0] == " ")
    except Exception:
        return None


def _number(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def parse_bindcraft_output(design_path: str, binder_chain: str = "B") -> list:
    """
    Reads Accepted/*.pdb and the BindCraft stats CSV
    (final_design_stats.csv, else mpnn_design_stats.csv). Every CSV column is
    kept in `metrics`; the sequence comes from the CSV or, failing that, from
    the binder chain of the PDB.
    """
    pdbs = sorted(glob.glob(os.path.join(design_path, "Accepted", "*.pdb")))
    rows = {}
    for name in ("final_design_stats.csv", "mpnn_design_stats.csv"):
        path = os.path.join(design_path, name)
        if os.path.exists(path):
            try:
                with open(path, newline="") as f:
                    for row in csv.DictReader(f):
                        key = next((row[k] for k in row if k and k.strip().lower() == "design"), None)
                        if key:
                            rows[key.strip()] = {k: _number(v) for k, v in row.items() if k}
            except Exception as e:
                raise OutputFailure(f"Could not read {path}: {e}")
            if rows:
                break

    designs = []
    for pdb in pdbs:
        stem = os.path.splitext(os.path.basename(pdb))[0]
        match = rows.get(stem) or next(
            (v for k, v in rows.items() if stem.startswith(k + "_model") or stem == k), None)
        metrics = dict(match) if match else {}
        seq = None
        for k, v in metrics.items():
            if k.strip().lower() == "sequence" and isinstance(v, str):
                seq = v
        seq = seq or sequence_from_pdb(pdb, binder_chain)
        designs.append(BindCraftDesign(stem, pdb, seq, len(seq) if seq else None, metrics))
    return designs


def collect_designs(target_name: str, design_root: Optional[str] = None, binder_chain: str = "B") -> list:
    """Parses an already finished BindCraft run for this target."""
    path = os.path.join(design_root or f"{DRIVE_ROOT}/BindCraft", target_name)
    return parse_bindcraft_output(path, binder_chain)


# -- running ---------------------------------------------------------------
def _git_commit(root: str) -> Optional[str]:
    try:
        return subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def run_bindcraft(
    target_pdb_path: str,
    target_name: str,
    target_chain: str = "A",
    hotspot_residues: str = "",
    num_designs: int = 5,
    binder_lengths: Optional[list] = None,
    filter_profile: str = "default",
    advanced_profile: str = "default_4stage_multimer",
    advanced_overrides: Optional[dict] = None,
    bindcraft_root: Optional[str] = None,
    python_exe: Optional[str] = None,
    design_root: Optional[str] = None,
    timeout_hours: float = 11.0,
    preflight: bool = True,
    progress_every_seconds: int = 300,
    auto_trim: bool = True,
) -> BindCraftRun:
    """
    Validate inputs -> preflight environment -> write settings -> run
    `bindcraft.py --settings --filters --advanced` -> parse results.
    Re-running with the same target continues the existing BindCraft campaign.
    """
    root = bindcraft_root or DEFAULT_BINDCRAFT_ROOT
    py = python_exe or sys.executable
    if filter_profile not in FILTER_FILES:
        raise SettingsFailure(f"filter_profile must be one of {list(FILTER_FILES)}")

    max_binder_length = (binder_lengths or [50, 120])[1]
    trim_result = None
    if auto_trim:
        trim_result = auto_trim_target(
            target_pdb_path, target_chain, hotspot_residues, max_binder_length
        )
        if trim_result["trimmed"]:
            print(trim_result["message"])
            target_pdb_path = trim_result["pdb_path"]
    else:
        size_check = check_target_size(target_pdb_path, target_chain, max_binder_length)
        if not size_check["fits"]:
            raise InputFailure(size_check["message"])

    settings_path, design_path, info = build_bindcraft_settings(
        target_pdb_path, target_name, target_chain, hotspot_residues,
        binder_lengths, num_designs, design_root)

    if preflight:
        assert_environment(check_environment(root, py))

    filters_path = find_config(root, FILTER_FILES[filter_profile])
    advanced_path = find_config(root, advanced_profile + ".json")
    if advanced_overrides:
        advanced_path = _advanced_with_overrides(advanced_path, advanced_overrides, design_path)

    log_path = os.path.join(design_path, "oryqeva_run.log")
    provenance = {
        "oryqeva_backend_version": BACKEND_VERSION,
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "bindcraft_commit": _git_commit(root),
        "settings": json.load(open(settings_path)),
        "filters": os.path.basename(filters_path),
        "advanced": os.path.basename(advanced_path),
        "advanced_overrides": advanced_overrides or {},
        "auto_trim": trim_result,
    }
    with open(os.path.join(design_path, "oryqeva_provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2)

    cmd = [py, os.path.join(root, "bindcraft.py"), "--settings", settings_path,
           "--filters", filters_path, "--advanced", advanced_path]
    t0 = time.time()
    last = t0
    with open(log_path, "a") as log:
        log.write(f"\n=== {provenance['started']} {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=root)
        while proc.poll() is None:
            time.sleep(5)
            now = time.time()
            if now - t0 > timeout_hours * 3600:
                proc.kill()
                raise BindCraftFailure(f"Timed out after {timeout_hours} h. Re-run to continue; see {log_path}")
            if now - last >= progress_every_seconds:
                n = len(glob.glob(os.path.join(design_path, "Accepted", "*.pdb")))
                print(f"  {int((now - t0) / 60)} min | accepted so far: {n}")
                last = now
    if proc.returncode != 0:
        with open(log_path) as f:
            tail = f.read()[-1500:]
        raise BindCraftFailure(f"BindCraft exited with code {proc.returncode}. Log tail:\n{tail}")

    designs = parse_bindcraft_output(design_path)
    if not designs:
        raise OutputFailure(f"BindCraft finished but no accepted designs were found in {design_path}Accepted/")
    return BindCraftRun(design_path, settings_path, filters_path, advanced_path, log_path,
                        time.time() - t0, designs, info["warnings"])


def classify_v2(oryqeva_score: float, strategy: str) -> str:
    """
    Replaces the old classify() function, which used thresholds
    (GRAVY_P33/P67, CONJ_P33/P67) tuned on a single PD-L1 design and was
    inconsistent with the newer ESM2-based scoring used everywhere else
    in this pipeline.

    classify_v2 uses only the Oryqeva-Score (from
    score_developability_esm2 / get_final_conjugation_strategy), which is
    benchmarked on 2,379 real experimental outcomes (AUC-ROC=0.756 random split,
    0.523 by-target split — does not generalise to unseen targets), not a single legacy example. The old classify() and its GRAVY_P33/P67,
    CONJ_P33/P67 constants are kept in this file for backward
    compatibility but should be considered deprecated -- do not use them
    for new evaluation.

    Tiers (aligned with ORYQEVA_MASTER_RESULTS.csv used in prior batch runs):
      Tier 1 (score >= 0.65): high confidence
      Tier 2 (0.5 <= score < 0.65): moderate confidence
      Tier 3 (score < 0.5): low confidence
      "needs_manual_review" strategy always overrides to Tier 3, regardless
      of score, since no safe conjugation site could be identified at all.
    """
    if strategy == "needs_manual_review":
        return "Tier 3 - low confidence (no viable conjugation site found)"
    if oryqeva_score is None:
        return "PENDING (no ESM2 score available)"
    if oryqeva_score >= 0.65:
        return "Tier 1 - high confidence"
    elif oryqeva_score >= 0.5:
        return "Tier 2 - moderate confidence"
    else:
        return "Tier 3 - low confidence"


# ===========================================================================
# CONDITIONAL-BINDER TOOLKIT
#
# Post-processing for accepted BindCraft designs when the goal is a binder
# that is conditional (pH-dependent) and/or must cross-react between species:
#   find_interface_residues / target_contact_residues  - contact analysis
#   his_protonated_fraction                            - Henderson-Hasselbalch
#   propose_histidine_variants                         - pH-switch candidates
#   epitope_conservation                               - ortholog check
#   rank_designs / export_competition_csv              - ranked, validated CSV
#
# IMPORTANT: none of this predicts binding. The histidine step in particular
# is a heuristic that proposes variants; it does not tell you whether a
# variant binds at pH 6.5 or releases at pH 7.4. Only an experiment does.
# ===========================================================================

_CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")
_ACIDIC_ATOMS = {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2")}
DEFAULT_HIS_REPLACEABLE = "STNQAKR"


def his_protonated_fraction(pka: float, ph: float) -> float:
    """Henderson-Hasselbalch: fraction of a histidine side chain that is protonated."""
    return 1.0 / (1.0 + 10 ** (ph - pka))


def _standard_residues(chain):
    return [r for r in chain if r.id[0] == " "]


def _load_two_chains(pdb_path: str, binder_chain: str, target_chain: str):
    from Bio.PDB import PDBParser
    try:
        model = PDBParser(QUIET=True).get_structure("s", pdb_path)[0]
    except Exception as e:
        raise InputFailure(f"Could not read {pdb_path}: {e}")
    chains = {c.id for c in model}
    for cid in (binder_chain, target_chain):
        if cid not in chains:
            raise InputFailure(f"Chain '{cid}' not found in {pdb_path} (chains present: {sorted(chains)}).")
    return model[binder_chain], model[target_chain]


def find_interface_residues(pdb_path: str, binder_chain: str = "B", target_chain: str = "A",
                            contact_cutoff: float = 5.0) -> list:
    """
    Binder residues with any heavy atom within `contact_cutoff` A of the target.

    Each entry: position (1-based index in the binder sequence), resnum (PDB
    number), aa, n_contacts (target atoms in range), acid_distance (A from the
    nearest side-chain atom of this residue to a target Asp/Glu carboxylate
    oxygen, None if the target has none).
    """
    from Bio.PDB import NeighborSearch
    binder, target = _load_two_chains(pdb_path, binder_chain, target_chain)
    target_atoms = [a for r in _standard_residues(target) for a in r]
    acid_atoms = [a for r in _standard_residues(target) for a in r
                  if (r.resname, a.get_id()) in _ACIDIC_ATOMS]
    if not target_atoms:
        raise InputFailure(f"Target chain '{target_chain}' has no standard residues.")
    ns = NeighborSearch(target_atoms)

    out = []
    for pos, res in enumerate(_standard_residues(binder), start=1):
        n_contacts = 0
        for atom in res:
            n_contacts += len(ns.search(atom.coord, contact_cutoff))
        if n_contacts == 0:
            continue
        side = [a for a in res if a.get_id() not in ("N", "C", "O")] or list(res)
        # for Gly this falls back to CA; for others it includes CA, which is fine
        acid_dist = None
        if acid_atoms:
            acid_dist = min(float(((s.coord - o.coord) ** 2).sum() ** 0.5)
                            for s in side for o in acid_atoms)
        out.append({"position": pos, "resnum": res.id[1],
                    "aa": THREE_TO_ONE.get(res.resname, "X"),
                    "n_contacts": n_contacts, "acid_distance": acid_dist})
    return out


def target_contact_residues(pdb_path: str, binder_chain: str = "B", target_chain: str = "A",
                            contact_cutoff: float = 5.0) -> list:
    """Target residues (resnum, aa) within `contact_cutoff` A of the binder."""
    from Bio.PDB import NeighborSearch
    binder, target = _load_two_chains(pdb_path, binder_chain, target_chain)
    binder_atoms = [a for r in _standard_residues(binder) for a in r]
    if not binder_atoms:
        raise InputFailure(f"Binder chain '{binder_chain}' has no standard residues.")
    ns = NeighborSearch(binder_atoms)
    hits = []
    for res in _standard_residues(target):
        if any(ns.search(a.coord, contact_cutoff) for a in res):
            hits.append((res.id[1], THREE_TO_ONE.get(res.resname, "X")))
    return hits


def propose_histidine_variants(pdb_path: str, binder_chain: str = "B", target_chain: str = "A",
                               sequence: Optional[str] = None, n_his_options=(2, 3, 4),
                               contact_cutoff: float = 5.0, acid_cutoff: float = 7.0,
                               replaceable: str = DEFAULT_HIS_REPLACEABLE,
                               assumed_his_pka: float = 6.5,
                               ph_low: float = 6.5, ph_high: float = 7.4) -> dict:
    """
    Proposes histidine-enriched variants of a binder, aimed at binding that is
    stronger at low pH than at neutral pH.

    Rationale: a histidine is more often protonated (positively charged) at
    pH 6.5 than at 7.4, so a His placed next to a target Asp/Glu can form a
    charge interaction that exists mainly at low pH. Sites are ranked by
    distance to the nearest target carboxylate; only interface residues whose
    amino acid is in `replaceable` are considered, and Cys/Pro/Gly/His/Trp
    and hydrophobic core-like residues are left alone by default.

    LIMITS (read before using): this is a heuristic. The pH window is narrow
    (6.5 vs 7.4), so even an ideal His only changes its protonated fraction
    modestly -- see `protonation` in the result. Nothing here estimates
    binding affinity at either pH, and a substitution can just as easily
    weaken binding at both. Treat every variant as a hypothesis to test.
    """
    interface = find_interface_residues(pdb_path, binder_chain, target_chain, contact_cutoff)
    binder, _ = _load_two_chains(pdb_path, binder_chain, target_chain)
    pdb_seq = "".join(THREE_TO_ONE.get(r.resname, "X") for r in _standard_residues(binder))
    if sequence is not None and sequence != pdb_seq:
        raise InputFailure(
            "Provided sequence does not match the binder chain in the PDB "
            f"(sequence length {len(sequence)}, PDB chain length {len(pdb_seq)}). "
            "Pass the sequence of the same design, or omit `sequence`.")
    seq = pdb_seq

    warnings = []
    if not interface:
        raise InputFailure("No binder residues are within contact range of the target; "
                           "is this a complex structure with both chains placed together?")
    if not any(s["acid_distance"] is not None for s in interface):
        warnings.append("Target has no Asp/Glu near the interface; sites are ranked by contacts only.")

    candidates = [s for s in interface if s["aa"] in replaceable]
    near = [s for s in candidates if s["acid_distance"] is not None and s["acid_distance"] <= acid_cutoff]
    pool = near if near else candidates
    if near:
        pool = sorted(near, key=lambda s: (s["acid_distance"], s["n_contacts"]))
    else:
        warnings.append(f"No replaceable interface residue within {acid_cutoff} A of a target Asp/Glu; "
                        "falling back to all replaceable interface residues (weaker rationale).")
        pool = sorted(candidates, key=lambda s: s["n_contacts"])
    if not pool:
        raise InputFailure(f"No interface residue of type [{replaceable}] to replace with histidine.")

    variants = []
    for n in sorted(set(n_his_options)):
        if n > len(pool):
            warnings.append(f"Only {len(pool)} candidate sites; skipping the {n}-His variant.")
            continue
        sites = pool[:n]
        chars = list(seq)
        muts = []
        for s in sorted(sites, key=lambda s: s["position"]):
            muts.append(f"{s['aa']}{s['position']}H")
            chars[s["position"] - 1] = "H"
        variants.append({"variant_id": f"his{n}", "n_his_added": n, "mutations": muts,
                         "sequence": "".join(chars)})

    return {
        "parent_sequence": seq,
        "variants": variants,
        "protonation": {
            "assumed_his_pka": assumed_his_pka,
            f"protonated_at_pH_{ph_low}": round(his_protonated_fraction(assumed_his_pka, ph_low), 3),
            f"protonated_at_pH_{ph_high}": round(his_protonated_fraction(assumed_his_pka, ph_high), 3),
        },
        "warnings": warnings,
        "caveat": "Heuristic proposal only; not validated and not a binding prediction.",
    }


def epitope_conservation(target_residues: list, ortholog_seq: str, epitope_resnums: list,
                         ortholog_label: str = "ortholog") -> dict:
    """
    Checks whether epitope residues are conserved in another species.

    target_residues: [(resnum, aa), ...] for the target chain as it appears in
      the PDB (use target_contact_residues' companion, or build it from the
      chain). ortholog_seq: the other species' sequence (full or domain).
    epitope_resnums: PDB residue numbers to check (e.g. the target residues
      the binder contacts).

    Uses a local alignment, so a trimmed target aligns to a longer ortholog.
    "similar" means BLOSUM62 > 0. Conservation is necessary, not sufficient,
    for cross-reactivity: identical residues can still sit in a different
    local structure.
    """
    from Bio.Align import PairwiseAligner, substitution_matrices
    if not target_residues:
        raise InputFailure("target_residues is empty.")
    ortholog_seq = "".join(ortholog_seq.split()).upper()
    if not ortholog_seq or set(ortholog_seq) - _CANONICAL_AA:
        raise InputFailure("ortholog_seq must contain only the 20 standard amino acids.")

    target_seq = "".join(aa for _, aa in target_residues)
    idx_of = {resnum: i for i, (resnum, _) in enumerate(target_residues)}
    blosum = substitution_matrices.load("BLOSUM62")
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.substitution_matrix = blosum
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    aln = aligner.align(target_seq, ortholog_seq)[0]

    mapping = {}
    for (t0, t1), (q0, q1) in zip(*aln.aligned):
        for k in range(t1 - t0):
            mapping[t0 + k] = q0 + k

    rows = []
    for resnum in epitope_resnums:
        if resnum not in idx_of:
            rows.append({"resnum": resnum, "human": None, ortholog_label: None, "status": "not_in_target"})
            continue
        i = idx_of[resnum]
        a = target_residues[i][1]
        if i not in mapping:
            rows.append({"resnum": resnum, "human": a, ortholog_label: None, "status": "unaligned"})
            continue
        b = ortholog_seq[mapping[i]]
        status = "identical" if a == b else ("similar" if blosum[a][b] > 0 else "different")
        rows.append({"resnum": resnum, "human": a, ortholog_label: b, "status": status})

    scored = [r for r in rows if r["status"] in ("identical", "similar", "different")]
    n = len(scored)
    ident = sum(r["status"] == "identical" for r in scored)
    simil = sum(r["status"] in ("identical", "similar") for r in scored)
    return {
        "n_epitope_residues": len(rows),
        "n_scored": n,
        "identity": round(ident / n, 3) if n else None,
        "similarity": round(simil / n, 3) if n else None,
        "not_conserved": [r for r in rows if r["status"] == "different"],
        "unscored": [r for r in rows if r["status"] in ("unaligned", "not_in_target")],
        "per_residue": rows,
        "note": "Conservation is necessary, not sufficient, for cross-species binding.",
    }


def rank_designs(designs: list, metric_candidates=("Average_i_pTM", "i_pTM", "Average_pLDDT", "pLDDT"),
                 higher_is_better: bool = True) -> list:
    """
    Orders BindCraftDesign objects by the first metric in `metric_candidates`
    that every design has as a number. Raises OutputFailure listing the
    metrics that do exist if none match, rather than guessing.
    """
    if not designs:
        return []
    for name in metric_candidates:
        if all(isinstance(d.metrics.get(name), (int, float)) for d in designs):
            return sorted(designs, key=lambda d: d.metrics[name], reverse=higher_is_better)
    available = sorted({k for d in designs for k, v in d.metrics.items() if isinstance(v, (int, float))})
    raise OutputFailure(f"None of {list(metric_candidates)} found as a numeric metric on every design. "
                        f"Available numeric metrics: {available}")


def export_competition_csv(entries: list, out_path: str, max_designs: int = 20,
                           min_len: int = 10, max_len: int = 250,
                           molecule_class: str = "protein") -> dict:
    """
    Writes name,sequence,molecule_class, in the order given (best first).

    entries: [{"name": ..., "sequence": ...}, ...]. Invalid or duplicate
    sequences are skipped and reported, never silently kept. Stops at
    `max_designs`. Defaults match the Anthropic x Adaptyv rules as published
    (10-250 residues, at most 20 designs); re-check the live rules before
    submitting, since they can change.
    """
    written, skipped, seen_seq, seen_name = [], [], set(), set()
    for e in entries:
        name, seq = str(e.get("name", "")).strip(), "".join(str(e.get("sequence", "")).split()).upper()
        reason = None
        if not name or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            reason = "name missing or has characters outside A-Z a-z 0-9 _ . -"
        elif name in seen_name:
            reason = "duplicate name"
        elif not seq or set(seq) - _CANONICAL_AA:
            reason = f"non-standard characters: {sorted(set(seq) - _CANONICAL_AA)}"
        elif not (min_len <= len(seq) <= max_len):
            reason = f"length {len(seq)} outside {min_len}-{max_len}"
        elif seq in seen_seq:
            reason = "duplicate sequence"
        if reason:
            skipped.append({"name": name, "reason": reason})
            continue
        if len(written) >= max_designs:
            skipped.append({"name": name, "reason": f"over the {max_designs}-design limit"})
            continue
        seen_seq.add(seq); seen_name.add(name)
        written.append({"name": name, "sequence": seq, "molecule_class": molecule_class})

    if not written:
        raise OutputFailure(f"No valid designs to write. Skipped: {skipped}")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "sequence", "molecule_class"])
        w.writeheader()
        w.writerows(written)
    return {"path": out_path, "n_written": len(written), "skipped": skipped}



if __name__ == "__main__":
    request = DesignRequest(
        target_name="PDL1",
        starting_pdb="/content/bindcraft/example/PDL1.pdb",
        chains="A",
        binder_chain="B",
        hotspot_residues="56,115,123",
        assay_type="western_blot",
        num_candidates=5,
    )
    results = run_pipeline(request)

    out_path = f"{ORYQEVA_ROOT}/results/{request.target_name}_pipeline_output.json"
    with open(out_path, "w") as f:
        json.dump([r.__dict__ for r in results], f, indent=2)

    print(f"{len(results)} candidates scored. Saved to {out_path}")
    for r in results:
        print(
            f"  {r.design_id}: {r.classification} "
            f"(GRAVY={r.solubility_gravy}, strategy={r.classification})"
        )
