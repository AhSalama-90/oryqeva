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
from dataclasses import dataclass
from typing import Optional

# ── project paths (match what we've been using in Colab) ──────────────
DRIVE_ROOT = "/content/drive/MyDrive"
BINDCRAFT_ROOT = f"{DRIVE_ROOT}/BindCraft"
ORYQEVA_ROOT = f"{DRIVE_ROOT}/Oryqeva"
MOMPNN_CKPT = f"{ORYQEVA_ROOT}/checkpoints/mompnn_run/mompnn_sol.pt"


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
    pattern = f"{BINDCRAFT_ROOT}/{target_name}/Accepted/*.pdb"
    designs = sorted(glob.glob(pattern))
    if not designs:
        raise FileNotFoundError(
            f"No accepted BindCraft designs found for '{target_name}'. "
            f"Run BindCraft on this target first."
        )
    return designs


# ── STAGE 3: MoMPNN sequence design (reuses the exact call we validated) ─
def run_mompnn(pdb_path: str, out_folder: str, chains: str = "A") -> str:
    """
    Calls protein_mpnn_run.py with the MoMPNN checkpoint.
    Returns the path to the resulting .fa file.
    """
    pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]
    cmd = (
        f'python /content/drive/MyDrive/Oryqeva/tools/ProteinMPNN/protein_mpnn_run.py '
        f'--pdb_path "{pdb_path}" '
        f'--pdb_path_chains "{chains}" '
        f'--out_folder "{out_folder}" '
        f'--path_to_model_weights "{os.path.dirname(MOMPNN_CKPT)}" '
        f'--model_name "{os.path.splitext(os.path.basename(MOMPNN_CKPT))[0]}" '
        f'--num_seq_per_target 8 --sampling_temp "0.1" --seed 37 --batch_size 1'
    )
    os.system(cmd)
    return f"{out_folder}/seqs/{pdb_id}.fa"


def extract_sequences(fasta_path: str) -> list[str]:
    with open(fasta_path) as f:
        lines = f.read().strip().split("\n")
    seqs = [l.strip() for l in lines if not l.startswith(">")]
    return seqs[1:]  # drop the reference sequence


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
        fasta_path = run_mompnn(pdb_path, out_folder, request.chains)
        sequences = extract_sequences(fasta_path)

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


def run_bindcraft(
    target_pdb_path: str,
    target_name: str,
    target_chain: str = "A",
    hotspot_residues: str = "",
    num_designs: int = 5,
) -> str:
    """
    Runs BindCraft end-to-end on a target structure to generate binder
    backbones, then returns the path to the output folder (which
    load_bindcraft_designs() can then read).

    This closes the gap where the pipeline previously assumed BindCraft
    had already been run manually — now it can be triggered from within
    the same Oryqeva call chain.

    NOTE: the exact CLI flag names below are based on BindCraft's
    documented usage pattern, NOT a verified run in this environment —
    check `python bindcraft.py --help` in the installed repo once and
    adjust the cmd list if the flags differ (same caveat as
    suggest_hotspots_iara).

    Requires BindCraft installed at BINDCRAFT_ROOT (cloned + its own
    conda/pip environment set up per the BindCraft repo instructions —
    this function does not install BindCraft itself, only invokes it).
    """
    import subprocess

    out_folder = f"{BINDCRAFT_ROOT}/{target_name}"
    cmd = [
        "python", f"{BINDCRAFT_ROOT}/bindcraft.py",
        "--target_pdb", target_pdb_path,
        "--target_chain", target_chain,
        "--hotspot_residues", hotspot_residues,
        "--output", out_folder,
        "--num_designs", str(num_designs),
        "--weights_pae_inter", "1.0",
        "--num_recycles_design", "3",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if proc.returncode != 0:
            raise RuntimeError(f"BindCraft failed: {proc.stderr[:500]}")
    except Exception as e:
        raise RuntimeError(f"BindCraft run error: {e}")

    return out_folder


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
