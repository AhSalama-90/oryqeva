"""
MoMPNN sequence-generation backend for BindCraft (Oryqeva fork).

BindCraft redesigns each accepted trajectory's binder sequence with
ProteinMPNN (ColabDesign / JAX implementation). This module provides the same
step with MoMPNN instead: the official ProteinMPNN code (PyTorch,
dauparas/ProteinMPNN, MIT) run with the solubility-tuned MoMPNN checkpoint.

In this fork it is the only sequence-redesign backend. Everything downstream
is untouched: AF2 re-prediction of each sequence, the filters, and the CSV.

It returns the same structure as colabdesign's mpnn_model.sample():
    {"seq": [..], "score": [..], "seqid": [..]}   (num_seqs entries each)
where "seq" is the binder sequence only, "score" is ProteinMPNN's mean
negative log-likelihood over designed positions (lower is better), and
"seqid" is ProteinMPNN's reported sequence recovery. These numbers come from
a different model than upstream's, so MPNN_score / MPNN_seq_recovery in the
CSVs are not comparable across the two backends.

Where the weights and script come from (first match wins):
  env ORYQEVA_PROTEINMPNN_SCRIPT / ORYQEVA_MOMPNN_CHECKPOINT, otherwise they are
  fetched once into ORYQEVA_CACHE_DIR (default ~/.oryqeva, the same cache Oryqeva uses): the official
  ProteinMPNN repo is cloned and the MoMPNN checkpoint is downloaded from the
  Oryqeva repository.
Python used to run ProteinMPNN: env ORYQEVA_MPNN_PYTHON, else sys.executable
(it needs PyTorch; on GPUs newer than the installed PyTorch build supports,
the run is retried on CPU, which is fast enough for a model this small).
"""
import json
import os
import re
import subprocess
import sys
import tempfile

_FLOAT = r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"


class MoMPNNError(RuntimeError):
    pass


CACHE_DIR = os.environ.get("ORYQEVA_CACHE_DIR", os.path.expanduser("~/.oryqeva"))
MOMPNN_CKPT_URL = ("https://raw.githubusercontent.com/AhSalama-90/oryqeva/main/"
                   "checkpoints/mompnn_run/mompnn_sol.pt")
PROTEINMPNN_REPO_URL = "https://github.com/dauparas/ProteinMPNN.git"


def _ensure_checkpoint():
    path = os.path.join(CACHE_DIR, "checkpoints", "mompnn_sol.pt")
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import urllib.request
    try:
        with urllib.request.urlopen(MOMPNN_CKPT_URL, timeout=120) as r, open(path + ".part", "wb") as f:
            f.write(r.read())
        os.replace(path + ".part", path)
    except Exception as e:
        raise MoMPNNError(f"Could not download the MoMPNN checkpoint from {MOMPNN_CKPT_URL}: {e}. "
                          f"Place it manually at {path} (or set ORYQEVA_MOMPNN_CHECKPOINT).") from e
    return path


def _ensure_script():
    repo = os.path.join(CACHE_DIR, "ProteinMPNN")
    path = os.path.join(repo, "protein_mpnn_run.py")
    if os.path.exists(path):
        return path
    os.makedirs(CACHE_DIR, exist_ok=True)
    proc = subprocess.run(["git", "clone", "--quiet", "--depth", "1", PROTEINMPNN_REPO_URL, repo],
                          capture_output=True, text=True, timeout=300)
    if proc.returncode != 0 or not os.path.exists(path):
        raise MoMPNNError(f"Could not clone {PROTEINMPNN_REPO_URL}: {proc.stderr[:300]}. "
                          f"Clone it manually to {repo} (or set ORYQEVA_PROTEINMPNN_SCRIPT).")
    return path


def _resolve_assets():
    py = os.environ.get("ORYQEVA_MPNN_PYTHON") or sys.executable
    script = os.environ.get("ORYQEVA_PROTEINMPNN_SCRIPT") or _ensure_script()
    ckpt = os.environ.get("ORYQEVA_MOMPNN_CHECKPOINT") or _ensure_checkpoint()
    for label, path in (("ProteinMPNN script", script), ("MoMPNN checkpoint", ckpt)):
        if not os.path.exists(path):
            raise MoMPNNError(f"{label} not found: {path}")
    return py, script, ckpt


def _chain_residue_order(pdb_path, chain):
    """Residues of `chain` in file order as (resnum, insertion_code) tuples."""
    seen, order = set(), []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[21] == chain:
                key = (int(line[22:26]), line[26].strip())
                if key not in seen:
                    seen.add(key)
                    order.append(key)
    return order


def _interface_indices(pdb_path, binder_chain, interface_residues):
    """'B12,B14' (BindCraft format: chain letter + PDB number) -> 1-based chain indices."""
    order = _chain_residue_order(pdb_path, binder_chain)
    index_of = {resnum: i for i, (resnum, icode) in enumerate(order, start=1) if not icode}
    indices, unknown = [], []
    for tok in (interface_residues or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok[0] != binder_chain or not tok[1:].lstrip("-").isdigit():
            unknown.append(tok)
            continue
        resnum = int(tok[1:])
        if resnum in index_of:
            indices.append(index_of[resnum])
        else:
            unknown.append(tok)
    if unknown:
        print(f"MoMPNN backend: ignoring interface tokens not found on chain {binder_chain}: {unknown}")
    return sorted(set(indices)), len(order)


def _parse_fasta_records(path):
    recs, header, seq = [], None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    recs.append((header, "".join(seq)))
                header, seq = line[1:], []
            elif line:
                seq.append(line)
    if header is not None:
        recs.append((header, "".join(seq)))
    return recs


def _largest_divisor_upto(n, cap):
    return max(d for d in range(1, min(n, cap) + 1) if n % d == 0)


def mompnn_gen_sequence(trajectory_pdb, binder_chain, trajectory_interface_residues, advanced_settings):
    """Drop-in replacement for colabdesign_utils.mpnn_gen_sequence (see module docstring)."""
    py, script, ckpt = _resolve_assets()
    num_seqs = int(advanced_settings["num_seqs"])

    fixed_idx, binder_len = _interface_indices(trajectory_pdb, binder_chain, trajectory_interface_residues) \
        if advanced_settings.get("mpnn_fix_interface", True) else ([], 0)
    name = os.path.splitext(os.path.basename(trajectory_pdb))[0]
    pdb_abs = os.path.abspath(trajectory_pdb)

    with tempfile.TemporaryDirectory(prefix="mompnn_") as tmp:
        cmd = [py, script,
               "--pdb_path", pdb_abs,
               "--pdb_path_chains", binder_chain,
               "--out_folder", tmp,
               "--path_to_model_weights", os.path.dirname(os.path.abspath(ckpt)),
               "--model_name", os.path.splitext(os.path.basename(ckpt))[0],
               "--num_seq_per_target", str(num_seqs),
               "--batch_size", str(_largest_divisor_upto(num_seqs, 20)),
               "--sampling_temp", str(advanced_settings["sampling_temp"]),
               "--backbone_noise", str(advanced_settings.get("backbone_noise", 0.0)),
               "--seed", "0",
               "--suppress_print", "1"]
        if fixed_idx:
            fixed_path = os.path.join(tmp, "fixed_positions.jsonl")
            with open(fixed_path, "w") as f:
                f.write(json.dumps({name: {"A": [], binder_chain: fixed_idx}}) + "\n")
            cmd += ["--fixed_positions_jsonl", fixed_path]
        omit = advanced_settings.get("omit_AAs")
        if omit:
            letters = "".join(a.strip().upper() for a in str(omit).split(",") if a.strip())
            if letters:
                cmd += ["--omit_AAs", letters]

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if proc.returncode != 0 and "CUDA" in (proc.stderr or ""):
            print("MoMPNN backend: GPU run failed with a CUDA error; retrying on CPU.")
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200,
                                  env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        if proc.returncode != 0:
            raise MoMPNNError("ProteinMPNN/MoMPNN run failed:\n" + (proc.stderr or "")[-800:])

        fa = os.path.join(tmp, "seqs", name + ".fa")
        if not os.path.exists(fa):
            raise MoMPNNError(f"Expected output not found: {fa}")
        samples = [(h, s) for h, s in _parse_fasta_records(fa) if "sample=" in h]

    out = {"seq": [], "score": [], "seqid": []}
    for header, seq in samples:
        m_score = re.search(r"(?<!\w)score=" + _FLOAT, header)
        m_rec = re.search(r"seq_recovery=" + _FLOAT, header)
        if not (m_score and m_rec):
            raise MoMPNNError(f"Could not read score/seq_recovery from FASTA header: {header}")
        out["seq"].append(seq.split("/")[-1])
        out["score"].append(float(m_score.group(1)))
        out["seqid"].append(float(m_rec.group(1)))

    if len(out["seq"]) != num_seqs:
        raise MoMPNNError(f"Expected {num_seqs} sequences, got {len(out['seq'])}.")
    return out
