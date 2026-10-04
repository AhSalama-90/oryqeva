#!/usr/bin/env python3
"""
Quick check (seconds, not hours) that the MoMPNN step works on YOUR machine,
using a real BindCraft design. Run this before starting any long BindCraft run.

It calls the same function BindCraft calls (functions/mompnn_backend.py) on one
accepted design, with the real ProteinMPNN script and the real MoMPNN checkpoint,
and prints what comes back.

  python scripts/smoke_test_mompnn.py
  python scripts/smoke_test_mompnn.py --pdb ~/Oryqeva/BindCraft/EGFR/Accepted/EGFR_l50_s585414.pdb

Needs PyTorch in the Python that runs ProteinMPNN: set ORYQEVA_MPNN_PYTHON, or
create ~/mpnn_env (the script uses ~/mpnn_env/bin/python if it exists).
"""
import argparse, glob, importlib.util, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(os.path.dirname(HERE), "third_party", "bindcraft_mompnn", "functions", "mompnn_backend.py")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdb", help="a BindCraft complex PDB (target chain A, binder chain B)")
    ap.add_argument("--binder-chain", default="B")
    ap.add_argument("--num-seqs", type=int, default=20)
    a = ap.parse_args()

    pdb = a.pdb
    if not pdb:
        found = sorted(glob.glob(os.path.expanduser("~/Oryqeva/BindCraft/*/Accepted/*.pdb")))
        if not found:
            sys.exit("No PDB given and none found under ~/Oryqeva/BindCraft/*/Accepted/. Use --pdb.")
        pdb = found[0]
    pdb = os.path.expanduser(pdb)
    if not os.path.exists(pdb):
        sys.exit(f"PDB not found: {pdb}")

    venv_py = os.path.expanduser("~/mpnn_env/bin/python")
    if not os.environ.get("ORYQEVA_MPNN_PYTHON") and os.path.exists(venv_py):
        os.environ["ORYQEVA_MPNN_PYTHON"] = venv_py
    print(f"design:          {pdb}")
    print(f"python for MPNN: {os.environ.get('ORYQEVA_MPNN_PYTHON', sys.executable)}")

    spec = importlib.util.spec_from_file_location("mompnn_backend", BACKEND)
    mb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mb)

    adv = {"num_seqs": a.num_seqs, "sampling_temp": 0.1, "backbone_noise": 0.0,
           "omit_AAs": "C", "mpnn_fix_interface": False}
    t0 = time.time()
    try:
        r = mb.mompnn_gen_sequence(pdb, a.binder_chain, "", adv)
    except mb.MoMPNNError as e:
        sys.exit(f"\nFAILED:\n{e}")
    dt = time.time() - t0

    lens = {len(s) for s in r["seq"]}
    print(f"\nOK in {dt:.1f}s: {len(r['seq'])} sequences, binder length(s) {sorted(lens)}")
    for k in range(min(3, len(r["seq"]))):
        print(f"  {k+1}: {r['seq'][k]}   score={r['score'][k]:.3f}  recovery={r['seqid'][k]:.3f}")
    print("\nThis only shows the step runs and returns sequences. It says nothing about binder quality.")


if __name__ == "__main__":
    main()
