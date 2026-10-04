#!/usr/bin/env python3
"""
Run an Oryqeva design campaign on EGFR using the bundled BindCraft copy in which
MoMPNN does the sequence-redesign step (third_party/bindcraft_mompnn).

Output goes to a separate folder (~/Oryqeva/BindCraft/<target-name>) so it never
mixes with designs from a standard BindCraft run.

  python scripts/run_egfr_fork.py                  # defaults for EGFR domain III
  nohup python scripts/run_egfr_fork.py > ~/run_fork.out 2>&1 &    # keeps running if the terminal closes

It can be stopped at any time (kill the process). Designs already accepted stay
in <output>/Accepted and can be turned into a CSV with scripts/postprocess_designs.py.

Before the first run, check the MoMPNN step on a real design (seconds):
  python scripts/smoke_test_mompnn.py
"""
import argparse, os, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import oryqeva_pipeline as op


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-pdb", default="~/Oryqeva/targets/6ARU.pdb")
    ap.add_argument("--target-chain", default="A")
    ap.add_argument("--hotspots", default="390-410")
    ap.add_argument("--target-name", default="EGFR_mompnn", help="output folder name")
    ap.add_argument("--num-designs", type=int, default=10, help="stop after this many accepted designs")
    ap.add_argument("--filter-profile", default="default", choices=sorted(op.FILTER_FILES))
    ap.add_argument("--advanced-profile", default="default_4stage_multimer")
    ap.add_argument("--helicity", type=float, default=-2.0, help="weights_helicity override (BindCraft default -0.3)")
    ap.add_argument("--fork-dir", default=os.path.join(REPO, "third_party", "bindcraft_mompnn"))
    ap.add_argument("--hours", type=float, default=11.0, help="give up after this many hours")
    a = ap.parse_args()

    # headless plotting (WSL has no display) and the MoMPNN python
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    venv_py = os.path.expanduser("~/mpnn_env/bin/python")
    if not os.environ.get("ORYQEVA_MPNN_PYTHON") and os.path.exists(venv_py):
        os.environ["ORYQEVA_MPNN_PYTHON"] = venv_py
    op.DRIVE_ROOT = os.path.expanduser("~/Oryqeva")

    fork = os.path.expanduser(a.fork_dir)
    pdb = os.path.expanduser(a.target_pdb)
    problems = []
    if not os.path.exists(os.path.join(fork, "bindcraft.py")):
        problems.append(f"bundled BindCraft not found at {fork}")
    if not os.path.exists(pdb):
        problems.append(f"target PDB not found: {pdb}")
    if problems:
        sys.exit("Cannot start:\n  - " + "\n  - ".join(problems))

    # reuse the AlphaFold2 weights from the existing BindCraft install
    params = os.path.join(fork, "params")
    shared = os.path.expanduser("~/BindCraft/params")
    if not os.path.exists(params):
        if os.path.isdir(shared):
            os.symlink(shared, params)
            print(f"linked AF2 weights: {params} -> {shared}")
        else:
            sys.exit(f"AlphaFold2 weights not found at {params} or {shared}")

    # MoMPNN needs PyTorch in the python that runs ProteinMPNN: check now, not after hours
    mpnn_py = os.environ.get("ORYQEVA_MPNN_PYTHON") or sys.executable
    probe = subprocess.run([mpnn_py, "-c", "import torch; print(torch.__version__)"],
                           capture_output=True, text=True)
    if probe.returncode != 0:
        sys.exit(f"PyTorch is not importable with {mpnn_py}.\n"
                 "Create it with:\n  python3 -m venv ~/mpnn_env\n"
                 "  ~/mpnn_env/bin/pip install torch numpy --index-url https://download.pytorch.org/whl/cpu")
    print(f"MoMPNN python: {mpnn_py} (torch {probe.stdout.strip()})")
    print(f"BindCraft:     {fork}")
    print(f"output:        {op.DRIVE_ROOT}/BindCraft/{a.target_name}\n")

    run = op.run_bindcraft(
        target_pdb_path=pdb, target_name=a.target_name, target_chain=a.target_chain,
        hotspot_residues=a.hotspots, num_designs=a.num_designs,
        filter_profile=a.filter_profile, advanced_profile=a.advanced_profile,
        advanced_overrides={"weights_helicity": a.helicity},
        bindcraft_root=fork, python_exe=sys.executable, auto_trim=True, timeout_hours=a.hours,
    )

    print(f"\nAccepted designs: {len(run.designs)}   runtime: {run.runtime_seconds/60:.1f} min")
    for w in run.warnings:
        print(" -", w)
    print(f"log: {run.log_path}")
    print("Next: python scripts/postprocess_designs.py --design-dir "
          f"{op.DRIVE_ROOT}/BindCraft/{a.target_name} --out ~/Oryqeva/results/{a.target_name}_candidates.csv")


if __name__ == "__main__":
    main()
