#!/usr/bin/env python3
"""
Batch post-processing of BindCraft designs with Oryqeva.

Run inside the BindCraft conda env:
    python run_oryqeva_all.py              # full pipeline (MoMPNN + conjugation + scoring)
    DRY_RUN=1 python run_oryqeva_all.py    # skip MoMPNN/ESM2: just table + CSV from native sequences

Reads:  ~/Oryqeva/BindCraft/<TARGET>/Accepted/*.pdb          (accepted by BindCraft's own filters)
        ~/Oryqeva/BindCraft/<TARGET>/ManualCandidates/*.pdb  (trajectory-only, NOT filter-validated)
Writes: ~/Oryqeva/results/<TARGET>/all_candidates.csv and submission.csv
"""
import os, sys, glob
from collections import OrderedDict

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.join(HOME, "Oryqeva"))

TARGET_NAME      = os.environ.get("TARGET_NAME", "EGFR")
BINDER_CHAIN     = "B"
TARGET_CHAIN     = "A"
MAX_SUBMISSIONS  = 20     # competition cap per track
MAX_PER_BACKBONE = 3      # native + first MoMPNN variants, keeps the set diverse
SUBMIT_TAGGED    = False  # False = submit the plain sequence (no Sortase tag / engineered Cys)
DRY_RUN          = os.environ.get("DRY_RUN") == "1"

BC_ROOT     = os.path.join(HOME, "Oryqeva", "BindCraft", TARGET_NAME)
RESULTS_DIR = os.path.join(HOME, "Oryqeva", "results", TARGET_NAME)
os.makedirs(RESULTS_DIR, exist_ok=True)

THREE = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H',
         'ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W',
         'TYR':'Y','VAL':'V'}

def read_chain(pdb_path, chain):
    """Sequence and mean per-residue pLDDT (B-factor column) of one chain."""
    seq, plddt = OrderedDict(), []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[21] == chain:
                seq[int(line[22:26])] = THREE.get(line[17:20].strip(), "X")
                plddt.append(float(line[60:66]))
    return "".join(seq.values()), (sum(plddt) / len(plddt) if plddt else None)

accepted = sorted(glob.glob(f"{BC_ROOT}/Accepted/*.pdb"))
manual   = sorted(glob.glob(f"{BC_ROOT}/ManualCandidates/*.pdb"))
print(f"Target {TARGET_NAME}: {len(accepted)} accepted, {len(manual)} manual candidate(s). DRY_RUN={DRY_RUN}")
if not accepted and not manual:
    sys.exit(f"No PDB files found under {BC_ROOT}")

op = esm2_model = esm2_alphabet = esm2_batch_converter = None
if not DRY_RUN:
    import oryqeva_pipeline as op
    op.DRIVE_ROOT = os.path.join(HOME, "Oryqeva")
    import esm
    esm2_model, esm2_alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    esm2_batch_converter = esm2_alphabet.get_batch_converter()
    esm2_model.eval()

rows = []
for source, pdbs in (("bindcraft_accepted", accepted), ("manual_unfiltered", manual)):
    for pdb in pdbs:
        backbone = os.path.splitext(os.path.basename(pdb))[0]
        native, mean_plddt = read_chain(pdb, BINDER_CHAIN)
        variants = [("native", native)]
        if not DRY_RUN:
            try:
                fasta = op.run_mompnn(pdb, RESULTS_DIR, chains=BINDER_CHAIN)
                for i, s in enumerate(op.extract_binder_sequences(fasta, pdb, BINDER_CHAIN)):
                    variants.append((f"mompnn{i+1}", s))
            except Exception as e:
                print(f"  [warn] MoMPNN failed for {backbone}: {e}")
        for vname, seq in variants[:MAX_PER_BACKBONE]:
            row = {"design_id": f"{backbone}_{vname}", "backbone": backbone, "source": source,
                   "variant": vname, "sequence": seq, "length": len(seq),
                   "binder_mean_plddt": round(mean_plddt, 2) if mean_plddt else None,
                   "strategy": None, "tagged_sequence": None, "oryqeva_score": None}
            if not DRY_RUN:
                try:
                    conj = op.get_final_conjugation_strategy(
                        pdb_path=pdb, sequence=seq, binder_chain=BINDER_CHAIN, target_chain=TARGET_CHAIN,
                        esm2_model=esm2_model, esm2_alphabet=esm2_alphabet,
                        esm2_batch_converter=esm2_batch_converter)
                    row.update(strategy=conj["strategy"], tagged_sequence=conj["final_sequence"],
                               oryqeva_score=conj.get("oryqeva_score"))
                except Exception as e:
                    print(f"  [warn] conjugation failed for {row['design_id']}: {e}")
                row["gravy"] = op.score_gravy(seq)
                row["net_charge_pct"] = op.score_net_charge(seq)
            rows.append(row)
        print(f"  {backbone}: {min(len(variants), MAX_PER_BACKBONE)} sequence(s) [{source}]")

import pandas as pd
df = pd.DataFrame(rows)
# Rank: filter-validated first, then structure confidence. (The ESM2 prior is ~random on new
# targets -- AUC 0.52 -- so it is reported but NOT used for ranking.)
df["_src"] = df["source"].map({"bindcraft_accepted": 0, "manual_unfiltered": 1})
df = df.sort_values(["_src", "binder_mean_plddt"], ascending=[True, False]).drop(columns="_src")
df = df.drop_duplicates("sequence").reset_index(drop=True)
df.to_csv(f"{RESULTS_DIR}/all_candidates.csv", index=False)

sub = df.head(MAX_SUBMISSIONS).copy()
if SUBMIT_TAGGED:
    sub["final"] = sub["tagged_sequence"].fillna(sub["sequence"])
else:
    sub["final"] = sub["sequence"]
sub[["design_id", "final"]].rename(columns={"design_id": "name", "final": "sequence"}) \
   .assign(molecule_class="protein").to_csv(f"{RESULTS_DIR}/submission.csv", index=False)

print(f"\n{len(df)} unique candidates -> {RESULTS_DIR}/all_candidates.csv")
print(f"{len(sub)} in {RESULTS_DIR}/submission.csv")
print(df[["design_id", "source", "length", "binder_mean_plddt"]].to_string(index=False))
