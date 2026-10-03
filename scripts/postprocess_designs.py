#!/usr/bin/env python3
"""
Post-process accepted BindCraft designs into a reviewable candidate list.

For every PDB in <design-dir>/Accepted it:
  * reads the binder sequence straight from the PDB (so it does not depend on
    how BindCraft names files) and attaches BindCraft's own metrics by
    matching that exact sequence in mpnn_design_stats.csv / final_design_stats.csv
  * lists interface residues
  * proposes histidine-enriched variants (HEURISTIC, unvalidated)
  * optionally checks epitope conservation against another species' sequence
  * writes <out> in the competition CSV format (name,sequence,molecule_class)

Nothing here predicts binding. Review the printed table and the CSV by hand.

Usage:
  python scripts/postprocess_designs.py \
      --design-dir ~/Oryqeva/BindCraft/EGFR \
      --out ~/Oryqeva/results/EGFR_candidates.csv \
      [--ortholog-fasta mouse_egfr.fasta --ortholog-label mouse] \
      [--his 2 3] [--max 20]
"""
import argparse, csv, glob, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.expanduser("~/Oryqeva"))
import oryqeva_pipeline as op


def read_stats_rows(design_dir):
    """Rows from mpnn_design_stats.csv and final_design_stats.csv (header-only files give none)."""
    rows = []
    for name in ("mpnn_design_stats.csv", "final_design_stats.csv"):
        path = os.path.join(design_dir, name)
        if not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("Design") and r.get("Sequence"):
                    rows.append(r)
    return rows


def match_row(stem, seq, rows):
    """Exact sequence match first (naming-independent), then name prefix."""
    by_seq = [r for r in rows if r["Sequence"].strip() == seq]
    if by_seq:
        return by_seq[0]
    by_name = [r for r in rows if r["Design"].startswith(stem) or stem.startswith(r["Design"])]
    return by_name[0] if len(by_name) == 1 else None


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read_fasta_seq(path):
    seq = []
    for line in open(path):
        if not line.startswith(">"):
            seq.append(line.strip())
    return "".join(seq)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--binder-chain", default="B")
    ap.add_argument("--target-chain", default="A")
    ap.add_argument("--his", type=int, nargs="*", default=[2, 3], help="numbers of His to add per variant")
    ap.add_argument("--ortholog-fasta", help="FASTA with the other species' target sequence")
    ap.add_argument("--ortholog-label", default="ortholog")
    ap.add_argument("--max", type=int, default=20)
    a = ap.parse_args()

    design_dir = os.path.expanduser(a.design_dir)
    pdbs = sorted(glob.glob(os.path.join(design_dir, "Accepted", "*.pdb")))
    if not pdbs:
        sys.exit(f"No PDB files in {design_dir}/Accepted")
    rows = read_stats_rows(design_dir)
    print(f"{len(pdbs)} accepted PDBs, {len(rows)} stats rows found\n")

    ortho = read_fasta_seq(os.path.expanduser(a.ortholog_fasta)) if a.ortholog_fasta else None

    items = []
    for pdb in pdbs:
        stem = os.path.splitext(os.path.basename(pdb))[0]
        seq = op.sequence_from_pdb(pdb, a.binder_chain)
        if not seq:
            print(f"SKIP {stem}: no binder chain '{a.binder_chain}'")
            continue
        row = match_row(stem, seq, rows)
        items.append({
            "id": stem, "pdb": pdb, "seq": seq, "row": row,
            "ipTM": num(row.get("Average_i_pTM")) if row else None,
            "pLDDT": num(row.get("Average_pLDDT")) if row else None,
            "i_pAE": num(row.get("Average_i_pAE")) if row else None,
        })

    unmatched = [i["id"] for i in items if i["row"] is None]
    if unmatched:
        print(f"NOTE: no BindCraft stats row matched for {unmatched}; order below is by file name.\n")
    if all(i["ipTM"] is not None for i in items):
        items.sort(key=lambda i: i["ipTM"], reverse=True)

    print(f"{'design':28s} {'len':>4s} {'i_pTM':>6s} {'pLDDT':>6s} {'i_pAE':>6s}")
    for i in items:
        f = lambda v: "  n/a" if v is None else f"{v:6.2f}"
        print(f"{i['id']:28s} {len(i['seq']):4d} {f(i['ipTM'])} {f(i['pLDDT'])} {f(i['i_pAE'])}")
    print()

    entries = []
    for i in items:
        print(f"=== {i['id']}  ({len(i['seq'])} aa)")
        print(f"parent: {i['seq']}")
        entries.append({"name": i["id"], "sequence": i["seq"]})
        try:
            iface = op.find_interface_residues(i["pdb"], a.binder_chain, a.target_chain)
            iface_txt = ", ".join(x["aa"] + str(x["position"]) for x in iface)
            print(f"interface residues on binder: {len(iface)}  ({iface_txt})")
            res = op.propose_histidine_variants(i["pdb"], a.binder_chain, a.target_chain,
                                                sequence=i["seq"], n_his_options=tuple(a.his))
            for v in res["variants"]:
                print(f"  {v['variant_id']}: {', '.join(v['mutations'])}")
                entries.append({"name": f"{i['id']}_{v['variant_id']}", "sequence": v["sequence"]})
            for w in res["warnings"]:
                print(f"  warning: {w}")
        except op.OryqevaError as e:
            print(f"  His-variant step skipped: {e}")

        if ortho:
            try:
                tr = op.target_contact_residues(i["pdb"], a.binder_chain, a.target_chain)
                from Bio.PDB import PDBParser
                chain = PDBParser(QUIET=True).get_structure("s", i["pdb"])[0][a.target_chain]
                target_residues = [(r.id[1], op.THREE_TO_ONE.get(r.resname, "X"))
                                   for r in chain if r.id[0] == " "]
                cons = op.epitope_conservation(target_residues, ortho, [n for n, _ in tr], a.ortholog_label)
                print(f"  vs {a.ortholog_label}: identity {cons['identity']}, similarity {cons['similarity']} "
                      f"over {cons['n_scored']} contacted target residues")
                for r in cons["not_conserved"]:
                    print(f"    not conserved: {r['human']}{r['resnum']} -> {r[a.ortholog_label]}")
                if cons["unscored"]:
                    print(f"    unscored: {[r['resnum'] for r in cons['unscored']]}")
            except op.OryqevaError as e:
                print(f"  conservation check skipped: {e}")
        print()

    info = op.export_competition_csv(entries, os.path.expanduser(a.out), max_designs=a.max)
    print(f"Wrote {info['n_written']} rows -> {info['path']}")
    for s in info["skipped"]:
        print(f"  skipped {s['name']}: {s['reason']}")
    print("\nReminders: His variants are an untested hypothesis. Re-check the live competition "
          "rules (length, count, format) before submitting.")


if __name__ == "__main__":
    main()
