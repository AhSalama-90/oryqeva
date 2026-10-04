#!/usr/bin/env python3
"""
Turn accepted BindCraft designs into one reviewable candidate list and a competition CSV.

For every PDB in <design-dir>/Accepted (one or more design dirs) it:
  * reads the binder sequence straight from the PDB, and attaches BindCraft's own metrics by matching
    that exact sequence in mpnn_design_stats.csv / final_design_stats.csv (so it does not depend on file naming)
  * lists interface residues
  * proposes histidine-enriched variants (HEURISTIC, unvalidated)
  * optionally checks epitope conservation against another species' sequence
  * writes <out> in the competition format (name,sequence,molecule_class), best i_pTM first

Nothing here predicts binding. Review the printed table and the CSV by hand.

Usage (inside the BindCraft conda environment):
  python scripts/postprocess_designs.py --design-dir ~/Oryqeva/BindCraft/EGFR ~/Oryqeva/BindCraft/EGFR_mompnn \\
      --out ~/Oryqeva/results/EGFR_candidates.csv [--ortholog-fasta mouse_egfr.fasta --ortholog-label mouse] [--his 2 3]
"""
import argparse, csv, glob, os, sys

try:
    import Bio  # noqa: F401
except ImportError:
    sys.exit("Biopython is not installed in this Python:\n  " + sys.executable +
             "\nYou are probably not inside the BindCraft environment. Run:\n"
             "  conda activate BindCraft\nand try again.")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.expanduser("~/Oryqeva"))
import oryqeva_pipeline as op


def read_stats_rows(design_dir):
    rows = []
    for name in ("mpnn_design_stats.csv", "final_design_stats.csv"):
        path = os.path.join(design_dir, name)
        if not os.path.exists(path):
            print(f"  {name}: not found")
            continue
        with open(path, newline="") as f:
            data = list(csv.DictReader(f))
        usable = [r for r in data if r.get("Design") and r.get("Sequence")]
        print(f"  {name}: {len(data)} data rows, {len(usable)} with Design and Sequence")
        rows.extend(usable)
    return rows


def chain_ids_in_pdb(path):
    ids = {}
    for line in open(path):
        if line.startswith("ATOM"):
            ids[line[21]] = ids.get(line[21], 0) + 1
    return ids


def binder_sequence(pdb, chain):
    from Bio.PDB import PDBParser
    model = PDBParser(QUIET=True).get_structure("s", pdb)[0]
    if chain not in [c.id for c in model]:
        return None
    return "".join(op.THREE_TO_ONE.get(r.resname, "X") for r in model[chain] if r.id[0] == " ")


def match_row(stem, seq, rows):
    """Exact sequence match first (naming-independent), then an unambiguous name prefix."""
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
    return "".join(l.strip() for l in open(path) if not l.startswith(">"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design-dir", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--binder-chain", default="B")
    ap.add_argument("--target-chain", default="A")
    ap.add_argument("--his", type=int, nargs="*", default=[2, 3], help="numbers of His to add per variant")
    ap.add_argument("--ortholog-fasta", help="FASTA with the other species' target sequence")
    ap.add_argument("--ortholog-label", default="ortholog")
    ap.add_argument("--max", type=int, default=20)
    a = ap.parse_args()

    ortho = read_fasta_seq(os.path.expanduser(a.ortholog_fasta)) if a.ortholog_fasta else None

    items = []
    for d in a.design_dir:
        design_dir = os.path.expanduser(d)
        pdbs = sorted(glob.glob(os.path.join(design_dir, "Accepted", "*.pdb")))
        print(f"{design_dir}: {len(pdbs)} accepted PDBs")
        if not pdbs:
            continue
        print("  BindCraft stats files:")
        rows = read_stats_rows(design_dir)
        for pdb in pdbs:
            stem = os.path.splitext(os.path.basename(pdb))[0]
            seq = binder_sequence(pdb, a.binder_chain)
            if not seq:
                print(f"  SKIP {stem}: no residues for binder chain '{a.binder_chain}'. "
                      f"Chains present (ATOM counts): {chain_ids_in_pdb(pdb)}. "
                      f"Use --binder-chain / --target-chain to match.")
                continue
            row = match_row(stem, seq, rows)
            items.append({"id": stem, "pdb": pdb, "seq": seq, "row": row,
                          "ipTM": num(row.get("Average_i_pTM")) if row else None,
                          "pLDDT": num(row.get("Average_pLDDT")) if row else None,
                          "i_pAE": num(row.get("Average_i_pAE")) if row else None})
        print()

    if not items:
        sys.exit("No design could be read, so nothing was written. See the lines above.")

    unmatched = [i["id"] for i in items if i["row"] is None]
    if unmatched:
        print(f"NOTE: no BindCraft stats row matched for {unmatched}; their metrics show n/a.\n")
    if all(i["ipTM"] is not None for i in items):
        items.sort(key=lambda i: i["ipTM"], reverse=True)

    print(f"{'design':30s} {'len':>4s} {'i_pTM':>6s} {'pLDDT':>6s} {'i_pAE':>6s}")
    for i in items:
        f = lambda v: "   n/a" if v is None else f"{v:6.2f}"
        print(f"{i['id']:30s} {len(i['seq']):4d} {f(i['ipTM'])} {f(i['pLDDT'])} {f(i['i_pAE'])}")
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
                target_residues = [(r.id[1], op.THREE_TO_ONE.get(r.resname, "X")) for r in chain if r.id[0] == " "]
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
