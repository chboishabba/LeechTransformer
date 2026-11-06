"""Extract profiler-visible Tensile/WGM candidates without overstating causality."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--trace-dir",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    files=sorted(Path(a.trace_dir).rglob("*kernel*trace*.csv"))
    observed=[]
    candidates=[]
    for file in files:
        with file.open(newline="",errors="replace") as stream:
            for row in csv.DictReader(stream):
                name=row.get("Kernel_Name", "").strip()
                if name:
                    observed.append({"trace":str(file.relative_to(a.trace_dir)),"kernel":name})
                low=name.lower()
                if any(token in low for token in ("tensile","cijk","wgm","gemm")):
                    candidates.append({"trace":str(file.relative_to(a.trace_dir)),"kernel":name})
    unique=[]; seen=set()
    for row in candidates:
        key=(row["trace"],row["kernel"])
        if key not in seen: seen.add(key); unique.append(row)
    output={"trace_files":[str(f.relative_to(a.trace_dir)) for f in files],
            "observed_kernels":observed,"candidate_kernels":unique,
            "caveat":"Candidates are profiler-visible kernels. Bind them to the leech:block0:qkv ROCTX region and compare profiled/unprofiled receipts before attributing the defect to a specific kernel."}
    Path(a.output).write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps({"traces":len(files),"candidates":len(unique)}))
if __name__ == "__main__": main()
