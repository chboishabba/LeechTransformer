"""Extract profiler-visible Tensile/WGM candidates without overstating causality."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--trace-dir",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    files=sorted(Path(a.trace_dir).glob("*kernel*trace*.csv"))
    candidates=[]
    for file in files:
        with file.open(newline="",errors="replace") as stream:
            for row in csv.DictReader(stream):
                name=" ".join(str(v) for k,v in row.items() if k and "kernel" in k.lower()).strip()
                low=name.lower()
                if any(token in low for token in ("tensile","cijk","wgm","gemm")):
                    candidates.append({"trace":file.name,"kernel":name})
    unique=[]; seen=set()
    for row in candidates:
        key=(row["trace"],row["kernel"])
        if key not in seen: seen.add(key); unique.append(row)
    output={"trace_files":[f.name for f in files],"candidate_kernels":unique,
            "caveat":"Candidates are profiler-visible kernels. Bind them to the leech:block0:qkv ROCTX region and compare profiled/unprofiled receipts before attributing the defect to a specific kernel."}
    Path(a.output).write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps({"traces":len(files),"candidates":len(unique)}))
if __name__ == "__main__": main()
