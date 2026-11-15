"""Analyze a real trained-model receipt and calculate bandwidth break-even.

Usage: python -m scripts.analyze_trained_moe_receipt receipt.json
This uses observed codec times/packet sizes, and ideal serialization only.
It does not claim live network timing or geometric MoE certification.
"""
import argparse
import json
from pathlib import Path


def analyze(r):
    a,b=r["wire"]["raw"],r["wire"]["adaptive"]
    saved=a["wire_bytes"]-b["wire_bytes"]
    extra_ms=(b["encode_ms"]+b["decode_ms"])-(a["encode_ms"]+a["decode_ms"])
    # Receipt's remote latency model charges both upload and response at packet size.
    # Saving each way leads to 16*saved bits total. This assumption is explicit.
    crossover_bps=(16*saved*1000/extra_ms if extra_ms>0 and saved>0 else None)
    return {"moe_enabled":r["moe_enabled"],"cache_parity_pass":r["cache_parity_pass"],
            "max_logit_abs":r["full_vs_cache_logits"]["max_abs"],
            "byte_identical":a["sha256"]==b["sha256"],
            "wire_bytes_saved_one_way":saved,
            "compression_ratio_framed":a["wire_bytes"]/b["wire_bytes"],
            "extra_codec_ms":extra_ms,"break_even_bps_bidirectional_model":crossover_bps,
            "break_even_mbps_bidirectional_model":crossover_bps/1e6 if crossover_bps else None,
            "winner_by_receipt_model":{speed:min(options,key=lambda name:options[name]["estimated_remote_latency_ms"])
               for speed,options in r["link_arithmetic"].items()},
            "note":"Not live WAN data; model assumes an equal-size compressed reply. Actual expert outputs may differ."}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("receipt",type=Path)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    result=analyze(json.loads(a.receipt.read_text()))
    if a.output:a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
