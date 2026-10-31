"""Deterministic five-strategy federated MoE Pareto arithmetic.

This is a *model*, not a network performance measurement.
Times in seconds. Payload sizes are *complete wire* bytes, including
metadata, masks, residuals and crypto framing if available.
"""
from dataclasses import dataclass, asdict
import argparse
import json


@dataclass(frozen=True)
class Strategy:
    name: str
    send_bytes: int
    return_bytes: int
    encode_s: float = 0.0
    decode_s: float = 0.0
    compute_s: float = 0.0
    round_trips: int = 1
    extra_s: float = 0.0

    def latency_s(self, bit_rate: float, rtt_s: float) -> float:
        if bit_rate <= 0 or rtt_s < 0:
            raise ValueError("invalid link")
        if min(self.send_bytes, self.return_bytes, self.round_trips) < 0:
            raise ValueError("negative traffic or trips")
        if min(self.encode_s, self.decode_s, self.compute_s, self.extra_s) < 0:
            raise ValueError("negative stage cost")
        return (self.encode_s + self.decode_s + self.compute_s
                + self.extra_s + (self.send_bytes+self.return_bytes)*8/bit_rate
                + self.round_trips*rtt_s)


def compare(*, activation_bytes: int, bit_rate: float,
            rtt_s: float, codec_ratio: float = 2.0,
            codec_overhead_s: float = 0.0,
            local_compute_s: float = 0.0,
            remote_compute_s: float = 0.0,
            acknowledged_delta_bytes: int | None = None,
            action_bytes: int | None = None):
    if activation_bytes <= 0 or codec_ratio < 1 or codec_overhead_s < 0:
        raise ValueError("invalid workload")
    exact_wire = int(activation_bytes/codec_ratio + .999999999)
    strategies = [
        Strategy("local",0,0,compute_s=local_compute_s,round_trips=0),
        Strategy("remote_raw",activation_bytes,activation_bytes,compute_s=remote_compute_s),
        Strategy("remote_codec",exact_wire,exact_wire,
                 encode_s=codec_overhead_s/2,
                 decode_s=codec_overhead_s/2,
                 compute_s=remote_compute_s),
    ]
    if acknowledged_delta_bytes is not None:
        strategies.append(Strategy("remote_acknowledged_delta",
            acknowledged_delta_bytes,acknowledged_delta_bytes,
            compute_s=remote_compute_s))
    if action_bytes is not None:
        strategies.append(Strategy("local_admitted_action",
            action_bytes,action_bytes,compute_s=local_compute_s,
            round_trips=0))
    rows=[dict(name=s.name,one_way_bytes=s.send_bytes,
               total_bytes=s.send_bytes+s.return_bytes,
               latency_ms=1000*s.latency_s(bit_rate,rtt_s))
          for s in strategies]
    return {"bit_rate":bit_rate, "rtt_ms":1000*rtt_s,
            "activation_bytes":activation_bytes,
            "codec_ratio_assumed":codec_ratio,
            "strategies":rows,
            "best_modeled":min(rows,key=lambda r:r["latency_ms"])["name"],
            "warning":"Not measured. Real codecs, state ACK and proof may add costs."}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bytes",type=int,default=1536)
    ap.add_argument("--gbps",type=float,nargs="+",default=[1,10])
    ap.add_argument("--rtt-ms",type=float,default=5)
    ap.add_argument("--ratio",type=float,default=2)
    ap.add_argument("--codec-overhead-ms",type=float,default=0)
    ap.add_argument("--local-compute-ms",type=float,default=0)
    ap.add_argument("--remote-compute-ms",type=float,default=0)
    ap.add_argument("--delta-bytes",type=int)
    ap.add_argument("--action-bytes",type=int)
    args=ap.parse_args()
    print(json.dumps([compare(activation_bytes=args.bytes,
              bit_rate=gbps*1e9,rtt_s=args.rtt_ms/1000,
              codec_ratio=args.ratio,
              codec_overhead_s=args.codec_overhead_ms/1000,
              local_compute_s=args.local_compute_ms/1000,
              remote_compute_s=args.remote_compute_ms/1000,
              acknowledged_delta_bytes=args.delta_bytes,
              action_bytes=args.action_bytes)
              for gbps in args.gbps],indent=2))


if __name__=="__main__":
    main()
