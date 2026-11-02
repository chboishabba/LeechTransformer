"""Revision-bound, pre-encoding latency policy + authenticated JQ1 exchange.

Composes three already-owned mechanisms:
  - benchmark_joint_state_quotient.compression_admitted_for_link
  - federated_joint_state_quotient exact selected-execution codec
  - federated_authenticated_session HMAC replay envelope

This is an actual in-process packet interface. The profile comes from
previous calibration frames; the sender never probes competing encoders
during a real send. HMAC gives authentication/integrity, not encryption,
networking, durable exactly-once execution, or cross-peer atomic commit.

Original Leech-LILA model: Anatolii Kornienko,
DOI 10.5281/zenodo.18784424. DASHI finite consumer proof is distinct.
"""
from __future__ import annotations

from typing import Mapping

from scripts.benchmark_joint_state_quotient import compression_admitted_for_link
from scripts.federated_authenticated_session import SessionReceiver, seal
from scripts.federated_joint_state_quotient import (
    JointState, pack_joint_selected, unpack_joint_selected)


def pack_authenticated_selected(
    state: JointState, *, policy: dict, model_revision: str,
    history_revision: str, rate_gbps: int, key: bytes,
    session: bytes, sequence: int, selected: tuple[int, ...],
    expert_count: int, itemsize: int = 4, block_width: int = 24,
    acknowledged_baseline: JointState | None = None,
    acknowledged_revision: str | None = None,
    orbit_actions: Mapping[int, str] | None = None,
) -> tuple[bytes, dict]:
    """Decide from a PREVIOUS calibrated profile before encoding payload.

    The receiving peer must independently pin model, source history,
    selected IDs, allowed chart actions, and (if used) the acknowledged
    baseline. The caller must establish the MAC key and profile securely.
    """
    permitted = compression_admitted_for_link(
        policy, model_revision=model_revision,
        history_revision=history_revision, rate_gbps=rate_gbps)
    inner = pack_joint_selected(
        state, selected=selected, expert_count=expert_count,
        itemsize=itemsize, block_width=block_width,
        model_revision=model_revision, history_revision=history_revision,
        acknowledged_baseline=acknowledged_baseline,
        acknowledged_revision=acknowledged_revision,
        permit_compression=permitted,
        orbit_actions=orbit_actions if permitted else None,
        use_e8_actions=bool(permitted and orbit_actions))
    packet = seal(key, session, sequence, history_revision.encode("utf-8"),
                  inner)
    return packet, {
        "policy_choice": "quotient" if permitted else "raw",
        "model_revision": model_revision,
        "history_revision": history_revision,
        "rate_gbps": rate_gbps,
        "authenticated_envelope_bytes": len(packet),
        "inner_packet_bytes": len(inner),
        "authenticated_envelope": "HMAC-SHA256; no encryption or network",
        "commit_status": "receiver must verify, execute, record, then commit"}


def unpack_authenticated_selected(
    packet: bytes, *, receiver: SessionReceiver,
    expected_model: str, expected_history: str,
    expected_selected: tuple[int, ...],
    acknowledged_baseline: JointState | None = None,
    expected_ack_revision: str | None = None,
    expected_orbit_actions: Mapping[int, str] | None = None,
) -> JointState:
    """Verify envelope and exact state; intentionally DO NOT advance state.

    The application owns the authenticated verify->execute->durable
    record->receiver.commit(new_revision) transition. A bad packet must
    not consume a sequence or change authoritative expert history.
    """
    if expected_history.encode("utf-8") != receiver.expected_history_revision:
        raise ValueError("authenticated session and JQ1 history revisions differ")
    inner = receiver.verify(packet)
    return unpack_joint_selected(
        inner, expected_model=expected_model,
        expected_history=expected_history,
        expected_selected=expected_selected,
        acknowledged_baseline=acknowledged_baseline,
        expected_ack_revision=expected_ack_revision,
        expected_orbit_actions=expected_orbit_actions)
