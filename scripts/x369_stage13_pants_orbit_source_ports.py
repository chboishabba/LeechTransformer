"""Same-object Stage13 and typed pants ports over the exact capacity orbit section.

Original Leech-LILA: Anatolii Kornienko. DASHI geometry extension.
Existing source-to-packet, nine-plus-four BIDI, sheaf gluing and typed
boundary owners are used. This adapter does not invent source semantics.
"""
from dataclasses import dataclass
from collections.abc import Mapping

from scripts.x369_original_receipt_multichart_weld import require
from scripts.x369_reachable_capacity_residual_bounds import encode, reopen
from scripts.x369_hypervoxel_stage13_pants_same_object import (
    admit_native_to_source_stage13, stitch_stage13_from_native,
    source_to_packet, split_packet, reopen_packet, project_pants13)
from scripts.x369_full_fabric_pants_source_weld import BoundarySignature, matching_fields

INDEX_ROLES=('history','physical_time','refinement','frobenius','version')


@dataclass(frozen=True)
class RetainedSourcePorts:
    orbit_section: dict
    native9: tuple
    native_slots: tuple
    stage4: tuple
    indices: dict
    boundary: BoundarySignature


def status(stage_reader,pants_reader,index_transport):
    missing=[name for name,value in (
        ('source_hypervoxel_stage13',stage_reader),
        ('source_pants_signature',pants_reader),
        ('source_index_transport',index_transport)) if not callable(value)]
    return dict(status='missing_original_source_ports' if missing else 'source_ports_supplied',
                missing=missing,actual_source_semantics_certified=False)


def attach(h,config,stage_reader,pants_reader):
    require(callable(stage_reader) and callable(pants_reader),
            'original source Stage13 and five-field pants readers required')
    owned=stage_reader(h)
    require(isinstance(owned,Mapping) and set(owned)==
            {'native9','source13','native_slots','indices'},'incomplete actual source stage13')
    chart=admit_native_to_source_stage13(owned['native9'],owned['source13'],owned['native_slots'])
    require(chart['admitted'],'original native9/Stage13 slot disagreement')
    p=source_to_packet(owned['native9'],owned['source13'],h,owned['indices'])
    require(reopen_packet(split_packet(p))==p and project_pants13(p)==tuple(owned['source13']),
            'existing Stage13 BIDI failed')
    return RetainedSourcePorts(encode(h,config),tuple(owned['native9']),
        tuple(owned['native_slots']),tuple(chart['residual_stage4']),
        dict(owned['indices']),BoundarySignature.from_source(pants_reader(h)))


def reopen_ports(packet):
    h=reopen(packet.orbit_section)
    source13=stitch_stage13_from_native(packet.native9,packet.stage4,packet.native_slots)
    p=source_to_packet(packet.native9,source13,h,packet.indices)
    require(project_pants13(p)==source13,'Stage13 gluing mismatch')
    return p,packet.boundary


def source_edge(packet,u,step,stage_reader,pants_reader,index_transport):
    """Check actual source step and separately owned index transport.

    The index owner predicts all five successor index values from the
    previous packet and actual control. It cannot be inferred from orbit rank.
    Source readouts are preserved without requiring pants match.
    """
    require(callable(step) and callable(index_transport),'actual source step/index owner required')
    before,boundary=reopen_ports(packet)
    h=before.source_bytes
    cfg=packet.orbit_section['canonical_section']['config']
    require(attach(h,cfg,stage_reader,pants_reader)==packet,'source reader or packet changed')
    obs,n=step(h,u)
    after=attach(n,cfg,stage_reader,pants_reader)
    p,_=reopen_ports(after)
    require(p.source_bytes==n,'original successor bytes changed')
    expected=index_transport(before,u)
    require(isinstance(expected,Mapping) and set(expected)==set(INDEX_ROLES) and
            dict(expected)==after.indices,'independent source index transport mismatch')
    return obs,after,dict(original_source_readout_preserved=True,
        original_source_successor_reopened=True,
        source_index_transport_matches=True,
        boundary_fields_equal=matching_fields(boundary,after.boundary),
        all_future_reduced_chart_sufficiency_certified=False)


def model_ports(model):
    """Existing optional original source methods; never attach fake defaults."""
    return (getattr(model,'source_hypervoxel_stage13',None),
            getattr(model,'source_pants_signature',None),
            getattr(model,'source_index_transport',None))
