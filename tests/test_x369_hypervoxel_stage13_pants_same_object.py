"""Hypervoxel/tetration factorisation and original pants same-object tests."""
import unittest
from scripts.x369_hypervoxel_stage13_pants_same_object import (
    source_to_packet,split_packet,reopen_packet,central_local_family,
    compatible_locals,glue_local_family,project_pants13,refine_address,
    same_packet_pants_seam,certificate,
    stitch_stage13_from_native,split_stage13_to_native,
    admit_native_to_source_stage13)

INDICES=dict(history=0,physical_time=0,refinement=0,frobenius=0,version=0)


def packet():
    return source_to_packet(tuple((i%3)-1 for i in range(9)),
        tuple((i%3)-1 for i in range(13)),bytes.fromhex('00000080 00000000'),
        INDICES)


class HypervoxelStage13SameObject(unittest.TestCase):
    def test_packet_bidi_including_all_indices_and_signed_zero_bytes(self):
        s=packet()
        self.assertEqual(reopen_packet(split_packet(s)),s)
        self.assertEqual(split_packet(s)['original_source_bytes'],
                         bytes.fromhex('00000080 00000000'))
        self.assertEqual(len(split_packet(s)['original_nine']),9)
        self.assertEqual(len(split_packet(s)['source_stage13']),13)

    def test_actual_original_central_sheaf_pointwise_restrictions(self):
        s=packet()
        family=central_local_family(s)
        self.assertEqual(len(family),12)
        self.assertTrue(compatible_locals(family))
        self.assertEqual(glue_local_family(family),project_pants13(s))
        for i in range(12):
            self.assertEqual(family[i],(project_pants13(s)[i],
                                        project_pants13(s)[12]))

    def test_any_chosen_nine_slots_reopen_with_four_residual_trits(self):
        native=(-1,0,1,0,1,-1,1,0,-1)
        residual=(1,0,-1,1)
        for slots in (tuple(range(9)),tuple(range(4,13)),
                      (0,2,4,6,8,10,12,1,3)):
            with self.subTest(slots=slots):
                source=stitch_stage13_from_native(native,residual,slots)
                self.assertEqual(split_stage13_to_native(source,slots),
                                 (native,residual))
                receipt=admit_native_to_source_stage13(native,source,slots)
                self.assertTrue(receipt['admitted'])
                self.assertTrue(receipt['reopen_exact'])
                self.assertFalse(receipt.get('no_implicit_source_map',False))
                corrupted=list(source)
                corrupted[slots[0]]=1 if native[0]!=1 else 0
                failure=admit_native_to_source_stage13(native,tuple(corrupted),slots)
                self.assertFalse(failure['admitted'])
                self.assertTrue(failure['source_native_slot_mismatch'])

    def test_stage13_chart_rejects_colliding_or_incomplete_addresses(self):
        with self.assertRaises(ValueError):
            stitch_stage13_from_native((0,)*9,(0,)*4,(0,)*9)
        with self.assertRaises(ValueError):
            split_stage13_to_native((0,)*13,(0,)*9)
        with self.assertRaises(ValueError):
            stitch_stage13_from_native((0,)*9,(0,)*3,tuple(range(9)))
        with self.assertRaises(ValueError):
            stitch_stage13_from_native((0,)*9,(0,)*4,tuple(range(8))+(13,))

    def test_source_local_centre_mismatch_must_refuse(self):
        f=list(central_local_family(packet()))
        f[2]=(f[2][0],1)
        self.assertFalse(compatible_locals(tuple(f)))
        with self.assertRaises(ValueError):glue_local_family(tuple(f))

    def test_address_rank_refinement_not_self_indexed_tetration(self):
        parent=((0,0,-1),)
        child=((1,1,0),(-1,0,1))
        joined=refine_address(parent,child)
        self.assertEqual(joined,parent+child)
        self.assertEqual(joined[:len(parent)],parent)
        self.assertEqual(joined[len(parent):],child)
        with self.assertRaises(ValueError):
            refine_address(parent,((0,0),))

    def test_same_source_packet_pants_seam_needs_real_five_fields(self):
        s=packet()
        sig=lambda p:dict(proposition='actual',capacity=3,
            phase='a',wave=(0,1),open=True)
        r=same_packet_pants_seam(s,s,sig)
        self.assertTrue(r['edge_merge_licensed'])
        self.assertFalse(r['source_future_equivalence_proved'])
        mismatch=lambda p:dict(proposition='actual',capacity=3,
            phase='b',wave=(0,1),open=True)
        with self.assertRaises(ValueError):
            same_packet_pants_seam(s,s,lambda p:dict(
                proposition='actual',capacity=3,phase='a',wave=(0,1)))

    def test_finite_receipt_does_not_promote_native_source_or_tower(self):
        r=certificate()
        self.assertTrue(r['same_object_pants13_projection'])
        self.assertTrue(r['three_24d_wave_lanes_reopened'])
        self.assertFalse(r['self_indexed_tower_identified_with_product'])
        self.assertFalse(r['actual_native_source_interpreter_executed'])

    def test_wrong_native_domain_and_index_refused(self):
        with self.assertRaises(ValueError):
            source_to_packet((0,)*8,(0,)*13,b'original',INDICES)
        with self.assertRaises(ValueError):
            source_to_packet((0,)*9,(0,)*12,b'original',INDICES)
        with self.assertRaises(ValueError):
            source_to_packet((0,)*9,(0,)*13,b'original',
                             dict(INDICES,unknown=3))


if __name__=='__main__':
    unittest.main()
