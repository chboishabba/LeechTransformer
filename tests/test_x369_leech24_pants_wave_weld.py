"""Weld existing 369 triple fabric/pants seam to 24D ambient E8-wave chart."""
import unittest
from scripts.x369_leech24_pants_wave_weld import (
    pack_three_native_legs,reopen_three_native_legs,selected_wave_pants_seam,certificate)


def source_signature(**kw):
    result=dict(proposition='same',capacity=3,phase='p0',
                wave=('w0','w1'),open=True)
    result.update(kw)
    return result


class Leech24PantsWaveMaxCut(unittest.TestCase):
    def test_three_27_voxel_sources_reopen_through_existing_24d_wave_chart(self):
        source=((1,0,-1,0,0,0,0,0,0),
                (0,0,0,1,0,-1,0,0,0),
                (0,0,0,0,0,0,-1,0,1))
        wave,centre=pack_three_native_legs(source)
        self.assertEqual(wave.shape,(24,1))
        self.assertEqual(len(centre),3)
        self.assertEqual(reopen_three_native_legs(wave,centre),source)

    def test_matching_five_fields_licenses_only_signature_merge(self):
        a=((0,)*9,(1,)*9,(-1,)*9)
        r=selected_wave_pants_seam(a,(source_signature(),)*3,
                                    ('interaction','A','B'))
        self.assertTrue(r['edge_merging_licensed'])
        self.assertEqual(r['merged_source_provenance'],('interaction','A','B'))
        self.assertEqual(reopen_three_native_legs(r['wave24'],r['centre3']),a)
        self.assertFalse(r['actual_leech_lattice_coset_membership_proved'])
        self.assertFalse(r['original_neural_hidden_state_identified_with_chart'])

    def test_gluing_refuses_each_missing_interface_field(self):
        a=((0,)*9,)*3
        changes=dict(proposition='other',capacity=4,phase='p1',
                     wave=('different','wave'),open=False)
        for field,value in changes.items():
            r=selected_wave_pants_seam(a,
                (source_signature(),source_signature(**{field:value}),source_signature()),
                ('I','A','B'))
            self.assertFalse(r['edge_merging_licensed'])
            self.assertFalse(r['five_fields'][0][field])
            self.assertEqual(reopen_three_native_legs(r['wave24'],r['centre3']),a)

    def test_centre_loss_is_real_on_all_three_legs(self):
        zeros=((0,)*9,)*3
        ones=((1,)*9,)*3
        wa,ca=pack_three_native_legs(zeros)
        wb,cb=pack_three_native_legs(ones)
        self.assertEqual(wa,wb)
        self.assertNotEqual(ca,cb)
        self.assertEqual(reopen_three_native_legs(wa,ca),zeros)
        self.assertEqual(reopen_three_native_legs(wb,cb),ones)
        with self.assertRaises(ValueError):
            reopen_three_native_legs(wa,ca[:2])

    def test_exact_source_written_receipt(self):
        r=certificate()
        self.assertTrue(r['selected_seam_matched'])
        self.assertTrue(r['mismatched_seam_refused'])
        self.assertEqual(r['ambient_wave_dimension'],24)
        self.assertFalse(r['actual_leech_glue_constructed'])


if __name__=='__main__':
    unittest.main()
