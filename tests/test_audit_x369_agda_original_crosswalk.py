"""Exact source-slot order crosswalk and sign-inversion controls."""
import unittest
from scripts.audit_x369_agda_original_crosswalk import (
    audit_original_agda_site_crosswalk,SLOT_NAMES)
from models.punctured_nine_wave_prior import SITES

symbol={-1:'residueMinus',0:'residueZero',1:'residuePlus'}
def fixture():
    order=' ∷ '.join(SLOT_NAMES)+' ∷ []'
    mappings='\n'.join(
        f'sourceToOriginalTorus {name} =\n  Torus.torusPoint Torus.{symbol[x]} Torus.{symbol[y]}'
        for name,(x,y) in zip(SLOT_NAMES,SITES))
    return ('-- FiniteThreeCycleTorusExact\n'
        'sourceSlotOrder : List OriginalSourceSlot\nsourceSlotOrder =\n  '
        +order+'\n\nsourceToOriginalTorus : type\n'+mappings+'\n')

class AuditOriginalAgdaCrosswalkTests(unittest.TestCase):
    def test_full_original_signed_slot_order(self):
        result=audit_original_agda_site_crosswalk(fixture())
        self.assertTrue(result['pointwise_original_torus_map_exact'])
        self.assertFalse(result['source_agda_kernel_accepted'])
    def test_reject_swapped_order(self):
        s=fixture().replace('siteMinusZero ∷ sitePlusZero',
                            'sitePlusZero ∷ siteMinusZero',1)
        with self.assertRaisesRegex(ValueError,'order'):
            audit_original_agda_site_crosswalk(s)
    def test_reject_coordinate_sign_flip(self):
        s=fixture().replace(
            'sourceToOriginalTorus sitePlusZero =\n  Torus.torusPoint Torus.residuePlus',
            'sourceToOriginalTorus sitePlusZero =\n  Torus.torusPoint Torus.residueMinus',1)
        with self.assertRaisesRegex(ValueError,'mapping'):
            audit_original_agda_site_crosswalk(s)
    def test_reject_missing_carrier(self):
        with self.assertRaisesRegex(ValueError,'carrier'):
            audit_original_agda_site_crosswalk('sourceToOriginalTorus')

if __name__=='__main__':unittest.main()
