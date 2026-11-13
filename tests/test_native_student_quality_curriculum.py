"""Quality curriculum contracts: no heldout leakage, monotone stage growth.

These CPU tests validate stage and source data governance only; real LILA
quality and native throughput require the pinned Schaka GPU experiment.
"""
import unittest
import numpy as np
from scripts.train_native_student_quality_curriculum import (
    parse_stages,split_train_calibration)

class QualityCurriculumContracts(unittest.TestCase):
    def test_cumulative_layers(self):
        stages=parse_stages("1;0,1;0,1,2,3",12)
        self.assertEqual(stages,((1,),(0,1),(0,1,2,3)))

    def test_auto_cumulative_ranking(self):
        from scripts.train_native_student_quality_curriculum import cumulative_stages_from_ranking
        self.assertEqual(cumulative_stages_from_ranking([4,2,0,3,1]),
                         ((4,),(2,4),(0,2,3,4),(0,1,2,3,4)))
        with self.assertRaises(ValueError):
            cumulative_stages_from_ranking([0,0])

    def test_full_completion(self):
        stages=parse_stages("0;0,1;0,1,2",3)
        self.assertEqual(len(stages[-1]),3)

    def test_bad_stage_cuts(self):
        for item in ("",";", "0;", ";0","-1","0,0","1;0",
                     "0;0","0;0,7","0;x","0;0,,1"):
            with self.subTest(item=item):
                with self.assertRaises(ValueError):
                    parse_stages(item,4)

    def test_fit_and_dev_only_from_train_and_heldout_disjoint(self):
        train=np.tile(np.arange(8,dtype=np.int64),(12,1))
        train[:,0]=np.arange(12)+2
        hold=np.tile(np.arange(8,dtype=np.int64),(4,1))
        hold[:,0]=np.arange(4)+100
        fit,dev,fi,di=split_train_calibration(train,hold,41,.25)
        self.assertEqual(len(fit)+len(dev),12)
        self.assertEqual(len(set(fi)&set(di)),0)
        self.assertTrue(set(fi+di)==set(range(12)))
        heldout_windows={row.tobytes() for row in hold}
        self.assertTrue(all(row.tobytes() not in heldout_windows for row in fit))
        self.assertTrue(all(row.tobytes() not in heldout_windows for row in dev))
        fit2,dev2,fi2,di2=split_train_calibration(train,hold,41,.25)
        self.assertEqual(fi2,fi)
        self.assertEqual(di2,di)
        self.assertTrue(np.array_equal(fit2,fit))
        self.assertTrue(np.array_equal(dev2,dev))

    def test_duplicate_train_rows_grouped_without_fit_dev_leakage(self):
        train=np.tile(np.arange(8,dtype=np.int64),(12,1))
        train[:,0]=np.repeat(np.arange(4),3)
        held=np.tile(np.arange(8,dtype=np.int64),(3,1))
        held[:,0]+=100
        fit,dev,fi,di=split_train_calibration(train,held,41,.2)
        self.assertTrue(len(fit)>1 and len(dev)>0)
        self.assertFalse({row.tobytes() for row in fit}&{row.tobytes() for row in dev})
        self.assertEqual(sorted(fi+di),list(range(12)))

    def test_train_heldout_leakage_rejected(self):
        train=np.tile(np.arange(8,dtype=np.int64),(12,1))
        train[:,0]=np.arange(12)
        held=np.tile(np.arange(8,dtype=np.int64),(2,1))
        held[:,0]=np.array([0,100])
        with self.assertRaises(ValueError):
            split_train_calibration(train,held,41,.2)

    def test_invalid_fraction_rejected(self):
        train=np.tile(np.arange(8,dtype=np.int64),(12,1))
        train[:,0]=np.arange(12)
        held=np.tile(np.arange(8,dtype=np.int64),(2,1))
        held[:,0]=np.array([100,101])
        for frac in (0.,.01,.5,1.):
            with self.subTest(frac=frac):
                with self.assertRaises(ValueError):
                    split_train_calibration(train,held,41,frac)

if __name__=="__main__":
    unittest.main()
