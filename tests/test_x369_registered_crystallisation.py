import unittest
from scripts.audit_x369_registered_crystallisation import trajectory


def row(t,train,test,law):
    return dict(step=t,audit=dict(training_eight_correct=train,heldout_p22_correct=test,
        hard_label_action_defect_count=0 if test else 2),
        geometry=dict(logit_spectrum=dict(frequency_pairs=[[1,0],[0,1],[1,1],[1,-1]],
            character_pair_fraction=[(1-law)/3,(1-law)/3,law,(1-law)/3],
            centered_energy=1.,effective_rank=2.,parseval_absolute_residual=0.),
            actual_router_steps=[]))


class CrystallisationAudit(unittest.TestCase):
    def test_sampled_fit_and_delayed_transition(self):
        r=trajectory([row(1,False,False,.2),row(100,True,False,.4),row(200,True,True,.9)])
        self.assertEqual(r['training_fit']['first_observed_step'],100)
        self.assertEqual(r['training_fit']['previous_sample_step'],1)
        self.assertEqual(r['joint_fit']['previous_sample_step'],100)
        self.assertTrue(r['delayed_sampled_joint_fit'])
        self.assertEqual(r['sampled_geometry'][-1]['law_character_fraction'],.9)
        self.assertFalse(r['causal_crystallisation_certified'])

    def test_no_training_fit_is_not_grokking(self):
        r=trajectory([row(1,False,True,.8)])
        self.assertIsNone(r['training_fit'])
        self.assertIsNone(r['joint_fit'])
        self.assertFalse(r['delayed_sampled_joint_fit'])
        with self.assertRaises(ValueError):
            trajectory([row(1,False,False,.2),row(1,True,True,.8)])

if __name__=='__main__':unittest.main()
