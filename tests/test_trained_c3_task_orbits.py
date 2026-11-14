"""F2 supervised C3 task orbit phase-probe measurement boundaries."""
import unittest,math
import numpy as np
from scripts.audit_trained_c3_task_orbits import phase_diagnostics,validate_task_orbits
class C3TaskOrbitProbeTests(unittest.TestCase):
    def test_exact_phase_orbit_has_small_eq_and_composition(self):
        a=np.arange(3)*2*math.pi/3
        p=np.stack((np.cos(a),np.sin(a)),-1)
        data=np.tile(p,(6,1,1))
        result=phase_diagnostics(data)
        self.assertLess(result["heldout_equivariance_max_defect"],1e-12)
        self.assertLess(result["heldout_phase_composition_max_defect"],1e-12)
        self.assertFalse(result["trained_original_model_c3_circuit_proved"])
    def test_noncompositional_phase_negative_control(self):
        p=np.array([[1,0],[0,1],[-1,0]],dtype=float)
        d=phase_diagnostics(np.tile(p,(6,1,1)))
        self.assertGreater(d["heldout_phase_composition_mean_defect"],.1)
    def test_document_id_leakage_fail_closed(self):
        original=np.arange(100,124).reshape(1,24)
        heldout=np.arange(200,224).reshape(1,24)
        dev=np.arange(300,324).reshape(1,24)
        final=np.arange(400,424).reshape(1,24)
        fit=np.arange(500,572).reshape(1,3,24)
        fit=np.repeat(fit,3,axis=0)
        ev=np.arange(600,672).reshape(1,3,24)
        ev=np.repeat(ev,3,axis=0)
        manifest={"kind":"task_defined_c3_orbits","input_action_rule":"explicit",
          "output_action_rule":"plus_one","tokenizer_sha256":"sha",
          "fit_document_ids":["a","b","c"],
          "eval_document_ids":["c","d","e"]}
        with self.assertRaises(ValueError):
            validate_task_orbits(fit,ev,original,heldout,dev,final,manifest)
if __name__=="__main__":unittest.main()
