"""Regression controls for observed-score pair certificate. Does not require GPU."""
from fractions import Fraction as Q
from scripts.observe_x369_quantized_gpu import retrospective_pair_certificate

def run():
    cases=0
    def check(source,candidate,order,source_ok,candidate_ok):
        nonlocal cases
        result=retrospective_pair_certificate(source,candidate,order)
        assert result["observed_strict"] is source_ok,result
        assert result["candidate_strict"] is candidate_ok,result
        assert result["both_orders_certified"] is (source_ok and candidate_ok),result
        assert not result["native_future_error_bound_available"]
        cases+=1
    check([8,7,3,1],[8,7,3,1],[0,1],True,True)
    check([8,7,3,1],[8,7,7,1],[0,1],True,False)
    check([8,7,3,1],[7,8,3,1],[0,1],True,False)
    check([8,7,3,1],[8,7,9,1],[0,1],True,False)
    check([8,8,3,1],[8,7,3,1],[0,1],False,True)
    check([8,7,3,1],[8,7,3,1],[1,0],False,False)
    check([Q(1,3),Q(1,4),Q(1,5)],[Q(1,3),Q(1,4),Q(1,5)],[0,1],True,True)
    r=retrospective_pair_certificate([8,7,3,1],[8,6,3,1],[0,1])
    assert r["max_observed_pair_perturbation"]=="1"
    assert r["source_min_pair_gap"]=="1"
    assert r["candidate_min_pair_gap"]=="2"
    cases+=1
    for bad in ([],[0,0],[0,4],[-1,0]):
        try:retrospective_pair_certificate([3,2,1],[3,2,1],bad)
        except ValueError:cases+=1
        else:raise AssertionError("bad order accepted")
    print("X369 retrospective pair certificate",cases,"PASS; prospective bound remains unavailable")
if __name__=="__main__":run()
