import numpy as np
from battery_moo.pareto import igd,igd_plus,normalize,pooled_front,spacing

def test_normalization_endpoints():
    assert np.allclose(normalize([[1,10]],[1,10],[3,20]),[[0,0]])

def test_pooled_front_removes_dominated_point():
    f=pooled_front([[0,1],[1,0],[1,1]])
    assert len(f)==2

def test_identical_reference_metrics_are_zero():
    f=np.array([[0.,1.],[1.,0.]])
    assert igd(f,f)==0 and igd_plus(f,f)==0 and spacing(f)>=0

