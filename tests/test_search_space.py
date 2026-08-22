from battery_moo.search_space import Candidate,feasible,repair

def test_manual_configuration_is_feasible():
    assert feasible(Candidate(20,192,2,8,.001))

def test_infeasible_horizon_is_repaired_deterministically():
    assert repair(Candidate(20,64,1,15,.001)).H==8

