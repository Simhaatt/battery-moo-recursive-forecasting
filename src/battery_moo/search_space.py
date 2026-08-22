"""Immutable search-domain definition and feasibility repair."""
from dataclasses import dataclass

L_VALUES=(10,15,20); HIDDEN_VALUES=(64,96,128,160,192)
LAYER_VALUES=(1,2,3); HORIZON_VALUES=(3,5,8,10,15)
LR_LOW=1e-4; LR_HIGH=2e-3; MAX_LENGTH=29

@dataclass(frozen=True)
class Candidate:
    L:int; hidden:int; layers:int; H:int; lr:float

def feasible(c:Candidate)->bool:
    return (c.L in L_VALUES and c.hidden in HIDDEN_VALUES and c.layers in LAYER_VALUES
            and c.H in HORIZON_VALUES and LR_LOW<=c.lr<=LR_HIGH and c.L+c.H<=MAX_LENGTH)

def repair(c:Candidate)->Candidate:
    if feasible(c): return c
    allowed=[h for h in HORIZON_VALUES if c.L+h<=MAX_LENGTH]
    if not allowed: raise ValueError("No feasible rollout horizon for candidate")
    return Candidate(c.L,c.hidden,c.layers,min(allowed,key=lambda h:(abs(h-c.H),-h)),c.lr)

