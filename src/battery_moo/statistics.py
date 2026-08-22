"""Small statistical helpers used by analysis scripts."""
from __future__ import annotations
import numpy as np

def vargha_delaney_a12(x,y)->float:
    x=np.asarray(x,float); y=np.asarray(y,float)
    return float(((x[:,None]>y).sum()+0.5*(x[:,None]==y).sum())/(len(x)*len(y)))

def paired_bootstrap_ci(differences, resamples=10000, seed=2026, alpha=.05):
    d=np.asarray(differences,float); rng=np.random.default_rng(seed)
    means=rng.choice(d,(resamples,len(d)),replace=True).mean(axis=1)
    return tuple(float(v) for v in np.quantile(means,[alpha/2,1-alpha/2]))

