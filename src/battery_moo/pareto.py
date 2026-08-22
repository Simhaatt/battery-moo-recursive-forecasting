"""Normalization and Pareto-quality helpers."""
from __future__ import annotations
import numpy as np
from .metrics import nondominated_mask

def normalize(values, minimum, maximum):
    values=np.asarray(values,float); minimum=np.asarray(minimum,float); maximum=np.asarray(maximum,float)
    if np.any(maximum<=minimum): raise ValueError("Every normalization maximum must exceed its minimum")
    return (values-minimum)/(maximum-minimum)

def igd(approximation, reference)->float:
    a=np.asarray(approximation,float); r=np.asarray(reference,float)
    return float(np.mean(np.min(np.linalg.norm(r[:,None,:]-a[None,:,:],axis=2),axis=1)))

def igd_plus(approximation, reference)->float:
    a=np.asarray(approximation,float); r=np.asarray(reference,float)
    distances=np.linalg.norm(np.maximum(a[None,:,:]-r[:,None,:],0),axis=2)
    return float(np.mean(np.min(distances,axis=1)))

def spacing(front)->float:
    f=np.asarray(front,float)
    if len(f)<2: return 0.0
    distances=np.abs(f[:,None,:]-f[None,:,:]).sum(axis=2)
    np.fill_diagonal(distances,np.inf); nearest=distances.min(axis=1)
    return float(np.sqrt(np.sum((nearest-nearest.mean())**2)/(len(f)-1)))

def pooled_front(values):
    values=np.asarray(values,float)
    return values[nondominated_mask(values)]

