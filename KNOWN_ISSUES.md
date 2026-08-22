# Known issues and unresolved provenance

1. The precise historical derivation of `k_exp` is not fully documented. Deposited audits show it equals neither the raw cycle counter nor the checkup index for all rows.
2. The provenance and encoding details of SOC-window and age/type descriptors are incomplete in the original notebook trail.
3. The activation-energy descriptor’s exact prior/regularization provenance, including `E_a,0` and all historical λ values, must be reported from the final manuscript/source record; it should not be reconstructed by inference.
4. Some historical notebook names use `Rct0`. The measured field is treated publicly as pulse-resistance proxy `Rpulse0`, not electrochemical charge-transfer resistance.
5. The archived notebook lineage does not provide a complete independent proof against every possible preprocessing leakage path. The immutable cell splits and grouped folds are therefore published for audit.
6. An older 40-cell transfer artifact reports a source-minus-target difference of −3.2764 points, while the authoritative final evidence pack and manuscript report +6.987501 points. Both are preserved; the latter is the final result. The generating-code divergence remains unresolved.
7. GPU retraining and latency values may change with hardware, CUDA, PyTorch, and Kaggle scheduling. Artifact-only analysis is the exact reproducibility tier.

