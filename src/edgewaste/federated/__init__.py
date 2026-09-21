"""Federated learning simulation (blueprint Module 10).

A single-machine FedAvg simulator: partitions the training manifest across
N virtual clients, trains a local copy of the hybrid classifier on each
client's shard, and averages the resulting weights (sample-size weighted)
into a new global model — repeated for several rounds. No raw image ever
leaves a client's `local_train` call; only state_dicts are aggregated,
which is the privacy-preserving property Federated Learning is for.

This is a *simulation* (all clients run in this one process, sequentially).
Real distributed deployment across physical edge devices would swap
`local_train`'s in-process call for a Flower `NumPyClient` sending/receiving
these same state_dicts over the network — the FedAvg math doesn't change,
only the transport does. That swap is documented as future work rather than
built, since it needs multiple physical/virtual devices to demonstrate
anything a single-process simulation doesn't already show.
"""

from .simulate import fedavg, local_train, partition_indices, run_simulation

__all__ = ["fedavg", "local_train", "partition_indices", "run_simulation"]
