"""Execution-state primitives.

SPEC-02 adds only neural cache snapshot/restore. The durable State Fabric,
artifact graph, task DAG, evidence bank and ledger remain deferred to step 4.
"""

from formic.state.snapshot import (
    BranchActivationError,
    ExecutionSnapshot,
    ExecutionStateController,
    PositionState,
    RestoredExecutionState,
    SnapshotError,
    capture_cache_layers,
    capture_model_state,
    restore,
    snapshot,
)

__all__ = [
    "BranchActivationError",
    "ExecutionSnapshot",
    "ExecutionStateController",
    "PositionState",
    "RestoredExecutionState",
    "SnapshotError",
    "capture_cache_layers",
    "capture_model_state",
    "restore",
    "snapshot",
]
