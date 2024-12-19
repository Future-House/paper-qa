"""Centralized place for lazy LDP imports."""

__all__ = [
    "HAS_LDP_INSTALLED",
    "Agent",
    "Callback",
    "ComputeTrajectoryMetricsMixin",
    "HTTPAgentClient",
    "Memory",
    "MemoryAgent",
    "ReActAgent",
    "RolloutManager",
    "SimpleAgent",
    "SimpleAgentState",
    "UIndexMemoryModel",
    "_Memories",
    "discounted_returns",
    "evaluate_consensus",
    "set_training_mode",
]

from pydantic import TypeAdapter

try:
    from ldp.agent import (
        Agent,
        HTTPAgentClient,
        MemoryAgent,
        ReActAgent,
        SimpleAgent,
        SimpleAgentState,
    )
    from ldp.alg import (
        Callback,
        ComputeTrajectoryMetricsMixin,
        RolloutManager,
        evaluate_consensus,
    )
    from ldp.graph.memory import Memory, UIndexMemoryModel
    from ldp.graph.op_utils import set_training_mode
    from ldp.utils import discounted_returns

    _Memories = TypeAdapter(dict[int, Memory] | list[Memory])  # type: ignore[var-annotated]

    HAS_LDP_INSTALLED = True
except ImportError:
    HAS_LDP_INSTALLED = False

    class ComputeTrajectoryMetricsMixin:  # type: ignore[no-redef]
        """Placeholder parent class for when ldp isn't installed."""

    class Callback:  # type: ignore[no-redef]
        """Placeholder parent class for when ldp isn't installed."""

    RolloutManager = None  # type: ignore[assignment,misc]
    discounted_returns = None  # type: ignore[assignment]
    evaluate_consensus = None  # type: ignore[assignment]
