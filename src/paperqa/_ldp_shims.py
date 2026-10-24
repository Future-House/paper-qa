"""Centralized place for lazy LDP imports."""

__all__ = [
    "HAS_LDP_INSTALLED",
    "Agent",
    "Callback",
    "ComputeTrajectoryMetricsMixin",
    "HTTPAgentClient",
    "ReActAgent",
    "RolloutManager",
    "SimpleAgent",
    "SimpleAgentState",
    "bulk_evaluate_consensus",
    "discounted_returns",
    "set_training_mode",
]

try:
    from ldp.agent import (
        Agent,
        HTTPAgentClient,
        ReActAgent,
        SimpleAgent,
        SimpleAgentState,
    )
    from ldp.alg import (
        Callback,
        ComputeTrajectoryMetricsMixin,
        RolloutManager,
        bulk_evaluate_consensus,
    )
    from ldp.graph.op_utils import set_training_mode
    from ldp.utils import discounted_returns

    HAS_LDP_INSTALLED = True
except ImportError:
    HAS_LDP_INSTALLED = False

    class ComputeTrajectoryMetricsMixin:  # type: ignore[no-redef]
        """Placeholder parent class for when ldp isn't installed."""

    class Callback:  # type: ignore[no-redef]
        """Placeholder parent class for when ldp isn't installed."""

    Agent = None  # type: ignore[assignment,misc]
    HTTPAgentClient = None  # type: ignore[assignment,misc]
    ReActAgent = None  # type: ignore[assignment,misc]
    RolloutManager = None  # type: ignore[assignment,misc]
    SimpleAgent = None  # type: ignore[assignment,misc]
    SimpleAgentState = None  # type: ignore[assignment,misc]
    discounted_returns = None  # type: ignore[assignment]
    bulk_evaluate_consensus = None  # type: ignore[assignment]
    set_training_mode = None  # type: ignore[assignment]
