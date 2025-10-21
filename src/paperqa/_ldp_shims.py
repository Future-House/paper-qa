"""Centralized place for lazy LDP imports."""

__all__ = [
    "HAS_LDP_INSTALLED",
    "Agent",
    "Callback",
    "ComputeTrajectoryMetricsMixin",
    "EnvsTaskDataset",
    "HTTPAgentClient",
    "Memory",
    "MemoryAgent",
    "ReActAgent",
    "RolloutManager",
    "SimpleAgent",
    "SimpleAgentState",
    "UIndexMemoryModel",
    "_Memories",
    "bulk_evaluate_consensus",
    "discounted_returns",
    "set_training_mode",
]

from collections.abc import Sequence

from aviary.env import TaskDataset, TEnvironment
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
        bulk_evaluate_consensus,
    )
    from ldp.data_structures import Trajectory
    from ldp.graph.memory import Memory, UIndexMemoryModel
    from ldp.graph.op_utils import set_training_mode
    from ldp.utils import discounted_returns

    _Memories = TypeAdapter(dict[int, Memory] | list[Memory])  # type: ignore[var-annotated]

    class EnvsTaskDataset(  # TODO: remove after https://github.com/Future-House/aviary/pull/306
        TaskDataset[TEnvironment], ComputeTrajectoryMetricsMixin
    ):
        """
        Task dataset made up of a bunch of individual environments.

        This is useful when doing prototyping with individual environments.
        """

        def __init__(self, envs: TEnvironment | Sequence[TEnvironment]):
            self._envs = envs if isinstance(envs, Sequence) else (envs,)

        def __len__(self) -> int:
            return len(self._envs)

        def get_new_env_by_idx(self, idx: int) -> TEnvironment:
            return self._envs[idx]

        def get_new_env(self) -> TEnvironment:
            if len(self) == 1:
                return self.get_new_env_by_idx(0)
            raise NotImplementedError(
                f'"{self.__class__.__name__}" does not implement get_new_env with'
                f" {len(self)} envs"
            )

    HAS_LDP_INSTALLED = True
except ImportError:
    HAS_LDP_INSTALLED = False

    class ComputeTrajectoryMetricsMixin:  # type: ignore[no-redef]
        """Placeholder parent class for when ldp isn't installed."""

    class Callback:  # type: ignore[no-redef]
        """Placeholder parent class for when ldp isn't installed."""

    Agent = None  # type: ignore[assignment,misc]
    HTTPAgentClient = None  # type: ignore[assignment,misc]
    _Memories = None  # type: ignore[assignment]
    Memory = None  # type: ignore[assignment,misc]
    MemoryAgent = None  # type: ignore[assignment,misc]
    ReActAgent = None  # type: ignore[assignment,misc]
    RolloutManager = None  # type: ignore[assignment,misc]
    SimpleAgent = None  # type: ignore[assignment,misc]
    SimpleAgentState = None  # type: ignore[assignment,misc]
    Trajectory = None  # type: ignore[assignment,misc]
    UIndexMemoryModel = None  # type: ignore[assignment,misc]
    discounted_returns = None  # type: ignore[assignment]
    bulk_evaluate_consensus = None  # type: ignore[assignment]
    set_training_mode = None  # type: ignore[assignment]

    class EnvsTaskDataset(  # type: ignore[no-redef]  # TODO: remove after https://github.com/Future-House/aviary/pull/306
        TaskDataset[TEnvironment]
    ):
        """
        Task dataset made up of a bunch of individual environments.

        This is useful when doing prototyping with individual environments.
        """

        def __init__(self, envs: TEnvironment | Sequence[TEnvironment]):
            self._envs = envs if isinstance(envs, Sequence) else (envs,)

        def __len__(self) -> int:
            return len(self._envs)

        def get_new_env_by_idx(self, idx: int) -> TEnvironment:
            return self._envs[idx]

        def get_new_env(self) -> TEnvironment:
            if len(self) == 1:
                return self.get_new_env_by_idx(0)
            raise NotImplementedError(
                f'"{self.__class__.__name__}" does not implement get_new_env with'
                f" {len(self)} envs"
            )
