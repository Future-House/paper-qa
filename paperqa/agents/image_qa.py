from aviary.core import Environment

from paperqa._ldp_shims import Callback


class StoreEnvironmentsCallback(Callback):
    """
    Callback to store the environment underlying each trajectory.

    TODO: remove after https://github.com/Future-House/ldp/pull/313 is released.
    """

    def __init__(self):
        self.traj_id_to_envs: dict[str, Environment] = {}

    async def before_rollout(self, traj_id: str, env: Environment) -> None:
        self.traj_id_to_envs[traj_id] = env
