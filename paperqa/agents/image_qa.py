import tempfile
from collections.abc import Sequence
from enum import StrEnum, unique
from pathlib import Path
from typing import assert_never, cast

from aviary.core import Environment
from aviary.envs.litqa import GradablePaperQAEnvironment, LitQATaskDataset
from aviary.utils import MultipleChoiceQuestion
from datasets import Dataset, load_dataset
from PIL.Image import Image

from paperqa._ldp_shims import Callback


class ImageQAEnvironment(GradablePaperQAEnvironment):
    """ImageQA applies to both LAB-Bench's FigQA and TableQA."""

    def __init__(
        self,
        *args,
        images: Image | Sequence[Image],
        image_paths: str | Sequence[str],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not isinstance(self._query, MultipleChoiceQuestion):
            raise TypeError(
                f"{type(self).__name__} requires a {MultipleChoiceQuestion.__name__}"
                f" as the query, not {type(self._query)}."
            )
        # FigQA has 1 image with paths, TableQA has 2+ images with paths
        if not isinstance(images, Image) and not isinstance(image_paths, str):
            self._images_with_names: list[tuple[Image, str]] = [
                (image, Path(image_path).name)
                for image, image_path in zip(images, image_paths, strict=True)
            ]
        else:
            self._images_with_names = [
                (cast(Image, images), Path(cast(str, image_paths)).name)
            ]

    async def _reset_docs(self) -> None:
        """Hook to reset the docs when creating the initial state."""
        self._docs.clear_docs()

        # Now add the image(s) to the docs
        with tempfile.TemporaryDirectory() as tmpdir:
            for image, image_name in self._images_with_names:
                tmp_image_path = Path(tmpdir) / image_name
                image.save(tmp_image_path)
                await self._docs.aadd(
                    tmp_image_path,
                    citation=f"Row ID {self._query.question_id} filename {tmp_image_path.name}",
                    settings=self._settings,
                )


@unique
class ImageQASplits(StrEnum):
    FIG_QA = "FigQA"
    TABLE_QA = "TableQA"

    @property
    def images_column(self) -> str:
        if self == ImageQASplits.FIG_QA:
            return "figure"
        if self == ImageQASplits.TABLE_QA:
            return "tables"
        assert_never(self)

    @property
    def paths_column(self) -> str:
        if self == ImageQASplits.FIG_QA:
            return "figure-path"
        if self == ImageQASplits.TABLE_QA:
            return "table-path"
        assert_never(self)


class ImageQATaskDataset(LitQATaskDataset):
    def __init__(self, *args, split: ImageQASplits = ImageQASplits.FIG_QA, **kwargs):
        super().__init__(*args, **kwargs)
        self.split = split
        self.data: Dataset = load_dataset(
            "futurehouse/lab-bench", name=split.value, split="train"
        )

    def get_new_env_by_idx(self, idx: int) -> ImageQAEnvironment:
        return ImageQAEnvironment(
            query=MultipleChoiceQuestion(
                question_id=self.data[idx]["id"],
                question=self.data[idx]["question"],
                ideal_answer=self.data[idx]["ideal"],
                options=self.data[idx]["distractors"],
                prompt_without_id=True,
                **(self._question_kwargs or {}),
            ),
            settings=self._settings,
            docs=self._base_docs.model_copy(),
            rewards=self._rewards,
            images=self.data[idx][self.split.images_column],
            image_paths=self.data[idx][self.split.paths_column],
            **self._env_kwargs,
        )

    def __len__(self) -> int:
        return len(self.data)


class StoreEnvironmentsCallback(Callback):
    """
    Callback to store the environment underlying each trajectory.

    TODO: remove after https://github.com/Future-House/ldp/pull/313 is released.
    """

    def __init__(self):
        self.traj_id_to_envs: dict[str, Environment] = {}

    async def before_rollout(self, traj_id: str, env: Environment) -> None:
        self.traj_id_to_envs[traj_id] = env
