"""
Driver for PaperQA using Nvidia's nemotron-parse VLM.

For more info on nemotron-parse, check out:
- Technical blog: https://developer.nvidia.com/blog/turn-complex-documents-into-usable-data-with-vlm-nvidia-nemotron-parse-1-1/
- Hugging Face weights: https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1
- Model card: https://build.nvidia.com/nvidia/nemotron-parse/modelcard
- API docs: https://docs.nvidia.com/nim/vision-language-models/1.5.0/examples/nemotron-parse/overview.html#nemotron-parse-overview
- Cookbook: https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-Parse-v1.1/build_general_usage_cookbook.ipynb
- AWS Marketplace: https://aws.amazon.com/marketplace/pp/prodview-ny2ngku2i4ge6
"""

import contextlib
import json
import logging
import os
from enum import StrEnum, unique
from typing import (
    TYPE_CHECKING,
    Annotated,
    Literal,
    Self,
    TypeAlias,
    assert_never,
    cast,
    overload,
)

import litellm
from aviary.core import Message, ToolCall
from lmi.rate_limiter import GLOBAL_LIMITER, GLOBAL_RATE_LIMITER_TIMEOUT
from pydantic import (
    AfterValidator,
    BaseModel,
    Field,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

try:
    from aiobotocore.session import get_session
except ImportError:
    get_session = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import numpy as np
    from limits import RateLimitItem

logger = logging.getLogger(__name__)

NVIDIA_API_NEMOTRON_PARSE_RATE_LIMIT = (
    "40 per 1 minute"  # Default rate for Nvidia's API
)


class NemotronLengthError(ValueError):
    r"""
    Error for nemotron-parse running out of context, indicated by the 'length' finish reason.

    This 'length' finish reason comes from the Nvidia NIM wrapping
    nemotron-parse version 1.1 when the model starts babbling (e.g. repeating '\\n').
    It's been seen with the markdown_bbox tool on large figures.
    Retrying is a possible method to skirt this error, but it's a bad idea
    as a 'length' finish reason means nemotron-parse ran out of context,
    and retrying until success just provides a flawed output.
    """


class NemotronBBoxError(ValueError):
    """
    Error for nemotron-parse returning an invalid bounding box.

    Examples include values outside of [0, 1] or a non-positive gap between min and max.
    """


NemotronParseToolName: TypeAlias = Literal[
    "markdown_bbox", "markdown_no_bbox", "detection_only"
]


class NemotronParseBBox(BaseModel):
    """
    Bounding box, values target the range [0, 1], origin is upper left corner.

    In practice, nemotron-parse commonly gives values outside of [0, 1]
    at temperature of 1, but a re-request can get values inside [0, 1].
    """

    @staticmethod
    def validate_min_less_than_max(v: float, info: ValidationInfo) -> float:
        if info.field_name in {"xmax", "ymax"}:
            min_field_name = info.field_name.replace("max", "min")
            with contextlib.suppress(KeyError):
                if info.data[min_field_name] >= v:
                    raise ValueError(
                        f"{min_field_name} must be less than {info.field_name}."
                    )
        return v

    xmin: float = Field(
        description="Lower bound, when looking right across (horizontally) the page.",
        examples=[0.33],
        ge=0,
        le=1,
    )
    xmax: Annotated[
        float,
        Field(
            description="Upper bound, when looking right across (horizontally) the page.",
            examples=[0.65],
            ge=0,
            le=1,
        ),
        AfterValidator(validate_min_less_than_max),
    ]
    ymin: float = Field(
        description="Lower bound, when looking down (vertically) the page.",
        examples=[0.26],
        ge=0,
        le=1,
    )
    ymax: Annotated[
        float,
        Field(
            description="Upper bound, when looking down (vertically) the page.",
            examples=[0.34],
            ge=0,
            le=1,
        ),
        AfterValidator(validate_min_less_than_max),
    ]

    @classmethod
    def from_coordinates(
        cls, coords: tuple[float, float, float, float]
    ) -> "NemotronParseBBox":
        """Create a bbox from a (xmin, xmax, ymin, ymax) tuple."""
        return cls(xmin=coords[0], xmax=coords[1], ymin=coords[2], ymax=coords[3])

    def to_page_coordinates(
        self, height: float, width: float
    ) -> tuple[float, float, float, float]:
        return (
            self.xmin * width,
            self.ymin * height,
            self.xmax * width,
            self.ymax * height,
        )

    def iou(self, other: "NemotronParseBBox") -> float:
        """Calculate the Intersection over Union (IoU) with another bounding box.

        Args:
            other: The other bounding box to compare with.

        Returns:
            IoU score in range [0, 1], where 1 means perfect overlap.
        """
        # Calculate intersection coordinates
        inter_xmin = max(self.xmin, other.xmin)
        inter_ymin = max(self.ymin, other.ymin)
        inter_xmax = min(self.xmax, other.xmax)
        inter_ymax = min(self.ymax, other.ymax)

        # Calculate intersection area (0 if no overlap) and then the union area
        intersection = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
        self_area = (self.xmax - self.xmin) * (self.ymax - self.ymin)
        other_area = (other.xmax - other.xmin) * (other.ymax - other.ymin)
        union = self_area + other_area - intersection
        return intersection / union if union > 0 else 0.0

    def union(self, other: "NemotronParseBBox") -> Self:
        """Create a superset bounding box that contains both bounding boxes.

        Args:
            other: The other bounding box to merge with.

        Returns:
            A new bounding box that is a "superset" box containing both inputs.
        """
        return type(self)(
            xmin=min(self.xmin, other.xmin),
            ymin=min(self.ymin, other.ymin),
            xmax=max(self.xmax, other.xmax),
            ymax=max(self.ymax, other.ymax),
        )


@unique
class NemotronParseClassification(StrEnum):
    """
    Classes that can come from nemotron-parse, values match the API.

    SEE: https://docs.nvidia.com/nim/vision-language-models/1.2.0/examples/retriever/overview.html
    """

    BIBLIOGRAPHY = "Bibliography"
    CAPTION = "Caption"  # Table or picture
    FOOTNOTE = "Footnote"
    FORMULA = "Formula"
    LIST_ITEM = "List-item"  # Numbered, alphanumeric, or bullet point
    PAGE_FOOTER = "Page-footer"
    PAGE_HEADER = "Page-header"
    PICTURE = "Picture"
    SECTION_HEADER = "Section-header"
    TABLE = "Table"
    TABLE_OF_CONTENTS = "TOC"
    TEXT = "Text"  # Regular paragraph text
    TITLE = "Title"


# These classifications will lead to parsing a media in addition to text
CLASSIFICATIONS_WITH_MEDIA = {
    NemotronParseClassification.FORMULA,
    NemotronParseClassification.PICTURE,
    NemotronParseClassification.TABLE,
}


class NemotronParseAnnotatedBBox(BaseModel):
    """Payload for 'detection_only' tool."""

    bbox: NemotronParseBBox
    type: NemotronParseClassification = Field(
        description="Possible classifications of the bbox."
    )


class NemotronParseMarkdown(BaseModel):
    """Payload for 'markdown_no_bbox' tool."""

    text: str = Field(description="Markdown text.")


class NemotronParseMarkdownBBox(NemotronParseAnnotatedBBox, NemotronParseMarkdown):
    """Payload for 'markdown_bbox' tool, or merges with results from detection_only."""

    text: str | None = Field(  # type: ignore[assignment]
        description="Markdown text, or None if from detection_only without text."
    )

    @classmethod
    def merge_with_detection(
        cls,
        markdown_results: "list[NemotronParseMarkdownBBox]",
        detection_results: list[NemotronParseAnnotatedBBox],
        iou_threshold: float,
    ) -> "list[NemotronParseMarkdownBBox]":
        """Merge markdown_bbox and detection_only results using IoU-based matching.

        Args:
            markdown_results: markdown_bbox tool results.
            detection_results: detection_only tool results.
            iou_threshold: Minimum IoU (inclusive) to consider a match.

        Returns:
            Merged results. Text comes from markdown_bbox. Bounding boxes are a superset of:
            1. Bounding boxes from markdown_bbox without an analog in detection_only.
            2. Bounding boxes from detection_only without an analog in markdown_bbox.
            3. Merged bounding boxes from both tools where IoU is at or above the threshold.
        """
        merged: list[NemotronParseMarkdownBBox] = []
        matched_detection_indices: set[int] = set()
        for md_result in markdown_results:
            best_iou = 0.0
            best_detection: NemotronParseAnnotatedBBox | None = None
            best_detection_idx: int | None = None

            for idx, detection_result in enumerate(detection_results):
                if detection_result.type != md_result.type:
                    continue  # Only consider matching types
                iou = md_result.bbox.iou(detection_result.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_detection = detection_result
                    best_detection_idx = idx

            if best_detection is not None and best_iou >= iou_threshold:
                # Create superset bbox containing both markdown and detection bboxes
                merged.append(
                    cls(
                        bbox=md_result.bbox.union(best_detection.bbox),
                        type=md_result.type,
                        text=md_result.text,
                    )
                )
                matched_detection_indices.add(cast(int, best_detection_idx))
            else:
                merged.append(md_result)

        # Add unmatched detection results without text
        for idx, detection_result in enumerate(detection_results):
            if idx not in matched_detection_indices:
                merged.append(
                    cls(
                        bbox=detection_result.bbox,
                        type=detection_result.type,
                        text=None,
                    )
                )

        return merged


MatrixNemotronParseAnnotatedBBox = TypeAdapter(list[list[NemotronParseAnnotatedBBox]])
VectorNemotronParseMarkdown = TypeAdapter(list[NemotronParseMarkdown])
MatrixNemotronParseMarkdownBBox = TypeAdapter(list[list[NemotronParseMarkdownBBox]])


@overload
async def _call_nvidia_api(
    image: "np.ndarray",
    tool_name: Literal["markdown_bbox"],
    api_key: str | None = None,
    api_base: str = ...,
    model_name: str = ...,
    **completion_kwargs,
) -> list[NemotronParseMarkdownBBox]: ...


@overload
async def _call_nvidia_api(
    image: "np.ndarray",
    tool_name: Literal["markdown_no_bbox"],
    api_key: str | None = None,
    api_base: str = ...,
    model_name: str = ...,
    **completion_kwargs,
) -> list[NemotronParseMarkdown]: ...


@overload
async def _call_nvidia_api(
    image: "np.ndarray",
    tool_name: Literal["detection_only"],
    api_key: str | None = None,
    api_base: str = ...,
    model_name: str = ...,
    **completion_kwargs,
) -> list[NemotronParseAnnotatedBBox]: ...


@retry(
    retry=retry_if_exception_type(NemotronBBoxError),
    stop=stop_after_attempt(3),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
@retry(
    retry=retry_if_exception_type(TimeoutError),  # Hitting rate limits
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=GLOBAL_RATE_LIMITER_TIMEOUT),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def _call_nvidia_api(
    image: "np.ndarray",
    tool_name: NemotronParseToolName,
    api_key: str | None = None,
    api_base: str = "https://integrate.api.nvidia.com/v1",
    model_name: str = "nvidia/nemotron-parse",
    rate_limit: "RateLimitItem | str | None" = NVIDIA_API_NEMOTRON_PARSE_RATE_LIMIT,
    **completion_kwargs,
) -> (
    list[NemotronParseMarkdownBBox]
    | list[NemotronParseMarkdown]
    | list[NemotronParseAnnotatedBBox]
):
    """Call the Nvidia API with an image via LiteLLM.

    Args:
        image: Image to parse.
        tool_name: Name of the nemotron-parse tool.
        api_key: Optional API key for Nvidia, default uses the NVIDIA_API_KEY env var.
        api_base: API base URL to pass to the completion,
            default uses Nvidia API's expected base URL.
        model_name: Model name to pass to the completion,
            default uses Nvidia API's expected model name.
        rate_limit: Optional rate limit key for rate limiting,
            default complies with Nvidia API's nemotron-parse limit.
        completion_kwargs: Keyword arguments to pass to the completion.

    Returns:
        Parsed response from the API.
    """
    await GLOBAL_LIMITER.try_acquire(
        ("client|request", "nvidia/nemotron-parse"), rate_limit=rate_limit
    )

    if api_key is None:
        api_key = os.environ["NVIDIA_API_KEY"]
    image_data = Message.create_message(images=[image]).model_dump()["content"][0][
        "image_url"
    ]["url"]
    tool_spec = ToolCall.from_name(tool_name).model_dump(
        exclude={"id": True, "function": {"arguments"}}
    )
    response = await litellm.acompletion(
        model=model_name,
        messages=[{"role": "user", "content": f'<img src="{image_data}" />'}],
        tools=[tool_spec],
        tool_choice=tool_spec,
        api_key=api_key,
        api_base=api_base,
        # Explicitly specify OpenAI-compatible provider over Nvidia NIM provider,
        # so this works with both Nvidia API and DGX Cloud Lepton
        custom_llm_provider=litellm.types.utils.LlmProviders.CUSTOM_OPENAI,
        **completion_kwargs,
    )
    if (
        not isinstance(response, litellm.ModelResponse)
        or len(response.choices) != 1
        or not isinstance(response.choices[0], litellm.Choices)
    ):
        raise NotImplementedError(
            f"Didn't yet handle choices shape of model response {response}."
        )
    if response.choices[0].finish_reason == "length":
        raise NemotronLengthError(
            f"Model response {response} indicates the input"
            f" image of shape {image.shape} is too large or the model started babbling.",
            response.choices[0],  # Include if callers want
        )
    if (
        response.choices[0].finish_reason != "tool_calls"
        or response.choices[0].message.tool_calls is None
        or len(response.choices[0].message.tool_calls) != 1
    ):
        raise NotImplementedError(
            f"Didn't yet handle choice shape of model response {response}."
        )

    args_json = response.choices[0].message.tool_calls[0]["function"]["arguments"]
    try:
        if tool_name == "markdown_bbox":
            (response,) = MatrixNemotronParseMarkdownBBox.validate_json(args_json)
        elif tool_name == "markdown_no_bbox":
            response = VectorNemotronParseMarkdown.validate_json(args_json)
        elif tool_name == "detection_only":
            (response,) = MatrixNemotronParseAnnotatedBBox.validate_json(args_json)
        else:
            assert_never(tool_name)
    except ValidationError as exc:
        raise NemotronBBoxError(
            f"nemotron-parse response {args_json} has invalid bounding box."
        ) from exc
    return response


@overload
async def _call_sagemaker_api(
    image: "np.ndarray",
    tool_name: Literal["markdown_bbox"],
    endpoint_name: str = ...,
    aws_region: str = ...,
    model_name: str = ...,
    **completion_kwargs,
) -> list[NemotronParseMarkdownBBox]: ...


@overload
async def _call_sagemaker_api(
    image: "np.ndarray",
    tool_name: Literal["markdown_no_bbox"],
    endpoint_name: str = ...,
    aws_region: str = ...,
    model_name: str = ...,
    **completion_kwargs,
) -> list[NemotronParseMarkdown]: ...


@overload
async def _call_sagemaker_api(
    image: "np.ndarray",
    tool_name: Literal["detection_only"],
    endpoint_name: str = ...,
    aws_region: str = ...,
    model_name: str = ...,
    **completion_kwargs,
) -> list[NemotronParseAnnotatedBBox]: ...


@retry(
    retry=retry_if_exception_type(NemotronBBoxError),
    stop=stop_after_attempt(3),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def _call_sagemaker_api(
    image: "np.ndarray",
    tool_name: NemotronParseToolName,
    endpoint_name: str = "nemotron-parse",
    aws_region: str = "us-west-2",
    model_name: str = "nvidia/nemotron-parse",
    **completion_kwargs,
) -> (
    list[NemotronParseMarkdownBBox]
    | list[NemotronParseMarkdown]
    | list[NemotronParseAnnotatedBBox]
):
    """Call the AWS SageMaker API with an image via aiobotocore.

    Args:
        image: Image to parse.
        tool_name: Name of the nemotron-parse tool.
        endpoint_name: Name of the AWS SageMaker endpoint,
            default assumes the name is 'nemotron-parse'.
        aws_region: AWS region where the endpoint is deployed, defaults to us-west-2.
        model_name: Model name to pass to the completion,
            default uses AWS Marketplace container's expected model name.
        completion_kwargs: Keyword arguments to pass to the endpoint.

    Returns:
        Parsed response from the API.
    """
    # NOTE: since LiteLLM's AWS SageMaker driver does things like:
    # - Not supporting tool calling per
    #   https://github.com/BerriAI/litellm/blob/v1.79.3-stable/litellm/llms/sagemaker/completion/transformation.py#L70-L71
    # - Converting requests to use "inputs" (aligning with Hugging Face API) per
    #   https://github.com/BerriAI/litellm/blob/v1.79.3-stable/litellm/llms/sagemaker/completion/transformation.py#L182
    # We just use aiobotocore directly here
    tool_spec = ToolCall.from_name(tool_name).model_dump(
        exclude={"id": True, "function": {"arguments"}}
    )
    payload = {  # noqa: FURB173
        "model": model_name,
        "messages": [Message.create_message(images=[image]).model_dump(mode="json")],
        "tools": [tool_spec],
        "tool_choice": tool_spec,
        **completion_kwargs,
    }

    try:
        session = get_session()
    except TypeError as exc:
        raise ImportError(
            "Calling nemotron-parse on AWS SageMaker requires installing with the"
            " 'sagemaker' extra for the 'aiobotocore' package."
            " Please `pip install paper-qa-nemotron[sagemaker]`."
        ) from exc
    async with session.create_client(
        "sagemaker-runtime", region_name=aws_region
    ) as client:
        response = await client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload),
        )
        response_body = await response["Body"].read()
    response = litellm.ModelResponse.model_validate_json(response_body.decode())
    if len(response.choices) != 1 or not isinstance(
        response.choices[0], litellm.Choices
    ):
        raise NotImplementedError(
            f"Didn't yet handle choices shape of model response {response}."
        )
    if response.choices[0].finish_reason == "length":
        raise NemotronLengthError(
            f"Model response {response} indicates the input"
            f" image of shape {image.shape} is too large or the model started babbling.",
            response.choices[0],  # Include if callers want
        )
    if (
        response.choices[0].finish_reason != "stop"
        or response.choices[0].message.tool_calls is None
        or len(response.choices[0].message.tool_calls) != 1
    ):
        raise NotImplementedError(
            f"Didn't yet handle choice shape of model response {response}."
        )

    args_json = response.choices[0].message.tool_calls[0]["function"]["arguments"]
    try:
        if tool_name == "markdown_bbox":
            (response,) = MatrixNemotronParseMarkdownBBox.validate_json(args_json)
        elif tool_name == "markdown_no_bbox":
            response = VectorNemotronParseMarkdown.validate_json(args_json)
        elif tool_name == "detection_only":
            (response,) = MatrixNemotronParseAnnotatedBBox.validate_json(args_json)
        else:
            assert_never(tool_name)
    except ValidationError as exc:
        raise NemotronBBoxError(
            f"nemotron-parse response {args_json} has invalid bounding box."
        ) from exc
    return response
