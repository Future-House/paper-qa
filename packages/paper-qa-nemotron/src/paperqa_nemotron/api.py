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
from typing import TYPE_CHECKING, Annotated, Literal, TypeAlias, assert_never, overload

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

logger = logging.getLogger(__name__)

NEMOTRON_PARSE_RATE_LIMIT = "40 per 1 minute"  # Default rate for nemotron-parse API


class NemotronLengthError(ValueError):
    r"""
    Error for nemotron-parse rejecting an input with 'length' finish reason.

    This can happen if nemotron-parse starts babbling '\\n' a bunch
    at lower DPI, but a re-request can skirt this error.
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

    def to_page_coordinates(
        self, height: float, width: float
    ) -> tuple[float, float, float, float]:
        return (
            self.xmin * width,
            self.ymin * height,
            self.xmax * width,
            self.ymax * height,
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
    """Payload for 'markdown_bbox' tool."""


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
    retry=retry_if_exception_type((NemotronLengthError, NemotronBBoxError)),
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
            default uses nemotron-parse API's expected base URL.
        model_name: Model name to pass to the completion,
            default uses nemotron-parse API's expected model name.
        completion_kwargs: Keyword arguments to pass to the completion.

    Returns:
        Parsed response from the API.
    """
    await GLOBAL_LIMITER.try_acquire(
        ("client|request", "nvidia/nemotron-parse"),
        rate_limit=NEMOTRON_PARSE_RATE_LIMIT,
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
        base_url=api_base,  # Duplicate so LiteLLM can infer LLM provider
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
            f" image of shape {image.shape} is too large or the model started babbling."
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
    retry=retry_if_exception_type((NemotronLengthError, NemotronBBoxError)),
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
            f" image of shape {image.shape} is too large or the model started babbling."
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
