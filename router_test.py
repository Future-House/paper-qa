import asyncio

from litellm import Router


async def test(router):
    completion = await router.acompletion(
        "gpt-4o-2024-08-06",
        [{"role": "user", "content": "Recite the Declaration of independence."}],
        stream=True,
        stream_options={"include_usage": True},
    )

    async for chunk in completion:
        pass


async def main():
    router = Router(
        model_list=[
            {
                "model_name": "gpt-4o-2024-08-06",
                "litellm_params": {"model": "gpt-4o-2024-08-06", "temperature": 0.0},
                "rpm": 125,
                "tpm": 7500,
            }
        ],
        num_retries=3,
        retry_after=10,
        cooldown_time=100,
        allowed_fails=2,
    )
    await asyncio.gather(*[test(router) for _ in range(10)])


if __name__ == "__main__":
    asyncio.run(main())
