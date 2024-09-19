import asyncio
import random

from litellm import Router

pre_fill = """


Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc ut finibus massa. Quisque a magna magna. Quisque neque diam, varius sit amet tellus eu, elementum fermentum sapien. Integer ut erat eget arcu rutrum blandit. Morbi a metus purus. Nulla porta, urna at finibus malesuada, velit ante suscipit orci, vitae laoreet dui ligula ut augue. Cras elementum pretium dui, nec luctus nulla aliquet ut. Nam faucibus, diam nec semper interdum, nisl nisi viverra nulla, vitae sodales elit ex a purus. Donec tristique malesuada lobortis. Donec posuere iaculis nisl, vitae accumsan libero dignissim dignissim. Suspendisse finibus leo et ex mattis tempor. Praesent at nisl vitae quam egestas lacinia. Donec in justo non erat aliquam accumsan sed vitae ex. Vivamus gravida diam vel ipsum tincidunt dignissim.

Cras vitae efficitur tortor. Curabitur vel erat mollis, euismod diam quis, consequat nibh. Ut vel est eu nulla euismod finibus. Aliquam euismod at risus quis dignissim. Integer non auctor massa. Nullam vitae aliquet mauris. Etiam risus enim, dignissim ut volutpat eget, pulvinar ac augue. Mauris elit est, ultricies vel convallis at, rhoncus nec elit. Aenean ornare maximus orci, ut maximus felis cursus venenatis. Nulla facilisi.

Maecenas aliquet ante massa, at ullamcorper nibh dictum quis. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Quisque id egestas justo. Suspendisse fringilla in massa in consectetur. Quisque scelerisque egestas lacus at posuere. Vestibulum dui sem, bibendum vehicula ultricies vel, blandit id nisi. Curabitur ullamcorper semper metus, vitae commodo magna. Nulla mi metus, suscipit in neque vitae, porttitor pharetra erat. Vestibulum libero velit, congue in diam non, efficitur suscipit diam. Integer arcu velit, fermentum vel tortor sit amet, venenatis rutrum felis. Donec ultricies enim sit amet iaculis mattis.

Integer at purus posuere, malesuada tortor vitae, mattis nibh. Mauris ex quam, tincidunt et fermentum vitae, iaculis non elit. Nullam dapibus non nisl ac sagittis. Duis lacinia eros iaculis lectus consectetur vehicula. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Interdum et malesuada fames ac ante ipsum primis in faucibus. Ut cursus semper est, vel interdum turpis ultrices dictum. Suspendisse posuere lorem et accumsan ultrices. Duis sagittis bibendum consequat. Ut convallis vestibulum enim, non dapibus est porttitor et. Quisque suscipit pulvinar turpis, varius tempor turpis. Vestibulum semper dui nunc, vel vulputate elit convallis quis. Fusce aliquam enim nulla, eu congue nunc tempus eu.

Nam vitae finibus eros, eu eleifend erat. Maecenas hendrerit magna quis molestie dictum. Ut consequat quam eu massa auctor pulvinar. Pellentesque vitae eros ornare urna accumsan tempor. Maecenas porta id quam at sodales. Donec quis accumsan leo, vel viverra nibh. Vestibulum congue blandit nulla, sed rhoncus libero eleifend ac. In risus lorem, rutrum et tincidunt a, interdum a lectus. Pellentesque aliquet pulvinar mauris, ut ultrices nibh ultricies nec. Mauris mi mauris, facilisis nec metus non, egestas luctus ligula. Quisque ac ligula at felis mollis blandit id nec risus. Nam sollicitudin lacus sed sapien fringilla ullamcorper. Etiam dui quam, posuere sit amet velit id, aliquet molestie ante. Integer cursus eget sapien fringilla elementum. Integer molestie, mi ac scelerisque ultrices, nunc purus condimentum est, in posuere quam nibh vitae velit.
"""


async def test(router):
    completion = await router.acompletion(
        "gpt-4o-2024-08-06",
        [
            {
                "role": "user",
                "content": f"{pre_fill * 3}\n\nRecite the Declaration of independence at a speed of {random.random() * 100} words per minute.",
            }
        ],
        stream=True,
        temperature=0.0,
        stream_options={"include_usage": True},
    )

    async for chunk in completion:
        pass
    print("done", chunk)


async def main():
    router = Router(
        model_list=[
            {
                "model_name": "gpt-4o-2024-08-06",
                "litellm_params": {"model": "gpt-4o-2024-08-06", "temperature": 0.0},
                "rpm": 500,
                "tpm": 30000,
            }
        ],
    )
    await asyncio.gather(*[test(router) for _ in range(16)])


if __name__ == "__main__":
    asyncio.run(main())
