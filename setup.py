from setuptools import setup

# for typing
__version__ = ""
exec(open("paperqa/version.py").read())

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="paper-qa",
    version=__version__,
    description="LLM Chain for answering questions from docs ",
    author="Andrew White",
    author_email="white.d.andrew@gmail.com",
    url="https://github.com/whitead/paper-qa",
    license="Apache License 2.0",
    packages=["paperqa", "paperqa.contrib"],
    package_data={"paperqa": ["py.typed"]},
    install_requires=[
        "pypdf",
        "pydantic>=2",
        "openai>=1",
        "numpy",
        "PyCryptodome",
        "html2text",
        "tiktoken>=0.4.0",
    ],
    test_suite="tests",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
