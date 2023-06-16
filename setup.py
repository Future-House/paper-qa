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
    license="MIT",
    packages=["paperqa", "paperqa.contrib"],
    install_requires=[
        "pypdf",
        "langchain>=0.0.198",
        "openai >= 0.27.8",
        "faiss-cpu",
        "PyCryptodome",
        "html2text",
        "tiktoken>=0.4.0",
    ],
    test_suite="tests",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
