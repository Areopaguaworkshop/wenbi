from setuptools import setup, find_namespace_packages

setup(
    name="cli-anything-wenbi",
    version="0.1.0",
    description="Stateful CLI harness for wenbi media-to-markdown processing",
    packages=find_namespace_packages(include=["cli_anything.*"]),
    install_requires=[
        "click>=8.0",
    ],
    extras_require={
        "wenbi": ["wenbi>=0.140"],
    },
    entry_points={
        "console_scripts": [
            "cli-anything-wenbi=cli_anything.wenbi.wenbi_cli:cli",
            "cli-anything-list=cli_anything.list.list_cli:cli",
        ],
    },
    python_requires=">=3.10",
)