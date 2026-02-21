from setuptools import setup, find_packages
from pathlib import Path

this_dir = Path(__file__).parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8")

setup(
    # PyPI project name
    name="pop-python",
    version="1.1.2",  # update as needed

    author="Guotai Shen",
    author_email="sgt1796@gmail.com",
    description="Prompt Oriented Programming (POP): reusable, composable prompt functions for LLMs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sgt1796/POP",

    # This will pick up the 'POP' package directory
    packages=find_packages(),

    include_package_data=True,
    package_data={
        "POP": [
            "prompts/*.md",
            "schemas/*.json",
        ],
    },

    install_requires=[
        "openai>=1.0.0",
        "requests>=2.25.0",
        "python-dotenv",
        "pydantic>=1.10",
        "numpy>=1.21",
        "backoff",
        "Pillow>=9.0",
    ],
    extras_require={
        "local-embeddings": [
            "torch",
            "transformers>=4.30.0",
        ],
    },

    python_requires=">=3.8",

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
