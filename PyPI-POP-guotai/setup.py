from setuptools import setup, find_packages

setup(
    name="POP-guotai",  # Package name
    version="0.2.4",  # Version number
    author="Guotai Shen",
    author_email="sgt1796@gmail.com",  
    description="Reusable, mutable, prompt functions for LLMs.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sgt1796/POP",  
    packages=find_packages(),
    include_package_data=True,  # Include non-Python files
    package_data={
        "POP": ["prompts/*.md"],  # Include all markdown files in the prompts directory
    },
    install_requires=[
        "openai",
        "requests",
        "python-dotenv",
        "pydantic",
        "transformers",
        #"torch",
        "numpy",
        "backoff",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
