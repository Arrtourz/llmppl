from setuptools import setup, find_packages

setup(
    name="llmppl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch==2.5.0",
        "transformers==4.45.2",
        "openai==0.28.0",
        "tqdm==4.66.5",
        "sentencepiece==0.2.0",
        "bitsandbytes==0.44.1",
        "accelerate==1.0.1",
        "protobuf==5.28.2",
        "tiktoken==0.8.0",
    ],
    author="Zhenyu Xu",
    author_email="thornscrown1220@gmail.com",
    description="A package for calculating perplexity using various language models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Arrtourz/llmppl",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='==3.10',
)