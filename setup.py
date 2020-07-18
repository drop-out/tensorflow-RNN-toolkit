import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="RnnToolkit-by-dropout",
    version="0.1",
    author="drop-out",
    author_email="drop-out@foxmail.com",
    description="Convenient RNN building blocks for Tensorflow, including sequence generator, FC layer, RNN layer, etc.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/drop-out/tensorflow-RNN-toolkit",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)