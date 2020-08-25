import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CWN_WSD", 
    version="0.0.1",
    author="Yuyu Wu",
    author_email="b06902104@ntu.edu.tw",
    description="A package to use chinese word net to achieve word sense disambigution task",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "argparse",
        "numpy",
        "numpydoc",
        "pandas",
        "recommonmark",
        "torch",
        "tqdm",
        "transformers",
    ],
    include_package_data=True,
    python_requires='>=3.5',
)
