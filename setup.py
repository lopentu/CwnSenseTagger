import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CwnSenseTagger", 
    version="0.1",
    author="NTUGIL LOPE Lab",    
    description="A package to use chinese word net to achieve word sense disambigution task",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",        
        "Operating System :: OS Independent",
    ],
    setup_requires=["wheel"],
    install_requires=[
        "numpy",        
        "pandas",        
        "torch",
        "tqdm",
        "transformers",
    ],
    include_package_data=True,
    python_requires='>=3.5',
)
