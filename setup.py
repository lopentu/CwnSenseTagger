import setuptools

with open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CwnSenseTagger", 
    version="0.1.2",
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
        "torch>=1.6",
        "tqdm",
        "transformers>=3.2",
    ],
    include_package_data=True,
    python_requires='>=3.5',
)
