from setuptools import setup, find_packages

setup(
    name="stainwarpy",
    version="0.1.8",
    packages=find_packages(),
    install_requires=[
        "numpy>=2.2",
        "tifffile>=2025.5",
        "imagecodecs>=2025.11",
        "scikit-image>=0.25",
        "scipy>=1.16",
        "typer[all]>=0.20",
    ],
    python_requires=">=3.11",
    extras_require={
        "plots": [
            "matplotlib",
        ]
    },
    entry_points={
        "console_scripts": [
            "stainwarpy=stainwarpy.reg_cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
    author="Thusheera Kumarasekara",
    author_email="tckumarasekara@gmail.com",
    description="Tools for image registration between multiplexed and HnE stained tissue images",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tckumarasekara/stainwarpy", 
)