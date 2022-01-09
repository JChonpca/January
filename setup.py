import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="January",
    version="0.5",
    author="Jiacheng Huang",
    author_email="chonpcaacpnohc@gmail.com",
    description="a design of experiments platform based on an active learning method with artificial neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Acpnohc/January",
    packages=setuptools.find_packages(),
    install_requires=['torch>=1.2.0',
                      'numpy>=1.15.0',
                      'scikit-opt==0.6.5',
                      'scikit-learn>=0.24.0',
                      'matplotlib>=3.3.3',
                      'pandas>=1.0.0'],

    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ),
)
