import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="January",
    version="0.1",
    author="Jiacheng Huang",
    author_email="chonpcaacpnohc@gmail.com",
    description="a design of experiments platform based on an active learning method with artificial neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Acpnohc/January",
    packages=setuptools.find_packages(),
    install_requires=['Pillow>=5.1.0', 'numpy==1.14.4'],
    entry_points={
        'console_scripts': [
            'douyin_image=douyin_image:main'
        ],
    },
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
