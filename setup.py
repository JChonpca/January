import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="douyin_image",
    version="0.1",
    author="Zhongqiang Shen",
    author_email="",
    description="add douyin effect to image",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pythonml/douyin_image",
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
