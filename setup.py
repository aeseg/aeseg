import setuptools

long_description = ""

setuptools.setup(
    name="aeseg",
    version="0.0.18",
    author="Leo Cances",
    author_email="leo.cances@gmail.com",
    description="A python package for Sound Event Detection post-processing and optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aeseg/aeseg.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
)

