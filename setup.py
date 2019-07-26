import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sed_tool",
    version="0.0.14",
    author="Leo Cances",
    author_email="leo.cances@gmail.com",
    description="A test package in python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/leocances/SED_tools.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
)

