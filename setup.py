from pathlib import Path

from setuptools import find_packages, setup

ABS_PATH = Path().absolute()

VERSION = {}  # type: ignore
with (ABS_PATH / "arrayfire_wrapper" / "version.py").open("r") as version_file:
    exec(version_file.read(), VERSION)

setup(
    name="arrayfire-python-wrapper",
    version=VERSION["VERSION"],
    description="ArrayFire Python Wrapper",
    licence="BSD",
    long_description=(ABS_PATH / "README.md").open("r").read(),
    long_description_content_type="text/markdown",
    author="ArrayFire",
    author_email="technical@arrayfire.com",
    url="http://arrayfire.com",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries",
    ],
    keywords="arrayfire c python wrapper parallel computing gpu cpu opencl oneapi",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10.0",
    zip_safe=False,
)
