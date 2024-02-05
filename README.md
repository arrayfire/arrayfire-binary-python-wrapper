# arrayfire-binary-python-wrapper

[ArrayFire](https://github.com/arrayfire/arrayfire) is a high performance library for parallel computing with an easy-to-use API. It enables users to write scientific computing code that is portable across CUDA, OpenCL and CPU devices.
This project is meant to provide thin Python bindings for the ArrayFire C library. It also decouples releases of the main C/C++ library from the Python library by acting as a intermediate library and only wrapping the provided C calls.
This allows the building of large binary wheels only when the underlying ArrayFire version is increased, and the fully-featured Python library can be developed atop independently. This package is not intended to be used directly and merely exposes the 
C functionality required by downstream implementations. This package can exist in two forms, with a bundled binary distribution, or merely as a loader that will load the ArrayFire library from a system or user level install.

# Installing

The arrayfire-binary-python-wrapper can be installed from a variety of sources. [Pre-built wheels](https://repo.arrayfire.com/python/wheels/3.9.0/) are available for a number of systems and toolkits. These will include a binary distribution of the ArrayFire libraries. Installing from PyPI directly will only include a wrapper-only, source distribution that will not contain binaries. In this case, wrapper-only installations will require a separate installation of the ArrayFire C/C++ libraries.
You can get the ArrayFire C/C++ library from the following sources:

- [Download and install binaries](https://arrayfire.com/download)
- [Build and install from source](https://github.com/arrayfire/arrayfire)


**Install the last stable version of python wrapper:**
```
pip install arrayfire-binary-python-wrapper
```

**Install a pre-built wheel:**
```
pip install arrayfire-binary-python-wrapper -f https://repo.arrayfire.com/python/wheels/3.9.0/
```

# Building
The arrayfire-binary-python-wrapper can build wheels in packaged-binary or in system-wrapper modes.
[scikit-build-core](https://github.com/scikit-build/scikit-build-core) is used to provide the python build backend.
The minimal, wrapper-only mode that relies on a system install will be built by default though the regular python build process. For example:
```
pipx run build --wheel
```
Building a full pre-packaged local binary is an involved process that will require referencing the regular ArrayFire [build](https://github.com/arrayfire/arrayfire/wiki/Build-Instructions-for-Linux) [procedures](https://github.com/arrayfire/arrayfire/wiki/Build-Instructions-for-Windows).
Besides the regular ArrayFire CMake configuration, building the binaries is an opt-in process that is set by an environment variable `AF_BUILD_LOCAL_LIBS=1`. Once that environment variable is set, scikit-build-core will take care of cloning ArrayFire, building, and including the necessary binaries.


# Contributing

The community of ArrayFire developers invites you to build with us if you are
interested and able to write top-performing tensor functions. Together we can
fulfill [The ArrayFire
Mission](https://github.com/arrayfire/arrayfire/wiki/The-ArrayFire-Mission-Statement)
for fast scientific computing for all.

Contributions of any kind are welcome! Please refer to [the
wiki](https://github.com/arrayfire/arrayfire/wiki) and our [Code of
Conduct](33) to learn more about how you can get involved with the ArrayFire
Community through
[Sponsorship](https://github.com/arrayfire/arrayfire/wiki/Sponsorship),
[Developer
Commits](https://github.com/arrayfire/arrayfire/wiki/Contributing-Code-to-ArrayFire),
or [Governance](https://github.com/arrayfire/arrayfire/wiki/Governance).

# Citations and Acknowledgements

If you redistribute ArrayFire, please follow the terms established in [the
license](LICENSE).

ArrayFire development is funded by AccelerEyes LLC and several third parties,
please see the list of [acknowledgements](ACKNOWLEDGEMENTS.md) for an
expression of our gratitude.

# Support and Contact Info

* [Slack Chat](https://join.slack.com/t/arrayfire-org/shared_invite/MjI4MjIzMDMzMTczLTE1MDI5ODg4NzYtN2QwNGE3ODA5OQ)
* [Google Groups](https://groups.google.com/forum/#!forum/arrayfire-users)
* ArrayFire Services:  [Consulting](http://arrayfire.com/consulting)  |  [Support](http://arrayfire.com/download)   |  [Training](http://arrayfire.com/training)

# Trademark Policy

The literal mark "ArrayFire" and ArrayFire logos are trademarks of AccelerEyes
LLC (dba ArrayFire). If you wish to use either of these marks in your own
project, please consult [ArrayFire's Trademark
Policy](http://arrayfire.com/trademark-policy/)
