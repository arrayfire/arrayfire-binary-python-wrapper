import os

_MAJOR = "0"
_MINOR = "1"
# On main and in a nightly release the patch should be one ahead of the last
# released build.
_PATCH = "0"
# This is mainly for nightly builds which have the suffix ".dev$DATE". See
# https://semver.org/#is-v123-a-semantic-version for the semantics.
_SUFFIX = os.environ.get("AF_VERSION_SUFFIX", "")

VERSION_SHORT = "{0}.{1}".format(_MAJOR, _MINOR)
VERSION = "{0}.{1}.{2}{3}".format(_MAJOR, _MINOR, _PATCH, _SUFFIX)

FORGE_VER_MAJOR = "1"
ARRAYFIRE_VER_MAJOR = "3"
ARRAYFIRE_VER_MINOR = "9"
ARRAYFIRE_VERSION = "{0}.{1}".format(ARRAYFIRE_VER_MAJOR, ARRAYFIRE_VER_MINOR)

__version__ = VERSION
