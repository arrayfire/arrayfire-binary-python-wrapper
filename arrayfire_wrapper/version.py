import os

_MAJOR = "0"
_MINOR = "8"
# On main and in a nightly release the patch should be one ahead of the last
# released build.
_PATCH = "0"
# This is mainly for nightly builds which have the suffix ".dev$DATE". See
# https://semver.org/#is-v123-a-semantic-version for the semantics.
_SUFFIX = os.environ.get("AF_VERSION_SUFFIX", "")

WRAPPER_VERSION = "{0}.{1}.{2}{3}".format(_MAJOR, _MINOR, _PATCH, _SUFFIX)

FORGE_VER_MAJOR = "1"
ARRAYFIRE_VER_MAJOR = "3"
ARRAYFIRE_VER_MINOR = "10"
ARRAYFIRE_VER_PATCH = "0"
ARRAYFIRE_VERSION = "AF{0}.{1}.{2}".format(ARRAYFIRE_VER_MAJOR, ARRAYFIRE_VER_MINOR, ARRAYFIRE_VER_PATCH)

VERSION = "{0}+{1}".format(WRAPPER_VERSION, ARRAYFIRE_VERSION)
