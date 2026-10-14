"""
This module provides errors/exceptions and warnings of general use.

Exceptions that are specific to a given package should **not** be here,
but rather in the particular package.

This code is based on that provided by SunPy see
    licenses/SUNPY.rst
"""

import warnings

__all__ = [
    "iFireDeprecationWarning",
    "iFirePendingDeprecationWarning",
    "iFireUserWarning",
    "iFireWarning",
    "warn_deprecated",
    "warn_user",
]


class IFireNBinChangedError(ValueError):
    """
    Error when IFIRE energy bin count (`nbin`) changes within a single telemetry file.

    This can be raised by the parser when handling raw telemetry files, and can be handled to retry parsing and start fresh CDF files after the nbin change offset in the file.

    The message of IFireNBinChangedError should be an int offset into a raw telemetry file. The offset should mark the start of the packet at which `nbin` changes value.

    """


class iFireWarning(Warning):
    """
    The base warning class from which all IMPAX IFIRE warnings should inherit.

    Any warning inheriting from this class is handled by the IMPAX IFIRE
    logger. This warning should not be issued in normal code. Use
    "iFireUserWarning" instead or a specific sub-class.
    """


class iFireUserWarning(UserWarning, iFireWarning):
    """
    The primary warning class for IMPAX IFIRE.

    Use this if you do not need a specific type of warning.
    """


class iFireDeprecationWarning(FutureWarning, iFireWarning):
    """
    A warning class to indicate a deprecated feature.
    """


class iFirePendingDeprecationWarning(PendingDeprecationWarning, iFireWarning):
    """
    A warning class to indicate a soon-to-be deprecated feature.
    """


def warn_user(msg, stacklevel=1):
    """
    Raise a `iFireUserWarning`.

    Parameters
    ----------
    msg : str
        Warning message.
    stacklevel : int
        This is interpreted relative to the call to this function,
        e.g. ``stacklevel=1`` (the default) sets the stack level in the
        code that calls this function.
    """
    warnings.warn(msg, iFireUserWarning, stacklevel + 1)


def warn_deprecated(msg, stacklevel=1):
    """
    Raise a `iFireDeprecationWarning`.

    Parameters
    ----------
    msg : str
        Warning message.
    stacklevel : int
        This is interpreted relative to the call to this function,
        e.g. ``stacklevel=1`` (the default) sets the stack level in the
        code that calls this function.
    """
    warnings.warn(msg, iFireDeprecationWarning, stacklevel + 1)
