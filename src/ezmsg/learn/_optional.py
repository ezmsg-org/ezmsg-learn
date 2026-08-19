"""Reporting for the optional backend dependencies.

``ezmsg-learn`` ships its heavyweight backends as extras (``ezmsg-learn[torch]``
and ``ezmsg-learn[sklearn]``) so the pure-numpy processors can be installed
without them. Modules belonging to an extra still import their backend eagerly;
they only wrap the import so that a missing install reports which extra to
install instead of a bare ``ModuleNotFoundError: No module named 'torch'``.
"""


def missing_extra(extra: str, module: str) -> ImportError:
    """Build the error raised when a module's optional backend is not installed.

    Args:
        extra: Name of the extra that provides the backend, e.g. ``"torch"``.
        module: Importing module, normally passed as ``__name__``.

    Returns:
        The :class:`ImportError` to raise; chain it off the original with
        ``raise missing_extra(...) from exc``.
    """
    return ImportError(
        f"{module} requires the optional '{extra}' dependencies of ezmsg-learn, which are not installed. "
        f'Install them with: pip install "ezmsg-learn[{extra}]"'
    )
