"""Data sources. Each module exposes a ``Source`` subclass.

To add a new job board: subclass ``BaseSource``, implement ``fetch_raw`` and
``normalize``, then register it in ``REGISTRY`` below.
"""
from .base import BaseSource
from .remotive import RemotiveSource
from .jobicy import JobicySource
from .adzuna import AdzunaSource
from .jooble import JoobleSource
from .remoteok import RemoteOKSource
from .jsearch import JSearchSource

REGISTRY: dict[str, type[BaseSource]] = {
    "remotive": RemotiveSource,
    "jobicy": JobicySource,
    "adzuna": AdzunaSource,
    "jooble": JoobleSource,
    "remoteok": RemoteOKSource,
    "jsearch": JSearchSource,
}
