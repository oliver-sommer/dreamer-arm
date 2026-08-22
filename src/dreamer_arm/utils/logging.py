"""Console logging configuration for dreamer-arm.

One place to set up human-readable stdout logging so every line carries a
time, level, run phase and source logger, e.g.::

    [12:34:56] INFO  train training.trainer   prefill complete at step 520
    [12:35:26] INFO  train utils.tracking     step     1000  fps  123.4

The phase column answers "is this line from collection/training or from an
eval pass?", which is otherwise ambiguous — both run in the same process and
emit through the same loggers. It comes from a :mod:`contextvars` variable
rather than an argument so that *every* record picks it up, including ones
from torchrl / wandb / mujoco that we do not emit ourselves.

Metrics, videos and histograms still go to W&B via
:class:`dreamer_arm.utils.tracking.WandbLogger`; this module only governs the
console.  Hydra's own job logging is ``disabled`` in ``configs/config.yaml``;
that disabled config runs ``dictConfig(disable_existing_loggers=True)`` before
``main()``, which marks every already-imported logger (ours included)
``.disabled``, so :func:`configure_logging` re-enables them — see there.

``import logging`` inside this file resolves to the stdlib (Python 3 uses
absolute imports), so the module name does not shadow it.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from collections.abc import Iterator
from contextvars import ContextVar

#: ``strftime`` pattern for the bracketed timestamp.  Time only: the date is
#: already in the run directory name, and the two columns below need the width.
_DATEFMT = "%H:%M:%S"
#: Console line layout: ``[<time>] <LEVEL> <phase> <logger> <message>``.
_FORMAT = "[%(asctime)s] %(levelname)-5s %(phase)-5s %(src)-18.18s %(message)s"
#: Width of the prefix the formatter prepends, i.e. the column the message
#: starts at.  Multi-line records indent their continuation lines by this much
#: to line up flush under the message column.
CONTINUATION_INDENT = " " * 42

#: Prefix stripped from logger names — it is on every line of ours and so
#: carries no information, while third-party names keep their full path.
_OWN_PACKAGE = "dreamer_arm."

_PHASE: ContextVar[str] = ContextVar("phase", default="setup")


def set_phase(name: str) -> None:
    """Set the phase tag for subsequent log records."""
    _PHASE.set(name)


@contextlib.contextmanager
def phase(name: str) -> Iterator[None]:
    """Tag records emitted inside this block, restoring the previous tag after."""
    token = _PHASE.set(name)
    try:
        yield
    finally:
        _PHASE.reset(token)


def adopt_logger(name: str) -> None:
    """Route a third-party logger through our handler instead of its own.

    Libraries that install a handler *and* set ``propagate = False`` print in
    their own format and never reach :func:`configure_logging`'s columns.
    torchrl does both at import time — which is after ``configure_logging`` has
    run — so this has to be called from wherever the library gets imported,
    not from ``configure_logging`` itself.
    """
    third_party = logging.getLogger(name)
    for handler in list(third_party.handlers):
        third_party.removeHandler(handler)
    third_party.propagate = True


class _ContextFilter(logging.Filter):
    """Inject the ``phase`` and ``src`` fields :data:`_FORMAT` expects."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.phase = _PHASE.get()
        record.src = record.name.removeprefix(_OWN_PACKAGE)
        return True


def configure_logging(level: int | str = logging.INFO) -> None:
    """Install a single stdout handler on the root logger (idempotent).

    Safe to call more than once: any pre-existing handlers are removed first so
    repeated calls (e.g. across tests or reloads) do not double-print.
    """
    # Short, fixed-width level names keep the columns aligned: WARNING -> WARN,
    # CRITICAL -> CRIT; INFO/ERROR/DEBUG already fit five characters.
    logging.addLevelName(logging.WARNING, "WARN")
    logging.addLevelName(logging.CRITICAL, "CRIT")

    root = logging.getLogger()
    root.setLevel(level)

    # Hydra's `disabled` job_logging runs dictConfig(disable_existing_loggers=
    # True) before main(), which marks every logger created at import time
    # (all of ours) `.disabled`, so their records never reach the handler below.
    # Re-enable them so this function genuinely owns console output.
    root.disabled = False
    for existing in root.manager.loggerDict.values():
        if isinstance(existing, logging.Logger):
            existing.disabled = False

    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    # On the handler, not a logger: third-party records need the fields too.
    handler.addFilter(_ContextFilter())
    root.addHandler(handler)
