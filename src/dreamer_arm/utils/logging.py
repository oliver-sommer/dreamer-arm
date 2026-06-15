"""Console logging configuration for dreamer-arm.

One place to set up human-readable stdout logging so every line carries a
bracketed date+time and level, e.g.::

    [2026-06-14 12:34:56] INFO  step     1000  fps  123.4  score 12.34

Metrics, videos and histograms still go to W&B via
:class:`dreamer_arm.train.logger.WandbLogger`; this module only governs the
console.  Hydra's own job logging is ``disabled`` in ``configs/config.yaml``;
that disabled config runs ``dictConfig(disable_existing_loggers=True)`` before
``main()``, which marks every already-imported logger (ours included)
``.disabled``, so :func:`configure_logging` re-enables them — see there.

``import logging`` inside this file resolves to the stdlib (Python 3 uses
absolute imports), so the module name does not shadow it.
"""

from __future__ import annotations

import logging
import sys

#: ``strftime`` pattern for the bracketed timestamp.
_DATEFMT = "%Y-%m-%d %H:%M:%S"
#: Console line layout: ``[<date time>] <LEVEL> <message>``.
_FORMAT = "[%(asctime)s] %(levelname)-5s %(message)s"
#: Width of the ``[<date time>] LEVEL `` prefix the formatter prepends.  Equals
#: ``len("[2026-06-14 12:34:56] ") + 5 (level, padded) + 1 (space)``.  Multi-line
#: log records indent their continuation lines by this much to line up flush
#: under the message column.
CONTINUATION_INDENT = " " * 28


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
    root.addHandler(handler)
