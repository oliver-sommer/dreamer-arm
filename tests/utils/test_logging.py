"""Tests for the console log layout (dreamer_arm.utils.logging)."""

from __future__ import annotations

import io
import logging
from collections.abc import Iterator

import pytest

from dreamer_arm.utils.logging import (
    CONTINUATION_INDENT,
    adopt_logger,
    configure_logging,
    phase,
    set_phase,
)


class _Console:
    """The handler configure_logging installs, redirected to a buffer."""

    def __init__(self, buffer: io.StringIO) -> None:
        self._buffer = buffer

    def lines(self) -> list[str]:
        """Drain and split what has been logged since the last call."""
        text = self._buffer.getvalue()
        self._buffer.seek(0)
        self._buffer.truncate(0)
        return text.splitlines()

    def line(self) -> str:
        lines = self.lines()
        assert len(lines) == 1, f"expected one line, got {lines}"
        return lines[0]


@pytest.fixture
def console() -> Iterator[_Console]:
    """Swap the real handler's stream for a buffer, then put root back.

    configure_logging owns the root handler list, so without the restore the
    remaining tests in the session would lose caplog's handler.
    """
    root = logging.getLogger()
    saved = list(root.handlers)
    # The phase is process-global, so an earlier test's OnlineTrainer.begin
    # leaves it on "train"; reset both ways round to keep this fixture hermetic.
    set_phase("setup")
    configure_logging("INFO")
    buffer = io.StringIO()
    # setStream keeps the formatter and filter under test, unlike a fresh handler.
    root.handlers[-1].setStream(buffer)  # type: ignore[attr-defined]
    try:
        yield _Console(buffer)
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        for handler in saved:
            root.addHandler(handler)
        set_phase("setup")


def test_line_carries_phase_and_logger(console: _Console) -> None:
    logging.getLogger("dreamer_arm.training.trainer").info("prefill complete")
    line = console.line()

    assert " setup " in line, line
    assert "training.trainer" in line, line
    assert "dreamer_arm." not in line, "own package prefix should be stripped"
    assert line.endswith("prefill complete")


def test_third_party_logger_keeps_its_name(console: _Console) -> None:
    logging.getLogger("some_library.storages").info("Initialized LazyTensorStorage")
    assert "some_library" in console.line(), "only our own package prefix should be stripped"


def test_adopted_logger_reaches_our_handler(console: _Console) -> None:
    """torchrl installs a handler and sets propagate=False at import time."""
    noisy = logging.getLogger("noisy_library")
    noisy.addHandler(logging.NullHandler())
    noisy.propagate = False

    adopt_logger("noisy_library")
    noisy.info("Initialized LazyTensorStorage")

    assert "noisy_library" in console.line()


def test_phase_switches_and_restores(console: _Console) -> None:
    set_phase("train")
    with phase("eval"):
        logging.getLogger("dreamer_arm.inference.evaluate").info("inside")
    logging.getLogger("dreamer_arm.training.trainer").info("after")

    inside, after = console.lines()
    assert " eval  " in inside, inside
    assert " train " in after, after


def test_continuation_indent_matches_prefix_width(console: _Console) -> None:
    """The sub-line tracking.py emits must land flush under the message column."""
    logging.getLogger("dreamer_arm.utils.tracking").info("step 1000\n" + CONTINUATION_INDENT + "rew 4.9")
    head, cont = console.lines()

    assert len(CONTINUATION_INDENT) == head.index("step 1000")
    assert cont.index("rew 4.9") == head.index("step 1000")
