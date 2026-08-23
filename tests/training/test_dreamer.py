from unittest.mock import MagicMock

from dreamer_arm.training.dreamer import _restore_buffer


def test_restore_buffer_uses_checkpoint_stem(tmp_path: object) -> None:
    from pathlib import Path

    root = Path(str(tmp_path))
    checkpoint = root / "latest.pt"
    buffer_path = root / "latest_buffer"
    buffer_path.mkdir()
    buffer = MagicMock()
    _restore_buffer(checkpoint, buffer)
    buffer.load.assert_called_once_with(buffer_path)


def test_restore_buffer_ignores_missing_dump(tmp_path: object) -> None:
    from pathlib import Path

    buffer = MagicMock()
    _restore_buffer(Path(str(tmp_path)) / "best.pt", buffer)
    buffer.load.assert_not_called()
