from __future__ import annotations

import pytest

from src.config import Settings


def test_settings_reject_chunk_overlap_equal_to_chunk_size():
    with pytest.raises(ValueError, match="chunk_overlap must be smaller than chunk_size"):
        Settings(chunk_size=10, chunk_overlap=10)


@pytest.mark.parametrize("top_k", [0, -1, 51])
def test_settings_reject_invalid_top_k(top_k: int):
    with pytest.raises(ValueError):
        Settings(top_k=top_k)
