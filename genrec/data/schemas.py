"""Sequence sample schema used by TIGER collate."""
from typing import List, NamedTuple


class SeqData(NamedTuple):
    user_id: int
    item_ids: List[int]
    target_ids: List[int]
    target_item: int
    history_items: List[int] = ()
