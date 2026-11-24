"""Source membership and parent integrity are independent of arriving IO."""
import hashlib
from types import SimpleNamespace

import pytest

from chakra.src.converter.operator_io import OperandIO
from chakra.src.converter.operator_regions import OperatorRegions


def sparse_region():
    source_rows = ((0, 1), (0, 2), (0, 4), (1, 1), (1, 3), (1, 4))
    digest = hashlib.sha256(b"0/1:\n0/2:1\n0/4:2\n1/1:\n1/3:1\n1/4:3\n").hexdigest()
    rows = [dict(rank=rank, row=row, present=True,
                 parents=[] if row == 1 else ([1] if row < 4 else [2 + rank]))
            for rank, row in source_rows]
    io = {key: OperandIO(str(key[1]), (), ()) for key in source_rows}
    return dict(stage=0, entry=1, exit=4, rows=rows, source_owner_sha256=digest), io


def test_sparse_visit_skips_gaps_and_keeps_stage_and_prefix_offsets():
    region, io = sparse_region()
    bound = OperatorRegions([region], io, 6, sparse=True)
    bound.validate_rows([(0, 5)], 1, 2, 0)
    assert list(bound.stage_indices(0, 0, 5, 1)) == [0, 1, 3, 4]
    assert list(bound.stage_indices(1, 0, 5, 1)) == [0, 2, 3, 4]
    event = SimpleNamespace(npu_id=1, releases=["2"])
    with pytest.raises(ValueError, match="absent Page anchor rank=1 row=2"):
        bound.validate_movements(SimpleNamespace(events=[event], prior_events=[]), io,
            layers=[SimpleNamespace(name=str(i)) for i in range(6)])


@pytest.mark.parametrize("defect,reason", [
    ("missing_io", "missing original IO"), ("both_removed", "Source owner digest"),
    ("parent", "Source owner digest"), ("foreign_rank", "Source owner digest"),
    ("budget", "budget"), ("dense", "unknown fields"),
])
def test_sparse_rejects_lost_work_before_emission(defect, reason):
    region, io = sparse_region()
    if defect in ("missing_io", "both_removed"):
        del io[0, 2]
    if defect == "both_removed":
        region["rows"].pop(1)
    if defect == "parent":
        region["rows"][-1]["parents"] = [2]
    if defect == "foreign_rank":
        for row in region["rows"][3:]:
            io[3, row["row"]] = io.pop((1, row["row"]))
            row["rank"] = 3
    with pytest.raises(ValueError, match=reason):
        OperatorRegions([region], io, 5 if defect == "budget" else 6, sparse=defect != "dense")
