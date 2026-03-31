from __future__ import annotations

"""I/O helpers for SemEval-2015 pair data."""

from pathlib import Path
from typing import Optional, Tuple, List, Set


def parse_label(judge: Optional[str]) -> Optional[bool]:
    """Parse a raw judge field into a boolean label when possible."""
    if judge is None:
        return None
    if len(judge) == 0:
        return None

    # labelled by Amazon Mechanical Turk in format like "(2,3)"
    if judge[0] == "(":
        n_yes = eval(judge)[0]
        if n_yes >= 3:
            return True
        if n_yes <= 1:
            return False
        return None

    # labelled by expert in format like "2"
    if judge[0].isdigit():
        n_yes = int(judge[0])
        if n_yes >= 4:
            return True
        if n_yes <= 2:
            return False
        return None

    return None


def read_pair_data(filename: Path) -> Tuple[List[Tuple[Optional[bool], str, str, str]], Set[str]]:
    """Read pair data and return (label, source, target, trend_id) rows and trend ids."""
    data: List[Tuple[Optional[bool], str, str, str]] = []
    trends: Set[str] = set()

    for line in filename.open():
        line = line.strip()
        if len(line.split("\t")) == 7:
            trendid, _trendname, origsent, candsent, judge, _origtag, _candtag = line.split("\t")
        elif len(line.split("\t")) == 6:
            trendid, _trendname, origsent, candsent, _origtag, _candtag = line.split("\t")
            judge = None
        else:
            continue

        trends.add(trendid)
        label = parse_label(judge)
        data.append((label, origsent, candsent, trendid))

    return data, trends
