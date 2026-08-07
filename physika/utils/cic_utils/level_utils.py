from physika.core.level import (Level, LZero, LSucc, LMax, LIMax, LMVar)
from typing import Optional


def mk_level_max(l1: Level, l2: Level) -> Level:
    """
    Compares two universe levels (``l1``, ``l2``) and returns the maximum
    level between them.

    Parameters
    ----------
    l1 : Level
        The first universe level to compare.
    l2 : Level
        The second universe level to compare.

    Examples
    --------
    >>> from physika.core.level import LZero, LSucc
    >>> from physika.utils.cic_utils.level_utils import mk_level_max
    >>> mk_level_max(LSucc(LZero()), LZero())
    LSucc(pred=LZero())
    >>> mk_level_max(LSucc(LZero()), LSucc(LSucc(LZero())))
    LMax(l1=LSucc(pred=LZero()), l2=LSucc(pred=LSucc(pred=LZero())))
    """
    if isinstance(l1, LZero):
        return l2
    if isinstance(l2, LZero):
        return l1
    if l1 == l2:
        return l1
    return LMax(l1, l2)


def mk_level_imax(l1: Level, l2: Level) -> Level:
    """
    Compares two universe levels (``l1``, ``l2``) and returns the maximum
    level between them, with the special case that if ``l2`` is 0, the result
    is 0.

    Parameters
    ----------
    l1 : Level
        The first universe level to compare.
    l2 : Level
        The second universe level to compare.

    Examples
    --------
    >>> from physika.core.level import LZero, LSucc
    >>> from physika.utils.cic_utils.level_utils import mk_level_imax
    >>> mk_level_imax(LSucc(LZero()), LZero())
    LZero()
    >>> mk_level_imax(LSucc(LZero()), LSucc(LZero()))
    LSucc(pred=LZero())
    """
    if isinstance(l2, LZero):
        return LZero()
    if isinstance(l1, LZero):
        return l2
    if l1 == l2:
        return l1
    return LIMax(l1, l2)


def level_has_mvar(lvl: Level, mvar_id: Optional[str] = None) -> bool:
    """
    Occurs-check for universe-level metavariables. Check if ``l``contains an
    unsolved LMVar placeholder inside it.

    If ``mvar_id`` is None, ``level_has_mvar`` checks if any level is resolved
    before evaluating. When ``mvar_id`` is provided, ``level_has_mvar`` checks
    if ``LMVar`` with the specified id is present.

    Parameters
    ----------
    l : Level
        Level to search for a metavariable node.
    mvar_id : Optional[str], default None
        If given, only match an LMVar with this specific id; otherwise
        match any LMVar.

    Examples
    --------
    >>> from physika.core.level import LZero, LSucc, LMVar, LParam
    >>> level_has_mvar(LMVar("m1"))
    True
    >>> level_has_mvar(LSucc(LMVar("m1")))
    True
    >>> level_has_mvar(LParam("u"))
    False
    >>> level_has_mvar(LMVar("m1"), "m1")
    True
    >>> level_has_mvar(LMVar("m1"), "m2")
    False
    """
    if isinstance(lvl, LMVar):
        return mvar_id is None or lvl.id == mvar_id
    elif isinstance(lvl, LSucc):
        return level_has_mvar(lvl.pred, mvar_id)
    elif isinstance(lvl, (LMax, LIMax)):
        return level_has_mvar(lvl.l1, mvar_id) or level_has_mvar(
            lvl.l2, mvar_id)
    return False
