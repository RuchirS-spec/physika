from physika.core.level import LZero, LSucc, LMax, LIMax, LMVar
from physika.utils.cic_utils.level_utils import (
    mk_level_max,
    mk_level_imax,
    level_has_mvar,
)


class TestMkLevelMax:
    """
    Tests for ``mk_level_max``
    """

    def test_lmax_basic(self):
        """
        Basic tests checking correct lmax rule.
        """
        lvl = LSucc(LZero())
        assert mk_level_max(LZero(), lvl) == lvl

        # max(lvl, 0) returnx lvl.
        lvl = LSucc(LZero())
        assert mk_level_max(lvl, LZero()) == lvl

        # max(lvl, lvl) returns lvl
        lvl = LSucc(LSucc(LZero()))
        assert mk_level_max(lvl, lvl) == lvl

        # max(0, 0) returns LZero()
        assert mk_level_max(LZero(), LZero()) == LZero()

        l1 = LSucc(LZero())
        l2 = LSucc(LSucc(LZero()))
        # builds an LMax node
        assert mk_level_max(l1, l2) == LMax(l1, l2)


class TestMkLevelImax:
    """
    Basic tests checking correct lmax rule.
    """

    def test_level_imax_basics(self):
        """
        Checks different level configurations for imax rule.
        """
        # imax(lvl, 0) always returns 0
        assert mk_level_imax(LSucc(LZero()), LZero()) == LZero()

        # imax(0, lvl) returns lvl
        lvl = LSucc(LZero())
        assert mk_level_imax(LZero(), lvl) == lvl

        # imax(lvl, lvl) returns lvl
        lvl = LSucc(LSucc(LZero()))
        assert mk_level_imax(lvl, lvl) == lvl

        # imax(l1, l2) gives an LIMax node.
        l1 = LSucc(LZero())
        l2 = LSucc(LSucc(LZero()))
        assert mk_level_imax(l1, l2) == LIMax(l1, l2)


class TestLevelHasMvar:
    """
    Tests for ``level_has_mvar``
    """

    def test_find_mvar_without_target_id(self):
        """
        Any LMVar is found when no specific mvar_id is given.
        """
        assert level_has_mvar(LMVar("m1")) is True

        # recurse through LSucc node
        assert level_has_mvar(LSucc(LMVar("m1"))) is True

    def test_mvar_nested_max_levels(self):
        """
        Tests for nested LMVars within LMax and LIMax nodes.
        """
        # Searches both operands of LMax
        assert level_has_mvar(LMax(LZero(), LMVar("m1"))) is True
        assert level_has_mvar(LMax(LMVar("m1"), LZero())) is True

        # search recurses into both operands of LIMax.

        assert level_has_mvar(LIMax(LZero(), LMVar("m1"))) is True
        assert level_has_mvar(LIMax(LMVar("m1"), LZero())) is True

    def test_matching_specific_mvar_id(self):
        """
        Looks for LMVar with a specific given id.
        """
        assert level_has_mvar(LMVar("m1"), "m1") is True
        assert level_has_mvar(LMVar("m1"), "m2") is False

        # Looks foe MVar in LMax node
        lvl = LMax(LMVar("m1"), LMVar("m2"))
        assert level_has_mvar(lvl, "m1") is True
        assert level_has_mvar(lvl, "m2") is True
        assert level_has_mvar(lvl, "m3") is False
