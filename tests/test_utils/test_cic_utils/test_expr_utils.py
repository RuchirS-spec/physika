from physika.core.expr import (
    App,
    BVar,
    Const,
    ForallE,
    FVar,
    FVarId,
    Lam,
    LetE,
    Lit,
    MData,
    Proj,
    BinderInfo,
)
from physika.utils.cic_utils.expr_utils import (
    abstract,
    abstract_fvars,
    get_app_args,
    get_app_fn,
    get_app_fn_args,
)


class TestGetAppFn:
    """
    Tests for ``get_app_fn``
    """

    def test_get_app_fn(self):
        """
        Checks the head of a function application.s
        """
        add = Const("Nat.add", ())
        call = App(add, Lit(2))

        assert get_app_fn(call) == add

    def test_get_app_fn_multi_arg(self):
        """
        Checks the head of a curried application chain.
        """
        add = Const("Nat.add", ())
        call = App(App(add, Lit(2)), Lit(3))

        assert get_app_fn(call) == add

        # 4 level App nesting (``Vec.cons alpha n hd tl``)
        cons = Const("Vec.cons", ())
        chain = App(App(App(App(cons, Const("Real", ())), Lit(2)), Lit(1.0)),
                    Const("u", ()))

        assert get_app_fn(chain) == cons


class TestGetAppArgs:
    """
    Tests for ``get_app_args``
    """

    def test_get_app_args(self):
        """
        Checks function application with one argument
        returns it.
        """
        add = Const("Nat.add", ())
        call = App(add, Lit(2))

        assert get_app_args(call) == [Lit(2)]

    def test_get_app_multi_args(self):
        """
        Checks 2 arguments function application returns properly
        (from left to right).
        """
        add = Const("Nat.add", ())
        call = App(App(add, Lit(2)), Lit(3))

        assert get_app_args(call) == [Lit(2), Lit(3)]

    def test_get_app_args_deeply_nested(self):
        """
        Checks a 4 argument chain function application.
        """
        cons = Const("Vec.cons", ())
        alpha = Const("Real", ())
        n = Lit(2)
        hd = Lit(1.0)
        tl = Const("u", ())
        chain = App(App(App(App(cons, alpha), n), hd), tl)

        assert get_app_args(chain) == [alpha, n, hd, tl]


class TestGetAppFnArgs:
    """
    Tests for ``get_app_fn_args``
    """

    def test_get_app_fn(self):
        """
        Checks ``get_app_fn_args`` returns the same as
        applying ``get_app_fn`` and ``get_app_args`` separately.
        """
        add = Const("Nat.add", ())
        call = App(App(add, Lit(2)), Lit(3))

        head, args = get_app_fn_args(call)

        assert head == get_app_fn(call)
        assert args == get_app_args(call)
        assert head == add
        assert args == [Lit(2), Lit(3)]


class TestAbstract:
    """
    Tests for ``abstract``
    """

    def test_abstract_fvar_hit(self):
        """
        Checks an ``FVar`` whose id is in ``fvar_ids`` becomes a
        ``BVar`` at ``depth + i``.
        """
        x = FVar(FVarId("x.0"))

        assert abstract(x, ["x.0"], 0) == BVar(0)

    def test_abstract_fvar_miss(self):
        """
        Checks an ``FVar`` whose id is not in ``fvar_ids`` is left
        unchanged.
        """
        x = FVar(FVarId("x.0"))

        assert abstract(x, ["other.9"], 0) == x

    def test_abstract_multiple_fvars_ordering(self):
        """
        Checks each ``FVar`` gets the ``BVar`` index matching its own
        position in ``fvar_ids``.
        """
        x = FVar(FVarId("x.0"))
        y = FVar(FVarId("y.1"))
        term = App(App(Const("f", ()), x), y)

        assert abstract(term, ["x.0", "y.1"],
                        0) == App(App(Const("f", ()), BVar(0)), BVar(1))

    def test_abstract_app_recurses_both_sides(self):
        """
        Checks both the function and argument side of an ``App`` are
        walked.
        """
        x = FVar(FVarId("x.0"))
        term = App(Const("f", ()), x)

        assert abstract(term, ["x.0"], 0) == App(Const("f", ()), BVar(0))

    def test_abstract_lam_increments_depth_only_in_body(self):
        """
        Checks a ``Lam``'s ``binder_type`` is abstracted at the same
        depth as the ``Lam`` itself, while its ``body`` is abstracted
        one level deeper — the type isn't under the Lam's own scope,
        only the body is.
        """
        x = FVar(FVarId("x.0"))
        lam = Lam("n", x, x, BinderInfo.DEFAULT)

        result = abstract(lam, ["x.0"], 0)

        assert result.binder_type == BVar(0)
        assert result.body == BVar(1)

    def test_abstract_forall_increments_depth_only_in_body(self):
        """
        Checks ``ForallE`` follows the same depth rule as ``Lam``.
        """
        x = FVar(FVarId("x.0"))
        forall = ForallE("n", x, x, BinderInfo.DEFAULT)

        result = abstract(forall, ["x.0"], 0)

        assert result.binder_type == BVar(0)
        assert result.body == BVar(1)

    def test_abstract_lete_increments_depth_only_in_body(self):
        """
        Checks a ``LetE``'s ``type`` and ``value`` are abstracted at
        the same depth as the ``LetE`` itself, while its ``body`` is
        one level deeper.
        """
        x = FVar(FVarId("x.0"))
        let = LetE("n", x, x, x, False)

        result = abstract(let, ["x.0"], 0)

        assert result.type == BVar(0)
        assert result.value == BVar(0)
        assert result.body == BVar(1)

    def test_abstract_mdata(self):
        """
        Checks the wrapped expression inside ``MData`` is abstracted.
        """
        x = FVar(FVarId("x.0"))
        wrapped = MData((("line", 1), ), x)

        assert abstract(wrapped, ["x.0"], 0) == MData((("line", 1), ), BVar(0))

    def test_abstract_proj(self):
        """
        Checks the struct instance inside ``Proj`` is abstracted.
        """
        x = FVar(FVarId("x.0"))
        proj = Proj("Ray", 0, x)

        assert abstract(proj, ["x.0"], 0) == Proj("Ray", 0, BVar(0))

    def test_abstract_leaf_nodes_unchanged(self):
        """
        Checks nodes that can never contain an ``FVar`` (``BVar``,
        ``Const``) are returned as the same object, not a copy.
        """
        b = BVar(0)
        real = Const("Real", ())

        assert abstract(b, ["x.0"], 0) is b
        assert abstract(real, ["x.0"], 0) is real

    def test_abstract_depth_tracks_binder_scopes_not_call_depth(self):
        """
        Checks depth only increases when the walk actually crosses
        into a binder's body — passing through an ``App`` on the way
        to a nested ``Lam`` does not, by itself, add to depth.
        """
        x = FVar(FVarId("x.0"))
        real = Const("Real", ())
        # App(f, Lam(n : Real, x)) -- one binder (the Lam) encloses x,
        # even though reaching x also means recursing through the App.
        wrapped = App(Const("f", ()), Lam("n", real, x, BinderInfo.DEFAULT))

        result = abstract(wrapped, ["x.0"], 0)

        assert result == App(Const("f", ()),
                             Lam("n", real, BVar(1), BinderInfo.DEFAULT))


class TestAbstractFvars:
    """
    Tests for ``abstract_fvars``
    """

    def test_abstract_fvars_basic(self):
        """
        Checks a single ``FVar`` is abstracted to ``BVar(0)``.
        """
        x = FVar(FVarId("x.0"))
        term = App(Const("f", ()), x)

        assert abstract_fvars(term, [x]) == App(Const("f", ()), BVar(0))

    def test_abstract_fvars_multiple_ordering(self):
        """
        Checks ``fvars[0]`` becomes the innermost ``BVar(0)`` and
        ``fvars[-1]`` the outermost.
        """
        x = FVar(FVarId("x.0"))
        y = FVar(FVarId("y.1"))
        body = App(App(Const("g", ()), x), y)

        assert abstract_fvars(body,
                              [x, y]) == App(App(Const("g", ()), BVar(0)),
                                             BVar(1))

    def test_abstract_fvars_empty_list_is_noop(self):
        """
        Checks an empty ``fvars`` list returns the exact same object,
        untouched — there's no new binder for any FVar to be
        abstracted into.
        """
        x = FVar(FVarId("x.0"))
        body = App(Const("f", ()), x)

        assert abstract_fvars(body, []) is body

    def test_abstract_fvars_delegates_to_abstract(self):
        """
        Checks ``abstract_fvars`` produces the same result as calling
        ``abstract`` directly with the matching ids and depth 0.
        """
        x = FVar(FVarId("x.0"))
        y = FVar(FVarId("y.1"))
        body = App(App(Const("g", ()), x), y)

        assert abstract_fvars(body,
                              [x, y]) == abstract(body, ["x.0", "y.1"], 0)
