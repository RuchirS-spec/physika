from physika.core.expr import App, Const, Lit
from physika.utils.cic_utils.expr_utils import (
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
