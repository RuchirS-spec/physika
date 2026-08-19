from physika.core.expr import App, BVar, BinderInfo, ForallE, Lit
from physika.utils.cic_utils.elab_utils import (
    arg_decreases,
    body_stmts_assigned_names,
    collect_calls_to,
    return_type_contains_more_elements,
    flatten_loop_assigns,
    loop_body_assigned_names,
)
from physika.core.elab.dim_typespec import (_NAT_CONST, _REAL_CONST,
                                            _VEC_CONST, _NAT_ADD)


def append_shape() -> ForallE:
    """
    Pi-type for append function:
        ``∀{m:Nat}, Vec Real m → Real → Vec Real (m+1)``
    """
    return ForallE(
        "m",
        _NAT_CONST,
        ForallE(
            "x",
            App(App(_VEC_CONST, _REAL_CONST), BVar(0)),
            ForallE(
                "v",
                _REAL_CONST,
                App(App(_VEC_CONST, _REAL_CONST),
                    App(App(_NAT_ADD, BVar(2)), Lit(1))),
                BinderInfo.DEFAULT,
            ),
            BinderInfo.DEFAULT,
        ),
        BinderInfo.IMPLICIT,
    )


class TestContainsMoreElements:
    """
    Tests for ``return_type_contains_more_elements``.
    """

    def test_return_type_contains_more_elements_than_initial_params(self):
        """
        Tests a Pi-type contains the right structure when doing an append
        operation, which mean the return type contains more elements than its
        params.
        """
        assert return_type_contains_more_elements(append_shape()) is True

        # non Pi type should be rejected
        assert return_type_contains_more_elements(_REAL_CONST) is False

        # shpould catch outer binder wrong type
        shape = ForallE("m", _NAT_CONST, _REAL_CONST, BinderInfo.DEFAULT)
        assert return_type_contains_more_elements(shape) is False

        # returns same shape in of params is wrong
        shape = ForallE(
            "m",
            _NAT_CONST,
            ForallE(
                "x",
                App(App(_VEC_CONST, _REAL_CONST), BVar(0)),
                ForallE(
                    "v",
                    _REAL_CONST,
                    App(App(_VEC_CONST, _REAL_CONST), BVar(2)),
                    BinderInfo.DEFAULT,
                ),
                BinderInfo.DEFAULT,
            ),
            BinderInfo.IMPLICIT,
        )
        assert return_type_contains_more_elements(shape) is False


class TestFlattenLoopandBodyAssignments:
    """
    Tests for ``flatten_loop_assigns`` and ``loop_body_assigned_names``.
    """

    def test_flatten_loop_assingments(self):
        """
        Checks a loop assingment is properly flattened
        """
        # single loop assignemt
        assert flatten_loop_assigns([("loop_assign", "x", ("num", 1.0))
                                     ]) == [[("x", ("num", 1.0))]]

        result = flatten_loop_assigns([
            ("loop_tuple_unpack", ["a", "b"], ("expr_list", [("var", "x"),
                                                             ("var", "y")])),
        ])
        assert result == [[("a", ("var", "x")), ("b", ("var", "y"))]]

        # checks loop features, like tuple unpack are flattened properly
        result = flatten_loop_assigns([
            ("loop_assign", "x", ("num", 1.0)),
            ("loop_tuple_unpack", ["a", "b"], ("expr_list", [("var", "x"),
                                                             ("var", "y")])),
        ])
        assert result == [
            [("x", ("num", 1.0))],
            [("a", ("var", "x")), ("b", ("var", "y"))],
        ]

    def test_body_loop_assigned_names(self):
        """
        Checks body loop assignemnt for variable name is handlerd properly
        """
        assert loop_body_assigned_names([("loop_assign", "total", ("num", 1.0))
                                         ]) == {"total"}

        # body var names should be catvhed
        names = loop_body_assigned_names([
            ("loop_pluseq", "total", ("num", 1.0)),
            ("loop_index_assign_nd", "arr", [], ("num", 2.0)),
            ("loop_index_pluseq", "acc", [], ("num", 3.0)),
        ])
        assert names == {"total", "arr", "acc"}

        # var anmes in if-else branches
        names = loop_body_assigned_names([
            ("loop_if_else", ("cond", ), [("loop_assign", "a", ("num", 1.0))],
             [("loop_pluseq", "b", ("num", 2.0))]),
            ("loop_for_range", "k", 0, 10, [("loop_index_assign_nd", "c", [],
                                             ("num", 3.0))]),
        ])
        assert names == {"a", "b", "c"}

    def test_empty_body_returns_empty_set(self):
        assert loop_body_assigned_names([]) == set()


class TestBodyStmtsAssignedNames:
    """
    Tests for ``body_stmts_assigned_names``.
    """

    def test_body_assign(self):
        """
        Test body assingments variable names
        """
        assert body_stmts_assigned_names([("body_assign", "x", ("num", 1.0))
                                          ]) == {"x"}

        # support body assignment calls
        stmts = [
            ("body_decl", "a", "ℝ", ("num", 0.0)),
            ("body_zeros_decl", "b", ("tensor", [])),
            ("body_index_assign", "c", [], ("num", 0.0)),
            ("body_index_assign_nd", "d", [], ("num", 0.0)),
            ("body_for_accum", "e", [], []),
            ("body_for_map", "f", [], []),
        ]
        assert body_stmts_assigned_names(stmts) == {
            "a", "b", "c", "d", "e", "f"
        }

        # if else recursion
        names = body_stmts_assigned_names([
            ("body_if", ("cond", ), [("body_assign", "x", ("num", 1.0))]),
            ("body_if_else", ("cond", ), [("body_assign", "y", ("num", 1.0))],
             [("body_assign", "z", ("num", 2.0))]),
        ])
        assert names == {"x", "y", "z"}

        names = body_stmts_assigned_names([
            ("body_for", "i", [("loop_pluseq", "total", ("num", 1.0))], []),
            ("body_for_range", "j", 0, 10, [("loop_assign", "other", ("num",
                                                                      2.0))]),
        ])
        assert names == {"total", "other"}


class TestCollectCallsTo:
    """
    Tests for ``collect_calls_to``.
    """

    def test_call_in_body(self):
        """
        Chcks collect funciton calls in a AST node
        """
        # should return an empty list
        assert collect_calls_to({"f"}, ("num", 0.0), []) == []

        assert collect_calls_to({"f"}, ("call", "f", [("var", "x")]),
                                []) == [("f", [("var", "x")])]

        # nested call
        found = collect_calls_to({"f"}, ("add", ("call", "f", [("var", "x")]),
                                         ("num", 1.0)), [])
        assert found == [("f", [("var", "x")])]

        # call in statements not just in return
        found = collect_calls_to({"g"}, ("num", 0.0),
                                 [("call", "g", [("var", "y")])])
        assert found == [("g", [("var", "y")])]

        # multipple target names
        found = collect_calls_to(
            {"f", "g"},
            ("add", ("call", "f", [("var", "x")]), ("call", "h", [])),
            [("call", "g", [("var", "y")])],
        )
        assert found == [("f", [("var", "x")]), ("g", [("var", "y")])]


class TestArgDecreases:
    """
    Tests for ``arg_decreases``.
    """

    def test_param_decreases_rec_call(self):
        arg = ("sub", ("var", "n"), ("num", 1.0))
        assert arg_decreases(arg, "n", "ℕ") is True

        # should be decremented by 1 if Nats
        arg = ("sub", ("var", "n"), ("num", 2.0))
        assert arg_decreases(arg, "n", "ℕ") is False

        # Real type decrement should be true
        arg = ("sub", ("var", "n"), ("num", 5.0))
        assert arg_decreases(arg, "n", "ℝ") is True

        # decrementts using negative values should return Flase
        arg = ("sub", ("var", "n"), ("num", -1.0))
        assert arg_decreases(arg, "n", "ℕ") is False
