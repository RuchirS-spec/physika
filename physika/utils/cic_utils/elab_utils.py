from typing import List, Optional, Tuple
from physika.core.expr import (
    App,
    BVar,
    Const,
    Expr,
    ForallE,
    Lit,
    BinderInfo,
)

_NAT_CONST = Const("Nat", ())
_REAL_CONST = Const("Real", ())
_VEC_CONST = Const("Vec", ())
_NAT_ADD = Const("Nat.add", ())


def return_type_contains_more_elements(tp: Expr) -> bool:
    """
    Structurally recognize ``∀{m:Nat}, Vec Real m → Real → Vec Real
    (m+"n")``. Returns wether a Pi type has a body that outputs an array
    with more elements than original.

    Parameters
    ----------
    tp : Expr
        A function's registered Pi-type.

    Example
    -------
    >>> from physika.core.expr import ForallE, BinderInfo, App, BVar, Lit  # noqa: E501
    >>> from physika.utils.cic_utils.elab_utils import return_type_contains_more_elements
    >>> shape = ForallE("m", _NAT_CONST,
    ...     ForallE("x", App(App(_VEC_CONST, _REAL_CONST), BVar(0)),
    ...         ForallE("v", _REAL_CONST,
    ...             App(App(_VEC_CONST, _REAL_CONST),
    ...                 App(App(_NAT_ADD, BVar(2)), Lit(1))),
    ...             BinderInfo.DEFAULT),
    ...         BinderInfo.DEFAULT),
    ...     BinderInfo.IMPLICIT)
    >>> return_type_contains_more_elements(shape)
    True
    >>> return_type_contains_more_elements(_REAL_CONST)
    False

    """
    if not (isinstance(tp, ForallE) and tp.binder_info == BinderInfo.IMPLICIT
            and tp.binder_type == _NAT_CONST):
        return False
    lvl1 = tp.body
    if not (isinstance(lvl1, ForallE)
            and lvl1.binder_info == BinderInfo.DEFAULT and lvl1.binder_type
            == App(App(_VEC_CONST, _REAL_CONST), BVar(0))):
        return False
    lvl2 = lvl1.body
    if not (isinstance(lvl2, ForallE) and lvl2.binder_info
            == BinderInfo.DEFAULT and lvl2.binder_type == _REAL_CONST):
        return False
    ret = lvl2.body
    expected = App(App(_VEC_CONST, _REAL_CONST),
                   App(App(_NAT_ADD, BVar(2)),
                       Lit(1)))  # type: ignore[arg-type]
    return ret == expected


def flatten_loop_assigns(
        loop_body: list) -> Optional[List[List[Tuple[str, object]]]]:
    """
    Helper function to flatten loop body of ``loop_assign``/
    ``loop_tuple_unpack`` statements into ordered groups of
    ``(name, rhs_node)`` pairs.

    Parameters
    ----------
    loop_body : list
        A loop's statement list.

    Example
    -------
    >>> flatten_loop_assigns([("loop_assign", "x", ("num", 1.0))])
    [[('x', ('num', 1.0))]]
    >>> flatten_loop_assigns([("loop_if", None, [])]) is None
    True
    """
    groups: List[List[Tuple[str, object]]] = []
    for stmt in loop_body:
        if not (isinstance(stmt, tuple) and stmt):
            return None
        if stmt[0] == "loop_assign" and len(stmt) == 3:
            groups.append([(stmt[1], stmt[2])])
        elif stmt[0] == "loop_tuple_unpack" and len(stmt) == 3:
            names, expr_list_node = stmt[1], stmt[2]
            if not (isinstance(expr_list_node, tuple) and len(expr_list_node)
                    == 2 and expr_list_node[0] == "expr_list"
                    and len(expr_list_node[1]) == len(names)):
                return None
            groups.append(list(zip(names, expr_list_node[1])))
        else:
            return None
    return groups


def loop_body_assigned_names(loop_body: list) -> set:
    """
    Return a variable name that is being reassingned inside a loop body.


    Parameters
    ----------
    loop_body : list
        A loop's statement list.

    Example
    -------
    >>> from physika.utils.cic_utils.elab_utils import loop_body_assigned_names
    >>> loop_body_assigned_names([("loop_assign", "total", ("num", 1.0))])
    {'total'}
    """
    names = set()
    for stmt in loop_body or []:
        if not (isinstance(stmt, tuple) and stmt):
            continue
        tag = stmt[0]
        if tag in ("loop_assign", "loop_pluseq", "loop_index_assign_nd",
                   "loop_index_pluseq"):
            names.add(stmt[1])
        elif tag == "loop_if":
            names |= loop_body_assigned_names(stmt[2])
        elif tag == "loop_if_else":
            names |= loop_body_assigned_names(stmt[2])
            names |= loop_body_assigned_names(stmt[3])
        elif tag == "loop_for_range":
            names |= loop_body_assigned_names(stmt[4])
    return names


def body_stmts_assigned_names(stmts: list) -> set:
    """
    Return name of variables of function's body statement list that are
    reassinged.

    Parameters
    ----------
    stmts : list
        A function/branch body's statement list.

    Example
    -------
    >>> from physika.utils.cic_utils.elab_utils import body_stmts_assigned_names  # noqa: E501
    >>> body_stmts_assigned_names([("body_assign", "x", ("num", 1.0))])
    {'x'}
    """
    names = set()
    for stmt in stmts or []:
        if not (isinstance(stmt, tuple) and stmt):
            continue
        tag = stmt[0]
        if tag in ("body_decl", "body_assign", "body_zeros_decl",
                   "body_index_assign", "body_index_assign_nd",
                   "body_for_accum", "body_for_map"):
            names.add(stmt[1])
        elif tag == "body_if":
            names |= body_stmts_assigned_names(stmt[2])
        elif tag == "body_if_else":
            names |= body_stmts_assigned_names(stmt[2])
            names |= body_stmts_assigned_names(stmt[3])
        elif tag == "body_for":
            names |= loop_body_assigned_names(stmt[2])
        elif tag == "body_for_range":
            names |= loop_body_assigned_names(stmt[4])
    return names


def collect_calls_to(target_names: set, body_node,
                     statements: list) -> List[Tuple[str, list]]:
    """
    Return ``("call", name, args)`` nodes from ``body_node`` and
    ``statements``. Used to find self and mutual recursive calls for
    termination checking.

    Parameters
    ----------
    target_names : set
        Recursive callee name to look for.
    body_node :
        Function's recursive expression as AST Node.
    statements : list
        A function's statement list.

    Example
    -------
    >>> from physika.utils.cic_utils.elab_utils import collect_calls_to
    >>> collect_calls_to({"f"}, ("call", "f", [("var", "x")]), [])
    [('f', [('var', 'x')])]
    """
    found = []

    def walk(node: object) -> None:
        """
        Recurse into node and append ``call`` to target_names.

        Parameters
        ----------
        node : object
            AST node (tuple, list, or leaf) to search for recursive calls.
        """
        if isinstance(node, tuple) and node:
            if node[0] == "call" and node[1] in target_names:
                found.append((node[1], node[2] if len(node) > 2 else []))
            for child in node[1:]:
                walk(child)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(body_node)
    for stmt in statements or []:
        walk(stmt)
    return found


def arg_decreases(arg: Tuple, param_name: str, param_type: str) -> bool:
    """
    Checks if ``arg`` is the syntactic shape ``param_name - positive_constant``
    . A int: ``ℕ` parameter requires the constant to be 1. A ``ℝ`` typed
    parameter allows any positive constant.

    Parameters
    ----------
    arg : Tuple
        Recursive call's argument.
    param_name : str
        Recursive parameter's name.
    param_type : str
        Recursive parameter's declared type (``"ℕ"``, ``"ℝ"``, ...).

    Example
    -------
    >>> from physika.utils.cic_utils.elab_utils import arg_decreases
    >>> arg_decreases(("sub", ("var", "n"), ("num", 1.0)), "n", "ℕ")
    True
    >>> arg_decreases(("sub", ("var", "n"), ("num", 2.0)), "n", "ℕ")
    False
    """
    if not (isinstance(arg, tuple) and len(arg) == 3 and arg[0] == "sub"
            and isinstance(arg[1], tuple) and arg[1][0] == "var" and arg[1][1]
            == param_name and isinstance(arg[2], tuple) and arg[2][0] == "num"
            and isinstance(arg[2][1], (int, float)) and arg[2][1] > 0):
        return False
    if param_type == "ℕ":
        return arg[2][1] == 1
    return True
