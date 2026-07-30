from physika.core.expr import (Expr, BVar, FVar, App, Lam, ForallE, LetE,
                               MData, Proj)
from typing import Tuple, List


def get_app_fn(e: Expr) -> Expr:
    """
    Return the function at the top of an application (``App``) chain.

    Parameters
    ----------
    e : Expr
        An application chain. If ``e`` is not  an ``App`` instance,
        returns ``e`` as is.

    Returns
    -------
    Expr
        Head of the chain.

    Examples
    --------
    >>> from physika.utils.cic_utils.expr_utils import get_app_fn
    >>> from physika.core.expr import App, Const, Lit
    >>> add = Const("Nat.add", ())
    >>> call = App(App(add, Lit(2)), Lit(3))  # equivalent to Nat.add 2 3
    >>> get_app_fn(call) == add
    True
    """
    while isinstance(e, App):
        e = e.func
    return e


def get_app_args(e: Expr) -> List[Expr]:
    """
    Return the arguments of an application chain from left to right.

    Parameters
    ----------
    e : Expr
        An application chain. If ``e`` is not ``App``, an empty list is
        returned.

    Returns
    -------
    List[Expr]
        Function's arguments.

    Examples
    --------
    >>> from physika.utils.cic_utils.expr_utils import get_app_args
    >>> from physika.core.expr import App, Const, Lit
    >>> add = Const("Nat.add", ())
    >>> call = App(App(add, Lit(2)), Lit(3))
    >>> get_app_args(call)
    [Lit(val=2), Lit(val=3)]
    >>> get_app_args(add)
    []
    """
    args = []
    while isinstance(e, App):
        args.append(e.arg)
        e = e.func
    args.reverse()
    return args


def get_app_fn_args(e: Expr) -> Tuple[Expr, List[Expr]]:
    """
    Return both the head and the arguments of an application chain in
    one call. Is the combination of `get_app_fn` and `get_app_args`.

    Parameters
    ----------
    e : Expr
        An application chain.

    Returns
    -------
    Tuple[Expr, List[Expr]]
        Head of function and its arguments.

    Examples
    --------
    >>> from physika.utils.cic_utils.expr_utils import get_app_fn_args
    >>> from physika.core.expr import App, Const, Lit
    >>> add = Const("Nat.add", ())
    >>> call = App(App(add, Lit(2)), Lit(3))
    >>> head, args = get_app_fn_args(call)
    >>> head == add
    True
    >>> args
    [Lit(val=2), Lit(val=3)]
    """
    return get_app_fn(e), get_app_args(e)


def abstract_fvars(expr: Expr, fvars: List[FVar]) -> Expr:
    """
    Replace ``fvars[i]`` with ``BVar(i)``, inspired by Lean 4 locally
    nameless approach.

    Using de Bruijn indices (``BVar``) in local contexts (e.g. inside
    a binder) can lead to index shifts each time a binder is accesed.
    In Physika, inspired by Lean 4, when opening a binder, its
    ``BVar(0)`` is transformed into a globally unique ``FVar``, the
    body is elaborated with that name (does not read de Bruijn
    indices). Once the binder is fully elaborated, we go back from
    ``FVar`` to the original ``BVar`` and  ``Lam``/``ForallE`` node.

    ``abstract_fvars` produces the ``FVar`` to ``BVar`` transformation.

    Parameters
    ----------
    expr : Expr
        Expression node referencing ``fvars``.
    fvars : List[FVar]
        Free variables to abstract.

    Examples
    --------
    >>> from physika.utils.cic_utils.expr_utils import abstract_fvars  # noqa: E501
    >>> from physika.core.expr import FVar, FVarId, Const, App
    >>> x, y = FVar(FVarId("x.0")), FVar(FVarId("y.1"))
    >>> body = App(App(Const("Real.add", ()), x), y)
    >>> abstract_fvars(body, [x, y])
    App(func=App(func=Const(name='Real.add', levels=()), arg=BVar(idx=0)), arg=BVar(idx=1))
    """
    if not fvars:
        return expr
    ids = [fv.id.id for fv in fvars]
    return abstract(expr, ids, 0)


def abstract(e: Expr, fvar_ids: List[str], depth: int) -> Expr:
    """
    Convert a set of free variables (`FVar`) to bound variables (`BVar`.

    `abstract` recurse over `e` Expression tree and when an `FVar` is found (and
    its id is in `fvar_ids`), replace with `BVar(depth + i)`. `i` refers to the
    variable's position in the list where `FVar` was found.  `depth` is a counter
    of how many binder scopes (Lam/ForallE/LetE bodies) we are currently at.
    Other node type recurses into its children, threading depth through
    unchanged.

    Parameters
    ----------
    e: Expr
        Expression node to check for FVar occurrencies.
    fvar_ids: List[str]
        List of `FVar` id's as strings present in a given context
    depth: int
        Integer number that represents the current depth `abstract` is.

    Example
    -------
    >>> from physika.utils.cic_utils.expr_utils import abstract  # noqa: E501
    >>> from physika.core.expr import FVar, FVarId, Const, App, Lam, BinderInfo
    >>> x = FVar(FVarId("x.0"))
    >>> term = App(Const("Real.mul", ()), x)
    >>> abstract(term, ["x.0"], 0)
    App(func=Const(name='Real.mul', levels=()), arg=BVar(idx=0))
    >>> nested = Lam("y", Const("Real", ()), App(Const("Real.mul", ()), x), BinderInfo.DEFAULT)
    >>> abstract(nested, ["x.0"], 0)
    Lam(binder_name='y', binder_type=Const(name='Real', levels=()), body=App(func=Const(name='Real.mul', levels=()), arg=BVar(idx=1)), binder_info=<BinderInfo.DEFAULT: 1>)
    """
    if isinstance(e, FVar):
        try:
            i = fvar_ids.index(e.id.id)
            return BVar(depth + i)
        except ValueError:
            return e  # not in the list, keep as FVar
    elif isinstance(e, App):
        nf = abstract(e.func, fvar_ids, depth)
        na = abstract(e.arg, fvar_ids, depth)
        return App(nf, na) if (nf is not e.func or na is not e.arg) else e
    elif isinstance(e, Lam):
        nt = abstract(e.binder_type, fvar_ids, depth)
        nb = abstract(e.body, fvar_ids, depth + 1)
        return Lam(e.binder_name, nt, nb, e.binder_info) if (
            nt is not e.binder_type or nb is not e.body) else e
    elif isinstance(e, ForallE):
        nt = abstract(e.binder_type, fvar_ids, depth)
        nb = abstract(e.body, fvar_ids, depth + 1)
        return ForallE(e.binder_name, nt, nb, e.binder_info) if (
            nt is not e.binder_type or nb is not e.body) else e
    elif isinstance(e, LetE):
        nt = abstract(e.type, fvar_ids, depth)
        nv = abstract(e.value, fvar_ids, depth)
        nb = abstract(e.body, fvar_ids, depth + 1)
        return LetE(e.binder_name, nt, nv, nb, e.non_dep) if (
            nt is not e.type or nv is not e.value or nb is not e.body) else e
    elif isinstance(e, MData):
        ne = abstract(e.expr, fvar_ids, depth)
        return MData(e.kvs, ne) if ne is not e.expr else e
    elif isinstance(e, Proj):
        ne = abstract(e.expr, fvar_ids, depth)
        return Proj(e.type_name, e.idx, ne) if ne is not e.expr else e
    else:  # BVar, MVar, Sort, Const, Lit
        return e
