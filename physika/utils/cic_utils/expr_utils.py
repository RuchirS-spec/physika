from physika.core.expr import Expr, App
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
