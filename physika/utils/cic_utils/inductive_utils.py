from physika.core.expr import Const, Expr, Lam, App, ForallE, MData, LetE, Proj
from physika.utils.cic_utils.expr_utils import get_app_fn_args
from typing import Optional, Callable
from physika.core.inductive import InductiveDecl


def name_appears(name: str, expr: Expr) -> bool:
    """
    Recursively checks if an inductive type ``Const(name)`` appears inside
    an `Expr` when declaring an inductive type.

    Parameters
    ----------
    name : str
        Name of inductive type to search.
    expr : Expr
        Expression node to check for an inductive type.

    Examples
    --------
    >>> from physika.utils.cic_utils.inductive_utils import name_appears
    >>> from physika.core.expr import Const, ForallE
    >>> nat = Const("Nat", ())
    >>> succ_type = ForallE("n", nat, nat)  # Nat -> Nat
    >>> name_appears("Nat", succ_type)
    True
    >>> name_appears("Bool", succ_type)
    False
    """
    if isinstance(expr, Const):
        return expr.name == name
    if isinstance(expr, App):
        # Check appearance of inductive type inside a function application
        return name_appears(name, expr.func) or name_appears(name, expr.arg)
    if isinstance(expr, (Lam, ForallE)):
        # Check the type of a binder and body
        # Lam/ForallE expressions are (arg : binder_type) => body.
        return (name_appears(name, expr.binder_type)
                or name_appears(name, expr.body))
    if isinstance(expr, LetE):
        # Checks declared type, bounded value, and body
        return (name_appears(name, expr.type)
                or name_appears(name, expr.value)
                or name_appears(name, expr.body))
    if isinstance(expr, (MData, Proj)):
        # checks if inductve type is inside MData or Proj sub epxressions
        return name_appears(name, expr.expr)
    return False  # Case expr is BVar, FVar, MVar, Sort, or Lit


def strict_positive_check(type_name: str, expr: Expr,
                          is_inductive_former: Callable[[str], bool]) -> bool:
    """
    A constructor is strictly positive for the inductive type T if T
    appears in positive (covariant) positions in each field type.

    Strict positivity checks that an inductive type is constructed correctly
    before it is elaborated or the kernel sees it. First, inductive type ``T``
    must not appear in expr. Second, an inductive type must not be inside the
    domain of an arrow type or function application. This imples anywhere
    inside the domain, no matter how deeply nested arrows are within it.
    In other words, the domain of a constructor's field cannot contain a self
    referenced inductive type, because this can lead to logical
    inconsistencies. Finally, since a constructor is a chain of fields (one
    ``ForallE`` arrow per field), this domain check is applied separately
    to each field along that chain.

    Negative occurrence example since ``Bad`` appears in the domain of the
    field:
        Bad : (Bad → Nat) → Bad #

    Parameters
    ----------
    type_name : str
        Name of the inductive type being checked.
    expr : Expr
        Expression node to check for a negative occurrence of
        ``type_name``.
    is_inductive_former : Callable[[str], bool]
        True an inductive type name being declared or already
        registered.

    Examples
    --------
    >>> from physika.utils.cic_utils.inductive_utils import strict_positive_check  # noqa: E501
    >>> from physika.core.expr import Const, ForallE, App
    >>> vec, alpha, n = Const("Vec", ()), Const("Real", ()), Const("n", ())
    >>> field = App(App(vec, alpha), n)  # Vec.cons's tl : Vec alpha n
    >>> strict_positive_check("Vec", field, lambda name: name == "Vec")
    True
    >>> bad, nat = Const("Bad", ()), Const("Nat", ())
    >>> domain = ForallE("_", bad, nat)  # Bad -> Nat, a constructor's field
    >>> strict_positive_check("Bad", domain, lambda name: name == "Bad")
    False
    """

    # 1) Inductive type (`type_name`) does not appear in `expr`.
    if not name_appears(type_name, expr):
        return True
    # 2) Arrow type: A → B
    # type_name must not appear in A (at any depth), then check B
    if isinstance(expr, ForallE):
        if name_appears(type_name, expr.binder_type):
            return False
        return strict_positive_check(type_name, expr.body, is_inductive_former)
    # 3) Each argument of a function application (App) is checked recursively
    head, args = get_app_fn_args(expr)
    if not (isinstance(head, Const) and is_inductive_former(head.name)):
        return False
    return all(
        strict_positive_check(type_name, a, is_inductive_former) for a in args)


def check_positivity_for_inductive(decl: "InductiveDecl",
                                   is_inductive_former=None) -> Optional[str]:
    """
    Strict positivity check for an inductive type declaration (InductiveDecl).
    This step is done before elaboration, when adding inductive types that will
    be used in `Environment`.


    Parameters
    ----------
    decl : InductiveDecl
        The inductive type being checked.
    is_inductive_former : Callable[[str], bool]
        True an inductive type name being declared or already
        registered.

    Examples
    --------
    >>> from physika.utils.cic_utils.inductive_utils import check_positivity_for_inductive # noqa: E501
    >>> from physika.core.inductive import InductiveDecl, Constructor
    >>> from physika.core.expr import Const, ForallE, TYPE_0
    >>> nat = Const("Nat", ())
    >>> nat_decl = InductiveDecl(
    ...     name="Nat", level_params=(), num_params=0, type=TYPE_0,
    ...     constructors=(Constructor("Nat.zero", nat),
    ...                   Constructor("Nat.succ", ForallE("n", nat, nat))),
    ...     is_recursive=True,
    ... )
    >>> check_positivity_for_inductive(nat_decl) is None
    True
    >>> bad = Const("Bad", ())
    >>> bad_ctor_type = ForallE("x", ForallE("_", bad, nat), bad)  # (Bad -> Nat) -> Bad
    >>> bad_decl = InductiveDecl(
    ...     name="Bad", level_params=(), num_params=0, type=TYPE_0,
    ...     constructors=(Constructor("Bad.bad", bad_ctor_type),),
    ...     is_recursive=True,
    ... )
    >>> check_positivity_for_inductive(bad_decl)
    "constructor 'Bad.bad' violates strict positivity: 'Bad' appears in a negative position in a field type"
    """
    if is_inductive_former is None:
        is_inductive_former = lambda name: name == decl.name  # noqa: E731
    for ctor in decl.constructors:
        tp = ctor.type
        for _ in range(decl.num_params):
            if isinstance(tp, ForallE):
                tp = tp.body
        while isinstance(tp, ForallE):
            field_type = tp.binder_type
            if not strict_positive_check(decl.name, field_type,
                                         is_inductive_former):
                return (
                    f"constructor '{ctor.name}' violates strict positivity: "
                    f"'{decl.name}' appears in a negative position in a field type"  # noqa: E501
                )
            tp = tp.body
    return None
