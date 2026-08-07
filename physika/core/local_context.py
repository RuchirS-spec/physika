import itertools
from typing import Dict, Iterator, List, Optional, Tuple, Union

from physika.core.expr import (
    Expr,
    FVar,
    FVarId,
    ForallE,
    Lam,
    LetE,
    BinderInfo,
)
from physika.utils.cic_utils.expr_utils import abstract_fvars

fvar_counter: Iterator[int] = itertools.count()


def fresh_fvar_id(hint: str = "x") -> FVarId:
    """
    Helper function that returns a globally unique FVarId.

    Parameters
    ----------
    hint: str
      Display name prefix.

    Example
    -------
    The numeric suffix comes from a module-level counter shared by every
    caller (including other doctests in this file), so only the prefix
    and uniqueness are checked here, not one specific literal value.

    >>> from physika.core.local_context import fresh_fvar_id
    >>> fid = fresh_fvar_id("x")
    >>> fid.id.startswith("x.")
    True
    >>> fresh_fvar_id("x").id != fid.id
    True
    """
    return FVarId(f"{hint}.{next(fvar_counter)}")


class LocalDeclVar:
    """
    Represents a binder where a variable is introduced with a specific type
    but no assigned value.

    ``LocalDeclVar`` is used when opening a binder from a local context.

    Parameters
    ----------
    fvar_id: FVarId
        New free variable ID created for a given binder parameter name.
    index: int
        Total number of declarations within a local context (``LocalContext``)
    user_name: str
        Name of the current declaration which was given by user.
    type: Expr
        Type of declaration variable.
    binder_info: BinderInfo
        Indicates if declaration variable must be given explicitly inferred by
        the elaborator via unification, which creates a ``MVar``.

    Example
    -------
    >>> from physika.core.local_context import LocalDeclVar
    >>> from physika.core.expr import FVarId, Const, BinderInfo
    >>> decl = LocalDeclVar(
    ...     fvar_id=FVarId("x.0"),
    ...     index=0,
    ...     user_name="x",
    ...     type=Const("Real", ()),
    ...     binder_info=BinderInfo.DEFAULT,
    ... )
    >>> decl.user_name
    'x'
    >>> decl.type
    Const(name='Real', levels=())
    """

    def __init__(self, fvar_id: FVarId, index: int, user_name: str, type: Expr,
                 binder_info: BinderInfo):

        self.fvar_id = fvar_id
        self.index = index
        self.user_name = user_name
        self.type = type
        self.binder_info = binder_info


class LocalDeclDef:
    """
    Represents a local definition (let binding), which includes both a type
    and a value (``x: T = t``)

    ``LocalDeclDef`` is used when opening definition from a local context.

    Parameters
    ----------
    fvar_id: FVarId
        New free variable ID created for a given local definition name.
    index: int
        Total number of declarations within a local context (``LocalContext``)
    user_name: str
        Name of the current local definition which was given by user.
    type: Expr
        Type of declaration variable.
    value: Expr
        Value bound to the local definition.
    non_dep: bool, default False
        True if enclosing term's body never actually references this let bound
        value.

    Example
    -------
    >>> from physika.core.local_context import LocalDeclDef
    >>> from physika.core.expr import FVarId, Const
    >>> decl = LocalDeclDef(
    ...     fvar_id=FVarId("x.0"),
    ...     index=0,
    ...     user_name="x",
    ...     type=Const("Real", ()),
    ...     value=Const("one", ()),
    ... )
    >>> decl.user_name
    'x'
    >>> decl.value
    Const(name='one', levels=())
    >>> decl.non_dep
    False
    """

    def __init__(self,
                 fvar_id: FVarId,
                 index: int,
                 user_name: str,
                 type: Expr,
                 value: Expr,
                 non_dep: bool = False):
        self.fvar_id = fvar_id
        self.index = index
        self.user_name = user_name
        self.type = type
        self.value = value
        self.non_dep = non_dep


LocalDecl = Union[LocalDeclVar, LocalDeclDef]


class LocalContext:
    """
    Local variable declarations and definitions.


    ``LocalContext`` is used during elaboration to store variable declarations
    in order they were accesed and as a dict for ``FVarId`` lookup.

    Example
    -------
    >>> from physika.core.local_context import LocalContext
    >>> from physika.core.expr import Const
    >>> lctx = LocalContext()
    >>> lctx2, x_fv = lctx.push_local("x", Const("Real", ()))
    >>> len(lctx.decls), len(lctx2.decls)
    (0, 1)
    >>> lctx2.fvar_type(x_fv)
    Const(name='Real', levels=())
    >>> lctx2.find(x_fv.id).user_name
    'x'
    """

    def __init__(self) -> None:
        """
        Initialize method for ``LocalContext`` that contains two main fields:
        ``decls`` and ``fvar_map``. ``decls`` is a list of ``LocalDecl``
        objects (which can be function arguments or variable definitions)
        registered in irder of appearance within a local context. ``fvar_map``
        is a dictionary which serves for lookup FVar by ID (name).
        """
        self.decls: List[LocalDecl] = []
        self.fvar_map: Dict[str, LocalDecl] = {}

    def make(self, decls: List[LocalDecl],
             fvar_map: Dict[str, LocalDecl]) -> "LocalContext":
        """
        Create a new local context object (``LocalContext``) from existing data
        to avoid modifying current local context.

        Parameters
        ----------
        decls: List[LocalDecl]
            List of function arguments or variable definitions of current
            ``LocalContext``
        fvar_map: Dict[str, LocalDecl]
            Dictionary for lookup ``FVar`` by ID of current ``LocalContext``.

        Example
        -------
        >>> from physika.core.local_context import LocalContext, LocalDeclVar
        >>> from physika.core.expr import FVarId, Const, BinderInfo
        >>> decl = LocalDeclVar(
        ...     FVarId("x.0"), 0, "x", Const("Real", ()), BinderInfo.DEFAULT,
        ... )
        >>> lctx = LocalContext().make([decl], {"x.0": decl})
        >>> lctx.decls_l() == [decl]
        True
        """
        lctx = LocalContext.__new__(LocalContext)
        lctx.decls = decls
        lctx.fvar_map = fvar_map
        return lctx

    def push_local(
        self,
        name: str,
        type: Expr,
        bi: BinderInfo = BinderInfo.DEFAULT,
    ) -> Tuple["LocalContext", FVar]:
        """
        Used during elaboration when a binder is "opened" for variable
        declaration (e.g. unfolding a Pi-type and when inferring Lam/ForallE
        types).

        Register a new ``FVar`` for each binder's ``BVar``. Then, the body can
        be elaborated against a "name" instead of working with de Bruijn index
        (reducing chances of dealing with incorrect indexes).

        Parameters
        ----------
        name: str
            Name of the current local declaration.
        type: Expr
            Type of declaration variable.
        bi: BinderInfo
            Indicates if declaration variable must be given explicitly inferred
            by the elaborator via unification.

        Example
        -------
        >>> from physika.core.local_context import LocalContext
        >>> from physika.core.expr import Const
        >>> lctx = LocalContext()
        >>> lctx2, x_fv = lctx.push_local("x", Const("Real", ()))
        >>> lctx2.fvar_type(x_fv)
        Const(name='Real', levels=())
        >>> lctx.contains(x_fv.id)
        False
        """
        idx = len(self.decls)
        fvar_id = fresh_fvar_id(name)
        decl = LocalDeclVar(fvar_id, idx, name, type, bi)
        new_decls = self.decls + [decl]
        new_map = dict(self.fvar_map)
        new_map[fvar_id.id] = decl
        return self.make(new_decls, new_map), FVar(fvar_id)

    def push_let(
        self,
        name: str,
        type: Expr,
        value: Expr,
        non_dep: bool = False,
    ) -> Tuple["LocalContext", FVar]:
        """
        Used during elaboration when a binder is "opened" for local
        definitions.

        Returns a new ``LocalContext`` object, with new definitions and FVar
        IDs.

        Parameters
        ----------
        name: str
        Name of the current local definition which was given by user.
        type: Expr
            Type of declaration variable.
        value: Expr
            Value bound to the local definition.
        non_dep: bool, default False
            True if enclosing term's body never actually references this let
            bound value.

        Example
        -------
        >>> from physika.core.local_context import LocalContext  # noqa: E501
        >>> from physika.core.expr import Const
        >>> lctx = LocalContext()
        >>> lctx2, x_fv = lctx.push_let("x", Const("Real", ()), Const("one", ()))
        >>> lctx2.find(x_fv.id).value
        Const(name='one', levels=())
        """
        idx = len(self.decls)
        fvar_id = fresh_fvar_id(name)
        decl = LocalDeclDef(fvar_id, idx, name, type, value, non_dep)
        new_decls = self.decls + [decl]
        new_map = dict(self.fvar_map)
        new_map[fvar_id.id] = decl
        return self.make(new_decls, new_map), FVar(fvar_id)

    def find(self, fvar_id: FVarId) -> Optional[LocalDecl]:
        """
        Looks ``LocalDecl`` declaration object related to fvar_id. Returns None
        if not found in ``fvar_map``.

        Parameters
        ----------
        fvar_id: FVarId
            Free variable ID

        Example
        -------
        >>> from physika.core.local_context import LocalContext
        >>> from physika.core.expr import Const, FVarId
        >>> lctx = LocalContext()
        >>> lctx2, x_fv = lctx.push_local("x", Const("Real", ()))
        >>> lctx2.find(x_fv.id).user_name
        'x'
        >>> lctx2.find(FVarId("nope")) is None
        True
        """
        return self.fvar_map.get(fvar_id.id)

    def contains(self, fvar_id: FVarId) -> bool:
        """
        Checks if a ``FVar`` ID is registered in FVar's mapping for local
        context.

        Parameters
        ----------
        fvar_id: FVarId
            Free variable ID

        Example
        -------
        >>> from physika.core.local_context import LocalContext
        >>> from physika.core.expr import Const, FVarId
        >>> lctx = LocalContext()
        >>> lctx2, x_fv = lctx.push_local("x", Const("Real", ()))
        >>> lctx2.contains(x_fv.id)
        True
        >>> lctx2.contains(FVarId("nope"))
        False
        """
        return fvar_id.id in self.fvar_map

    def fvar_type(self, fv: Union[FVar, FVarId]) -> Expr:
        """
        Return the type of a local variable.

        Parameters
        ----------
        fv : Union[FVar, FVarId]
            Free variable (FVar) to check type either by name or ID.

        Example
        -------
        >>> from physika.core.local_context import LocalContext
        >>> from physika.core.expr import Const
        >>> lctx = LocalContext()
        >>> lctx2, x_fv = lctx.push_local("x", Const("Real", ()))
        >>> lctx2.fvar_type(x_fv)
        Const(name='Real', levels=())
        >>> lctx2.fvar_type(x_fv.id)
        Const(name='Real', levels=())
        """
        fvar_id = fv.id if isinstance(fv, FVar) else fv
        decl = self.find(fvar_id)
        if decl is None:
            raise KeyError(f"FVarId '{fvar_id.id}' not found in LocalContext")
        return decl.type

    def decls_l(self) -> List[LocalDecl]:
        """
        Gets all the local declaration objects within a local context.

        Example
        -------
        >>> from physika.core.local_context import LocalContext
        >>> from physika.core.expr import Const
        >>> lctx = LocalContext()
        >>> lctx2, x_fv = lctx.push_local("x", Const("Real", ()))
        >>> [d.user_name for d in lctx2.decls_l()]
        ['x']
        """
        return list(self.decls)

    def mk_lambda(self, fvars: List[FVar], body: Expr) -> Expr:
        """
        Used during elaboration for closing a body under Lam nodes for each
        ``FVar``.

        Parameters
        ----------
        fvars: List[FVar]
            Free variables to convert back to de bruijn indices.
        body: Expr
            Body of ``Lam`` node

        Example
        -------
        >>> from physika.core.local_context import LocalContext  # noqa: E501
        >>> from physika.core.expr import Const, FVar
        >>> lctx = LocalContext()
        >>> lctx2, x_fv = lctx.push_local("x", Const("Real", ()))
        >>> lctx2.mk_lambda([x_fv], FVar(x_fv.id))
        Lam(binder_name='x', binder_type=Const(name='Real', levels=()), body=BVar(idx=0), binder_info=<BinderInfo.DEFAULT: 1>)
        """
        return self.mk_binding(fvars, body, is_lam=True)

    def mk_forall(self, fvars: List[FVar], body: Expr) -> Expr:
        """
        Used during elaboration for closing a body under Pi-types (Forall)
        nodes for each ``FVar``.

        Parameters
        ----------
        fvars: List[FVar]
            Free variables to convert back to de bruijn indices.
        body: Expr
            Body of ``Forall`` node

        Example
        -------
        >>> from physika.core.local_context import LocalContext  # noqa: E501
        >>> from physika.core.expr import Const
        >>> lctx = LocalContext()
        >>> lctx2, x_fv = lctx.push_local("x", Const("Real", ()))
        >>> lctx2.mk_forall([x_fv], Const("Real", ()))
        ForallE(binder_name='x', binder_type=Const(name='Real', levels=()), body=Const(name='Real', levels=()), binder_info=<BinderInfo.DEFAULT: 1>)
        """
        return self.mk_binding(fvars, body, is_lam=False)

    def mk_binding(self, fvars: List[FVar], body: Expr, is_lam: bool) -> Expr:
        """
        Create a binding expression node (``Expr``) for ``Lam/LetE/ForallE``
        nodes in a local context.

        Parameters
        ----------
        fvars: List[FVar]
            Free variables to convert back to de bruijn indices.
        body: Expr
            Body of either ``Lam/LetE/Forall`` nodes
        is_lam: bool
            True if there is a ``Lam`` node in ``body``.

        Example
        -------
        >>> from physika.core.local_context import LocalContext  # noqa: E501
        >>> from physika.core.expr import Const, FVar
        >>> lctx = LocalContext()
        >>> lctx2, x_fv = lctx.push_local("x", Const("Real", ()))
        >>> lctx2.mk_binding([x_fv], FVar(x_fv.id), is_lam=True)
        Lam(binder_name='x', binder_type=Const(name='Real', levels=()), body=BVar(idx=0), binder_info=<BinderInfo.DEFAULT: 1>)
        """
        if not fvars:
            return body
        rev = list(reversed(fvars))
        closed = abstract_fvars(body, rev)
        result = closed
        for i, fv in enumerate(rev):
            d = self.find(fv.id)
            if d is None:
                raise KeyError(
                    f"FVarId '{fv.id.id}' not found in LocalContext")

            outer = [FVar(rv.id) for rv in rev[i + 1:]]
            btype = abstract_fvars(d.type, outer)
            if isinstance(d, LocalDeclDef):
                bvalue = abstract_fvars(d.value, outer)
                result = LetE(d.user_name, btype, bvalue, result)
            elif is_lam:
                result = Lam(d.user_name, btype, result, d.binder_info)
            else:
                result = ForallE(d.user_name, btype, result, d.binder_info)
        return result
