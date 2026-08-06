import itertools
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

from physika.core.level import (
    Level,
    LMVar,
    LSucc,
    LMax,
    LIMax,
)
from physika.utils.cic_utils.level_utils import mk_level_max, mk_level_imax, level_has_mvar  # noqa: E501
from physika.core.expr import (
    Expr,
    App,
    FVar,
    FVarId,
    ForallE,
    Lam,
    LetE,
    MData,
    MVar,
    MVarId,
    Proj,
    Sort,
    Const,
)
from physika.utils.cic_utils.expr_utils import loose_bvar_range, has_mvar
from physika.core.local_context import LocalContext

mvar_counter: itertools.count = itertools.count()


def fresh_mvar_id(hint: str = "m") -> MVarId:
    """Return a globally unique MVarId for a metavariable.

    Parameters
    ----------
    hint : str
        Display name prefix (e.g. "n" for a dimension metavariable).

    Example
    -------
    >>> from physika.core.metavar import fresh_mvar_id
    >>> fid = fresh_mvar_id("n")
    >>> fid.id.startswith("n.")
    True
    >>> fresh_mvar_id("n").id != fid.id
    True
    """
    return MVarId(f"{hint}.{next(mvar_counter)}")


class LMVarId:
    """
    Identifier for a universe-level metavariable.

    Parameters
    ----------
    id : str
        Unique string identifier for the level metavariable.

    Example
    -------
    >>> from physika.core.metavar import LMVarId
    >>> LMVarId("u.0") == LMVarId("u.0")
    True
    """

    def __init__(self, id: str):
        self.id = id

    def __eq__(self, other: object) -> bool:
        """
        Two ``LMVarId``\'s are equal when their ``id`` match.

        Parameters
        ----------
        other: object
            Value to compare.

        Example
        -------
        >>> from physika.core.metavar import LMVarId
        >>> LMVarId("u.0") == LMVarId("u.0")
        True
        >>> LMVarId("u.0") == LMVarId("u.1")
        False
        """
        return isinstance(other, LMVarId) and self.id == other.id

    def __hash__(self) -> int:
        """
        Hash by ``id``, equal ``LMVarId``\'s must hash the same to work as dict
        keys in ``MetaVarContext.level_assignments``.

        Example
        -------
        >>> from physika.core.metavar import LMVarId
        >>> hash(LMVarId("u.0")) == hash(LMVarId("u.0"))
        True
        """
        return hash(self.id)


class MetaVarKind(Enum):
    """Controls metavariable's kind at unification step.

    ``NATURAL`` is used for ordinary dependent-type elaboration (implicit
    args, dimension variables). ``SYNTHETIC`` corresponds to a a tactic
    placeholder. ``SYNTHETIC_OPAQUE`` marks an open tactic goal and unifier
    must never assign one (only closed with ``MetaVarContext.assign_opaque``).

    Example
    -------
    >>> from physika.core.metavar import MetaVarKind
    >>> MetaVarKind.NATURAL is MetaVarKind.NATURAL
    True
    >>> MetaVarKind.NATURAL == MetaVarKind.SYNTHETIC_OPAQUE
    False
    """
    NATURAL = auto()  # implicit args
    SYNTHETIC = auto()  # tactic inserted
    SYNTHETIC_OPAQUE = auto()  # open tactic goal


class MetaVarDecl:
    """
    Declaration for one metavariable.

    Metavariables are declared during elaboration as placeholders for terms
    not yet known.

    Parameters
    ----------
    mvar_id : MVarId
        Unique id for the declared metavariable.
    user_name : str
        Display name for the metavariable. For example, "n" for a dimension
        metavariable.
    lctx : LocalContext
        The LocalContext at creation time.
    type : Expr
        ``Expr`` type this metavariable must inhabit.
    depth : int
        Scope depth.
    kind : MetaVarKind
        Metavariable assignment kinds.

    Example
    -------
    >>> from physika.core.metavar import MetaVarDecl, MetaVarKind
    >>> from physika.core.expr import MVarId, Const
    >>> from physika.core.local_context import LocalContext
    >>> decl = MetaVarDecl(
    ...     mvar_id=MVarId("n.0"), user_name="n", lctx=LocalContext(),
    ...     type=Const("Nat", ()), depth=0, kind=MetaVarKind.NATURAL,
    ... )
    >>> decl.user_name
    'n'
    >>> decl.kind is MetaVarKind.NATURAL
    True
    """

    def __init__(self, mvar_id: MVarId, user_name: str, lctx: LocalContext,
                 type: Expr, depth: int, kind: MetaVarKind):

        self.mvar_id = mvar_id
        self.user_name = user_name
        self.lctx = lctx
        self.type = type
        self.depth = depth
        self.kind = kind


class MetaVarContextState:
    """
    Saved state of a MetaVarContext for unification.

    Used when the elaborator tries to unify two terms that might
    fail and need to restore ``MetaVarContext``.

    Parameters
    ----------
    expr_assignments : Dict[str, Expr]
        MVarId to Expr assignment map.
    level_assignments : Dict[LMVarId, Level]
        LMVarId to Level assignment map.

    Example
    -------
    >>> from physika.core.metavar import MetaVarContext
    >>> from physika.core.expr import Const
    >>> from physika.core.local_context import LocalContext
    >>> mctx = MetaVarContext()
    >>> mv = mctx.mk_mvar("n", LocalContext(), Const("Nat", ()))
    >>> mctx.expr_assignments[mv.id.id] = Const("Nat.zero", ())
    >>> snap = mctx.save()
    >>> mctx.expr_assignments[mv.id.id] = Const("Nat.succ", ())
    >>> mctx.restore(snap)
    >>> mctx.expr_assignments.get(mv.id.id)
    Const(name='Nat.zero', levels=())
    """

    def __init__(self, expr_assignments: Dict[str, Expr],
                 level_assignments: Dict[LMVarId, Level]):
        self.expr_assignments = expr_assignments
        self.level_assignments = level_assignments


class MetaVarContext:
    """
    Mutable context class that tracks all metavariables and their solutions.

    During elaboration step, one ``MetaVarContext`` is defined for the
    entire elaboration.  All inference and unification code mutates it.

    Expression metavars are stored in ``expr_assignments`` (``MVarId  → Expr``)
    and universe level metavars in ``level_assignments`` (``LMVarId → Level``)

    Example
    -------
    >>> from physika.core.metavar import MetaVarContext, MetaVarKind
    >>> from physika.core.expr import Const
    >>> from physika.core.local_context import LocalContext
    >>> mctx = MetaVarContext()
    >>> mv = mctx.mk_mvar("n", LocalContext(), Const("Nat", ()))
    >>> mctx.find_decl(mv.id).kind is MetaVarKind.NATURAL
    True
    """

    def __init__(self) -> None:
        self.decls: Dict[str, MetaVarDecl] = {}
        self.expr_assignments: Dict[str, Expr] = {}
        self.level_assignments: Dict[LMVarId, Level] = {}
        self.depth: int = 0

    def mk_mvar(self,
                name: str,
                lctx: LocalContext,
                type: Expr,
                kind: MetaVarKind = MetaVarKind.NATURAL,
                depth: Optional[int] = None) -> MVar:
        """
        Create a metavariable (``MVar``).

        ``depth`` defaults to the context's current depth.

        Parameters
        ----------
        name: str
            Display name for the metavariable.
        lctx: LocalContext
            LocalContext at creation time.
        type: Expr
            Type of the metavariable.
        kind: MetaVarKind
            Kind of the metavariable.
        depth: int, default None
            Depth of the metavariable.

        Example
        -------
        >>> from physika.core.metavar import MetaVarContext
        >>> from physika.core.expr import Const
        >>> from physika.core.local_context import LocalContext
        >>> mctx = MetaVarContext()
        >>> mv = mctx.mk_mvar("n", LocalContext(), Const("Nat", ()))
        >>> mv.id.id.startswith("n.")
        True
        """
        mvar_id = fresh_mvar_id(name)
        d = self.depth if depth is None else depth
        decl = MetaVarDecl(mvar_id, name, lctx, type, d, kind)
        self.decls[mvar_id.id] = decl
        return MVar(mvar_id)

    def find_decl(self, mvar_id: MVarId) -> Optional[MetaVarDecl]:
        """
        Looks for ``MetaVarDecl`` for a given ``MVarId``.

        Parameters
        ----------
        mvar_id: MVarId
            Metavariable identifier to look for.

        Example
        -------
        >>> from physika.core.metavar import MetaVarContext
        >>> from physika.core.expr import Const, MVarId
        >>> from physika.core.local_context import LocalContext
        >>> mctx = MetaVarContext()
        >>> mv = mctx.mk_mvar("n", LocalContext(), Const("Nat", ()))
        >>> mctx.find_decl(mv.id).user_name
        'n'
        >>> mctx.find_decl(MVarId("missing.0")) is None
        True
        """
        return self.decls.get(mvar_id.id)

    def is_valid_level_assignment(self, lmvar_id: LMVarId,
                                  val: Level) -> Tuple[bool, str]:
        """
        Check if ``lmvar_id`` with ``val``  is properly level assigned.  # noqa: E501


        Parameters
        ----------
        lmvar_id: LMVarId
            Level metavariable identifier to check.
        val: Level
            Level to check for assignment.

        Example
        -------
        >>> from physika.core.metavar import MetaVarContext, LMVarId
        >>> from physika.core.level import LZero, LMVar
        >>> mctx = MetaVarContext()
        >>> mctx.is_valid_level_assignment(LMVarId("u.0"), LZero())
        (True, '')
        >>> ok, reason = mctx.is_valid_level_assignment(LMVarId("u.0"), LMVar("u.0"))
        >>> ok
        False
        >>> "appears in its own solution" in reason
        True
        """
        # resolve val by substituing in any solved level mvar
        inst = self.instantiate_level_mvars(val)
        # checks if lmvar appears inside solved level
        if level_has_mvar(inst, lmvar_id.id):
            return False, (f"level mvar ?{lmvar_id.id} appears in its "
                           "own solution")
        return True, ""

    def instantiate_mvars(self, e: Expr) -> Expr:
        """
        Replace solved MVar nodes throughout e ``Expr`` node.

        Parameters
        ----------
        e: Expr
            Expression to instantiate.

        Example
        -------
        >>> from physika.core.metavar import MetaVarContext
        >>> from physika.core.expr import Const
        >>> from physika.core.local_context import LocalContext
        >>> mctx = MetaVarContext()
        >>> mv = mctx.mk_mvar("n", LocalContext(), Const("Nat", ()))
        >>> mctx.expr_assignments[mv.id.id] = Const("Nat.zero", ())
        >>> mctx.instantiate_mvars(mv)
        Const(name='Nat.zero', levels=())
        """
        if not has_mvar(e) and not expr_has_level_mvar(e):
            return e
        return self.inst_expr(e)

    def inst_expr(self, e: Expr) -> Expr:
        """
        Returns a new ``Expr`` with all solved MVar nodes replaced by their
        assigned values.

        Parameters
        ----------
        e: Expr
            Expression to instantiate.

        Example
        -------
        >>> from physika.core.metavar import MetaVarContext
        >>> from physika.core.expr import Const
        >>> from physika.core.local_context import LocalContext
        >>> mctx = MetaVarContext()
        >>> mv = mctx.mk_mvar("n", LocalContext(), Const("Nat", ()))
        >>> mctx.expr_assignments[mv.id.id] = Const("Nat.zero", ())
        >>> mctx.inst_expr(mv)
        Const(name='Nat.zero', levels=())
        """
        if isinstance(e, MVar):
            a = self.expr_assignments.get(e.id.id)
            if a is not None:
                return self.inst_expr(a)  # follow chain
            return e
        elif isinstance(e, App):
            nf = self.inst_expr(e.func)
            na = self.inst_expr(e.arg)
            return App(nf, na) if (nf is not e.func or na is not e.arg) else e
        elif isinstance(e, Lam):
            nt = self.inst_expr(e.binder_type)
            nb = self.inst_expr(e.body)
            return Lam(e.binder_name, nt, nb, e.binder_info) if (
                nt is not e.binder_type or nb is not e.body) else e
        elif isinstance(e, ForallE):
            nt = self.inst_expr(e.binder_type)
            nb = self.inst_expr(e.body)
            return ForallE(e.binder_name, nt, nb, e.binder_info) if (
                nt is not e.binder_type or nb is not e.body) else e
        elif isinstance(e, LetE):
            nt = self.inst_expr(e.type)
            nv = self.inst_expr(e.value)
            nb = self.inst_expr(e.body)
            return LetE(e.binder_name, nt, nv, nb,
                        e.non_dep) if (nt is not e.type or nv is not e.value
                                       or nb is not e.body) else e
        elif isinstance(e, MData):
            ne = self.inst_expr(e.expr)
            return MData(e.kvs, ne) if ne is not e.expr else e
        elif isinstance(e, Proj):
            ne = self.inst_expr(e.expr)
            return Proj(e.type_name, e.idx, ne) if ne is not e.expr else e
        elif isinstance(e, Sort):
            nl = self.instantiate_level_mvars(e.level)
            return Sort(nl) if nl is not e.level else e
        elif isinstance(e, Const):
            new_levels = tuple(
                self.instantiate_level_mvars(lvl) for lvl in e.levels)
            return Const(e.name, new_levels) if new_levels != e.levels else e
        else:  # BVar, FVar, Lit
            return e

    def instantiate_level_mvars(self, lvl: Level) -> Level:
        """
        Replace solved LMVar nodes throughout lvl (``Level``).

        Parameters
        ----------
        lvl: Level
            Level to instantiate.

        Example
        -------
        >>> from physika.core.metavar import MetaVarContext, LMVarId
        >>> from physika.core.level import LZero, LMVar
        >>> mctx = MetaVarContext()
        >>> mctx.level_assignments[LMVarId("u.0")] = LZero()
        >>> mctx.instantiate_level_mvars(LMVar("u.0"))
        LZero()
        """
        if isinstance(lvl, LMVar):
            a = self.level_assignments.get(LMVarId(lvl.id))
            if a is not None:
                return self.instantiate_level_mvars(a)  # follow chain
            return lvl
        elif isinstance(lvl, LSucc):
            np = self.instantiate_level_mvars(lvl.pred)
            return LSucc(np) if np is not lvl.pred else lvl
        elif isinstance(lvl, LMax):
            nl1 = self.instantiate_level_mvars(lvl.l1)
            nl2 = self.instantiate_level_mvars(lvl.l2)
            if nl1 is not lvl.l1 or nl2 is not lvl.l2:
                return mk_level_max(nl1, nl2)
            return lvl
        elif isinstance(lvl, LIMax):
            nl1 = self.instantiate_level_mvars(lvl.l1)
            nl2 = self.instantiate_level_mvars(lvl.l2)
            if nl1 is not lvl.l1 or nl2 is not lvl.l2:
                return mk_level_imax(nl1, nl2)
            return lvl
        else:  # LZero, LParam
            return lvl

    def is_valid_assignment(self, mvar_id: MVarId,
                            val: Expr) -> Tuple[bool, str]:
        """
        Checks if assignment of ``val: Expr`` to ``mvar_id``
        is correct.

        Parameters
        ----------
        mvar_id: MVarId
            Metavariable identifier to check.
        val: Expr
            Expression to check for assignment.

        Example
        -------
        >>> from physika.core.metavar import MetaVarContext
        >>> from physika.core.expr import Const
        >>> from physika.core.local_context import LocalContext
        >>> mctx = MetaVarContext()
        >>> mv = mctx.mk_mvar("n", LocalContext(), Const("Nat", ()))
        >>> mctx.is_valid_assignment(mv.id, Const("Nat.zero", ()))
        (True, '')
        >>> ok, reason = mctx.is_valid_assignment(mv.id, mv)
        >>> ok
        False
        >>> "occurs check" in reason
        True
        """
        decl = self.find_decl(mvar_id)
        if decl is None:
            return False, f"?{mvar_id.id} not found in MetaVarContext"
        if decl.kind is MetaVarKind.SYNTHETIC_OPAQUE:
            return False, (f"?{mvar_id.id} is a synthetic-opaque goal")
        if loose_bvar_range(val) > 0:
            return False, "solution contains loose BVars"

        if has_mvar(self.instantiate_mvars(val), mvar_id):
            return False, f"occurs check: ?{mvar_id.id} appears in its own solution"  # noqa: E501

        result: List[FVarId] = []
        seen: Set[str] = set()
        collect_fvars_rec(val, result, seen)
        for fvar_id in result:
            if not decl.lctx.contains(fvar_id):
                return False, (
                    f"solution mentions FVars outside ?{mvar_id.id}'s "
                    "creation scope")
        # val must not reference an MVar created at a deeper scope
        result_mvars: List[MVarId] = []
        seen_mvars: Set[str] = set()
        collect_all_mvars(val, result_mvars, seen_mvars)
        for other_id in result_mvars:
            other_decl = self.find_decl(other_id)
            if other_decl is not None and other_decl.depth > decl.depth:
                return False, (
                    f"solution references a metavariable from a deeper/more "
                    f"speculative scope than ?{mvar_id.id}'s")
        return True, ""

    def save(self) -> MetaVarContextState:
        """
        Returns current state of the MetaVarContext assingments.

        Example
        -------
        >>> from physika.core.metavar import MetaVarContext
        >>> mctx = MetaVarContext()
        >>> snap = mctx.save()
        """
        return MetaVarContextState(
            expr_assignments=dict(self.expr_assignments),
            level_assignments=dict(self.level_assignments),
        )

    def restore(self, snap: MetaVarContextState) -> None:
        """
        Restore MetaVarContext assignments to a ``snap``/'s state.

        Parameters
        ----------
        snap: MetaVarContextState
            State to restore the MetaVarContext to.

        Example
        -------
        >>> from physika.core.metavar import MetaVarContext
        >>> from physika.core.expr import Const
        >>> from physika.core.local_context import LocalContext
        >>> mctx = MetaVarContext()
        >>> mv = mctx.mk_mvar("n", LocalContext(), Const("Nat", ()))
        >>> mctx.expr_assignments[mv.id.id] = Const("Nat.zero", ())
        >>> snap = mctx.save()
        >>> mctx.expr_assignments[mv.id.id] = Const("Nat.succ", ())
        >>> mctx.restore(snap)
        >>> mctx.expr_assignments[mv.id.id]
        Const(name='Nat.zero', levels=())
        """
        self.expr_assignments = snap.expr_assignments
        self.level_assignments = snap.level_assignments

    def unassigned_mvars(self, e: Expr) -> List[MVarId]:
        """
        Collect ``MVarId``'/s in ``e`` that have not been assigned.

        Parameters
        ----------
        e: Expr
            Expression to collect unassigned metavariables from.

        Example
        -------
        >>> from physika.core.metavar import MetaVarContext
        >>> from physika.core.expr import Const
        >>> from physika.core.local_context import LocalContext
        >>> mctx = MetaVarContext()
        >>> mv = mctx.mk_mvar("n", LocalContext(), Const("Nat", ()))
        >>> len(mctx.unassigned_mvars(mv))
        1
        >>> mctx.expr_assignments[mv.id.id] = Const("Nat.zero", ())
        >>> mctx.unassigned_mvars(mv)
        []
        """
        result: List[MVarId] = []
        seen: Set[str] = set()
        collect_unassigned(e, self, result, seen)
        return result

    def all_mvars(self, e: Expr) -> List[MVarId]:
        """
        Collect all MVarIds that appear in e.

        Parameters
        ----------
        e: Expr
            Expression to collect metavariables from.

        Example
        -------
        >>> from physika.core.metavar import MetaVarContext
        >>> from physika.core.expr import Const
        >>> from physika.core.local_context import LocalContext
        >>> mctx = MetaVarContext()
        >>> mv = mctx.mk_mvar("n", LocalContext(), Const("Nat", ()))
        >>> mctx.expr_assignments[mv.id.id] = Const("Nat.zero", ())
        >>> len(mctx.all_mvars(mv))
        1
        >>> len(mctx.unassigned_mvars(mv))
        0
        """
        result: List[MVarId] = []
        seen: Set[str] = set()
        collect_all_mvars(e, result, seen)
        return result


def children(e: Expr) -> Tuple[Expr, ...]:
    """
    Helper funtion to collect ``e:Expr`` children nodes.

    Parameters
    ----------
    e: Expr
        Expression to collect children from.

    Example
    -------
    >>> from physika.core.metavar import children
    >>> from physika.core.expr import Const, App
    >>> f = Const("f", ())
    >>> a = Const("a", ())
    >>> children(App(f, a))
    (Const(name='f', levels=()), Const(name='a', levels=()))
    >>> children(f)
    ()
    """
    if isinstance(e, App):
        return (e.func, e.arg)
    elif isinstance(e, (Lam, ForallE)):
        return (e.binder_type, e.body)
    elif isinstance(e, LetE):
        return (e.type, e.value, e.body)
    elif isinstance(e, (MData, Proj)):
        return (e.expr, )
    return ()


def expr_has_level_mvar(e: Expr) -> bool:
    """
    Recursively checks if ``e:Expr`` contains LMVar inside a Sort or Const
    level list.

    Parameters
    ----------
    e: Expr
        Expression to check LMVar from

    Example
    -------
    >>> from physika.core.metavar import expr_has_level_mvar
    >>> from physika.core.expr import Sort
    >>> from physika.core.level import LZero, LMVar
    >>> expr_has_level_mvar(Sort(LZero()))
    False
    >>> expr_has_level_mvar(Sort(LMVar("u.0")))
    True
    """
    if isinstance(e, Sort):
        return level_has_mvar(e.level)
    if isinstance(e, Const):
        return any(level_has_mvar(lvl) for lvl in e.levels)
    return any(expr_has_level_mvar(c) for c in children(e))


def collect_fvars_rec(e: Expr, result: List[FVarId], seen: Set[str]) -> None:
    """
    Collect ``FVarId``\'s in ``e:Expr``. To avoid duplicates, we use ``seen``
    set and mutate ``result`` list.

    Parameters
    ----------
    e: Expr
        Expression to collect FVarId from.
    result: List[FVarId]
        List to append FVarId to.
    seen: Set[str]
        Set of FVarId strings to avoid duplicates.

    Example
    -------
    >>> from physika.core.metavar import collect_fvars_rec
    >>> from physika.core.expr import FVar, FVarId, Const, App
    >>> fv = FVar(FVarId("x.0"))
    >>> result = []
    >>> seen = set()
    >>> collect_fvars_rec(App(Const("f", ()), fv), result, seen)
    >>> result
    [FVarId(id='x.0')]
    """
    if isinstance(e, FVar):
        if e.id.id not in seen:
            seen.add(e.id.id)
            result.append(e.id)
        return
    for c in children(e):
        collect_fvars_rec(c, result, seen)


def collect_unassigned(e: Expr, mctx: MetaVarContext, result: List[MVarId],
                       seen: Set[str]) -> None:
    """
    Recursively collect ``MVarId``\'s in ``e:Expr`` that are unassigned. As
    ``collect_fvars_rec`` , we use ``seen`` set to avoid duplicates.

    Parameters
    ----------
    e: Expr
        Expression to collect unassigned MVarId from.
    mctx: MetaVarContext
        Context to check each MVarId's assignment against.
    result: List[MVarId]
        List to append unassigned MVarId to.
    seen: Set[str]
        Set of MVarId strings to avoid duplicates.

    Example
    -------
    >>> from physika.core.metavar import MetaVarContext, collect_unassigned
    >>> from physika.core.expr import Const
    >>> from physika.core.local_context import LocalContext
    >>> mctx = MetaVarContext()
    >>> mv = mctx.mk_mvar("n", LocalContext(), Const("Nat", ()))
    >>> result = []
    >>> seen = set()
    >>> collect_unassigned(mv, mctx, result, seen)
    >>> len(result)
    1
    >>> mctx.expr_assignments[mv.id.id] = Const("Nat.zero", ())
    >>> result2 = []
    >>> seen2 = set()
    >>> collect_unassigned(mv, mctx, result2, seen2)
    >>> result2
    []
    """
    if isinstance(e, MVar):
        if e.id.id in seen:
            return
        seen.add(e.id.id)
        a = mctx.expr_assignments.get(e.id.id)
        if a is None:
            result.append(e.id)
        else:
            collect_unassigned(a, mctx, result, seen)
        return
    for c in children(e):
        collect_unassigned(c, mctx, result, seen)


def collect_all_mvars(e: Expr, result: List[MVarId], seen: Set[str]) -> None:
    """
    Collect all MVarIds in e (assigned or not).
    Parameters
    ----------
    e: Expr
        Expression to collect MVarId from.
    result: List[MVarId]
        List to append MVarId to.
    seen: Set[str]
        Set of MVarId strings to avoid duplicates.

    Example
    -------
    >>> from physika.core.metavar import MetaVarContext, collect_all_mvars
    >>> from physika.core.expr import Const
    >>> from physika.core.local_context import LocalContext
    >>> mctx = MetaVarContext()
    >>> mv = mctx.mk_mvar("n", LocalContext(), Const("Nat", ()))
    >>> result = []
    >>> seen = set()
    >>> collect_all_mvars(mv, result, seen)
    >>> len(result)
    1
    """
    if isinstance(e, MVar):
        if e.id.id not in seen:
            seen.add(e.id.id)
            result.append(e.id)
        return
    for c in children(e):
        collect_all_mvars(c, result, seen)
