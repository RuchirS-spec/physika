from physika.core.expr import (
    App,
    BVar,
    Const,
    ForallE,
    Lam,
    LetE,
    MData,
    Proj,
    TYPE_0,
)
from physika.core.inductive import Constructor, InductiveDecl
from physika.utils.cic_utils.inductive_utils import (
    check_positivity_for_inductive,
    name_appears,
    strict_positive_check,
)


class TestNameAppears:
    """
    Tests for ``name_appears``
    """

    def test_name_appears(self):
        """
        Checks ``Const`` node with ``name``.
        """
        nat = Const("Nat", ())

        assert name_appears("Nat", nat) is True

        # Not registered name should fail
        nat = Const("Nat", ())

        assert name_appears("Bool", nat) is False

    def test_name_appears_in_app_func(self):
        """
        Checks ``name`` is found inside a function's application
        ``App``.
        """
        nat = Const("Nat", ())
        call = App(nat, Const("x", ()))

        assert name_appears("Nat", call) is True

        # should also find "name" when is applied as an arg
        nat = Const("Nat", ())
        call = App(Const("f", ()), nat)

        assert name_appears("Nat", call) is True

    def test_name_appears_in_lam(self):
        """
        Checks `name` is inside a ``Lam``.
        """

        nat = Const("Nat", ())
        lam = Lam("x", nat, Const("x", ()))

        # should be in Lam's binder type
        assert name_appears("Nat", lam) is True

        # should also be able to found "name" in body
        lam = Lam("x", Const("Real", ()), nat)

        assert name_appears("Nat", lam) is True

    def test_name_appears_in_forall(self):
        """
        Checks `name` is found in a ``ForallE``.
        """
        nat = Const("Nat", ())
        pi = ForallE("x", nat, Const("Real", ()))

        # should be in ForallE binder type
        assert name_appears("Nat", pi) is True

        # should also be able to found "name" in body
        nat = Const("Nat", ())
        pi = ForallE("x", Const("Real", ()), nat)

        assert name_appears("Nat", pi) is True

    def test_name_appears_in_lete(self):
        """
        Checks `name` is found in a ``LetE``
        """
        nat = Const("Nat", ())
        let = LetE("x", nat, Const("v", ()), Const("b", ()))

        # should be in LetE declared type
        assert name_appears("Nat", let) is True

        # should also appear in bound type
        let = LetE("x", Const("Real", ()), nat, Const("b", ()))
        assert name_appears("Nat", let) is True

        # should also appear in body
        let = LetE("x", Const("Real", ()), Const("v", ()), nat)
        assert name_appears("Nat", let) is True

    def test_name_appears_in_mdata(self):
        """
        Checks `name` is found in``MData`` wrapper.
        """
        nat = Const("Nat", ())
        wrapped = MData((("line", 1), ), nat)

        assert name_appears("Nat", wrapped) is True

    def test_name_appears_in_proj(self):
        """
        Checks `name` is in an instance of ``Proj``.
        """
        nat = Const("Nat", ())
        proj = Proj("Ray", 0, nat)

        assert name_appears("Nat", proj) is True


class TestStrictPositiveCheck:
    """
    Tests for ``strict_positive_check``
    """

    def test_direct_self_application(self):
        """
        Checks ``Vec.cons``'s field ``tl : Vec alpha n``
        """
        vec = Const("Vec", ())
        alpha = Const("Real", ())
        n = Const("n", ())
        # field: Vec α n
        field = App(App(vec, alpha), n)
        # self application must be positive
        assert strict_positive_check("Vec", field,
                                     lambda name: name == "Vec") is True

    def test_negative_in_arrow_domai(self):
        """
        Checks a type name appearing in the domain of an arrow
        type is rejected.
        """
        bad = Const("Bad", ())
        nat = Const("Nat", ())

        domain = ForallE("_", bad, nat)
        # bad type at the left should fail positivity check
        assert strict_positive_check("Bad", domain,
                                     lambda name: name == "Bad") is False

        domain = ForallE("_", nat, bad)
        # bad after nat in arrow type should be positive
        assert strict_positive_check("Bad", domain,
                                     lambda name: name == "Bad") is True

    def test_nested_positive_via_another_inductive_former(self):
        """
        Checks ``List Tree`` is positive when ``List`` is an inductive type
        former.
        """
        lst = Const("List", ())
        tree = Const("Tree", ())
        field = App(lst, tree)
        is_former = lambda name: name in ("List", "Tree")  # noqa: E731

        assert strict_positive_check("Tree", field, is_former) is True

    def test_rejected_when_head_is_not_an_inductive_former(self):
        """
        Checks ``f (Fix f)``, where ``f`` is a bound variable fails positivity
        test.
        """
        fix = Const("Fix", ())
        f_param = BVar(0)
        fix_f = App(fix, f_param)
        weird = App(f_param, fix_f)

        assert strict_positive_check("Fix", weird,
                                     lambda name: name == "Fix") is False


class TestCheckPositivityForInductive:
    """
    Tests for ``check_positivity_for_inductive``
    """

    def test_nat_passes(self):
        """
        Checks positivity for ``Nat`` inductive type
        """
        nat = Const("Nat", ())
        nat_decl = InductiveDecl(
            name="Nat",
            level_params=(),
            num_params=0,
            type=TYPE_0,
            constructors=(
                Constructor("Nat.zero", nat),
                Constructor("Nat.succ", ForallE("n", nat, nat)),
            ),
            is_recursive=True,
        )

        assert check_positivity_for_inductive(nat_decl) is None

    def test_bad_inductive_type(self):
        """
        Checks ``Bad.bad : (Bad -> Nat) -> Bad`` fails and an error message
        is reported.
        """
        bad = Const("Bad", ())
        nat = Const("Nat", ())
        bad_ctor_type = ForallE("x", ForallE("_", bad, nat), bad)
        bad_decl = InductiveDecl(
            name="Bad",
            level_params=(),
            num_params=0,
            type=TYPE_0,
            constructors=(Constructor("Bad.bad", bad_ctor_type), ),
            is_recursive=True,
        )

        result = check_positivity_for_inductive(bad_decl)
        assert result == "constructor 'Bad.bad' violates strict positivity: 'Bad' appears in a negative position in a field type"  # noqa: E501

    def test_custom_predicate_allows_nested_inductive(self):
        """
        Checks positivity ``Tree.node : List Tree -> Tree`` passes if
        ``List`` is an inductive type former.
        """
        lst = Const("List", ())
        tree = Const("Tree", ())
        tree_ctor_type = ForallE("x", App(lst, tree), tree)
        tree_decl = InductiveDecl(
            name="Tree",
            level_params=(),
            num_params=0,
            type=TYPE_0,
            constructors=(Constructor("Tree.node", tree_ctor_type), ),
            is_recursive=True,
        )

        def is_former(name):
            return name in ("List", "Tree")

        assert check_positivity_for_inductive(tree_decl, is_former) is None
