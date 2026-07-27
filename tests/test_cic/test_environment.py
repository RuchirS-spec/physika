from physika.core.environment import ConstantInfo, Environment, InductiveInfo
from physika.core.expr import BVar, Const, ForallE, TYPE_0
from physika.core.inductive import Constructor, InductiveDecl, Recursor, RecursorRule  # noqa: E501


class TestConstantInfo:
    """
    Tests for ``ConstantInfo``
    """

    def test_constant_info(self):
        """
        Checks proper construction of ``ConstantInfo``.
        """
        nat = Const("Nat", ())
        succ_type = ForallE("n", nat, nat)
        succ_ci = ConstantInfo("Nat.succ", (), succ_type, None)

        assert succ_ci.name == "Nat.succ"
        assert succ_ci.level_params == ()
        assert succ_ci.type == succ_type
        assert succ_ci.value is None

        # checks proper construction of Nat.add
        nat = Const("Nat", ())
        body = ForallE("n", nat,
                       nat)  # not the actual add Expr, just adding for test
        add_ci = ConstantInfo(name="Nat.add",
                              level_params=(),
                              type=ForallE("n", nat, nat),
                              value=body)

        assert add_ci.name == "Nat.add"
        assert add_ci.value == body
        assert add_ci.value is not None

        # checks universe polymorphism parameters are stored
        rec_ci = ConstantInfo("Nat.rec", ("u", ), Const("Nat", ()), None)

        assert rec_ci.level_params == ("u", )


class TestInductiveInfo:
    """
    Tests for ``InductiveInfo``
    """

    def nat_decl(self):
        """
        Nat inductive type declaration used for tests
        """
        nat = Const("Nat", ())
        return InductiveDecl(
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

    def test_inductive_info(self):
        """
        Checks proper construction of ``InductiveInfo`` for ``Nat``
        """
        nat = Const("Nat", ())
        nat_decl = self.nat_decl()
        zero_ci = ConstantInfo("Nat.zero", (), nat, None)
        succ_ci = ConstantInfo("Nat.succ", (), ForallE("n", nat, nat), None)
        rec_ci = ConstantInfo("Nat.rec", ("u", ), nat, None)

        nat_info = InductiveInfo(
            decl=nat_decl,
            ctors={
                "Nat.zero": zero_ci,
                "Nat.succ": succ_ci
            },
            recursor=rec_ci,
        )

        assert nat_info.decl == nat_decl
        assert nat_info.ctors == {"Nat.zero": zero_ci, "Nat.succ": succ_ci}
        assert nat_info.recursor == rec_ci
        # should return None because is an axiom
        assert nat_info.rec_info is None

    def test_inductive_info_with_rec_info(self):
        """
        Case when a ``Recursor`` (with iota-reduction rules)
        is registered.
        """
        nat = Const("Nat", ())
        nat_decl = self.nat_decl()
        zero_rule = RecursorRule("Nat.zero", nfields=0, rhs=BVar(0))
        recursor = Recursor(
            name="Nat.rec",
            type=nat,
            num_params=0,
            num_indices=0,
            num_motives=1,
            num_minors=2,
            rules=(zero_rule, ),
            level_params=("u", ),
        )

        nat_info = InductiveInfo(
            decl=nat_decl,
            ctors={"Nat.zero": ConstantInfo("Nat.zero", (), nat, None)},
            recursor=ConstantInfo("Nat.rec", ("u", ), nat, None),
            rec_info=recursor,
        )

        assert nat_info.rec_info is recursor
        assert nat_info.rec_info.name == "Nat.rec"


class TestEnvironment:
    """
    Tests for ``Environment``
    """

    def test_empty_environment(self):
        """
        Checks a ``Environment`` has no constants
        and no inductive types registered when first constructed.
        """
        env = Environment()

        assert env.constants == {}
        assert env.inductives == {}

    def test_add_constant(self):
        """
        Checks add_constant method works when accesing by its name.
        """
        env = Environment()
        zero_ci = ConstantInfo("Nat.zero", (), Const("Nat", ()), None)
        env.add_constant(zero_ci)

        assert "Nat.zero" in env.constants
        assert env.constants["Nat.zero"] is zero_ci

    def test_add_inductive(self):
        """
        Checks add_inductive registers ``InductiveInfo``,
        constructors and recursor as constants too.
        """
        env = Environment()
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
        nat_info = InductiveInfo(
            decl=nat_decl,
            ctors={
                "Nat.zero":
                ConstantInfo("Nat.zero", (), nat, None),
                "Nat.succ":
                ConstantInfo("Nat.succ", (), ForallE("n", nat, nat), None),
            },
            recursor=ConstantInfo("Nat.rec", ("u", ), nat, None),
        )

        env.add_inductive(nat_info)

        assert env.inductives["Nat"] is nat_info
        # constructors and recursor are constants
        assert "Nat.zero" in env.constants
        assert "Nat.succ" in env.constants
        assert "Nat.rec" in env.constants
