"""format_instr spells conditions the way vc4asm reads them."""
from piton.qpu import isa


def _alu(cond):
    return isa.Instr(kind="alu", add=isa.AluOp("mov", isa.ACC[0], isa.ACC[1], isa.ACC[1], cond=cond))


def test_carry_clear_is_ifcc():
    # vc4asm reads `.ifnc` as "N clear" (condition 5); carry clear (7) is `.ifcc`.
    assert ".ifcc" in isa.format_instr(_alu("ifnc"))
    assert ".ifnc" not in isa.format_instr(_alu("ifnc"))


def test_carry_clear_ldi_is_ifcc():
    ins = isa.Instr(kind="ldi", add=isa.AluOp("mov", isa.ACC[0], isa.NOP, isa.NOP, cond="ifnc"), imm=1)
    assert isa.format_instr(ins).startswith("ldi.ifcc ")


def test_other_conditions_unchanged():
    for cond in ("ifz", "ifnz", "ifn", "ifnn", "ifc"):
        assert f".{cond} " in isa.format_instr(_alu(cond))
