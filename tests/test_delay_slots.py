"""Branch slot filling must preserve data, flag and pipeline dependencies."""
import pytest

from piton.backend.sched import fill_delay_slots
from piton.qpu import isa
from piton.qpu.isa import AluOp, Imm, Instr


def branch():
    return Instr(kind="branch", bcond="anyn", target="loop")


def compare(dst=isa.NOP, source=isa.RA[5]):
    return Instr(add=AluOp("sub", dst, source, isa.RB[1]), setf=True)


def mac(dst, source):
    return Instr(add=AluOp("fadd", dst, isa.R0, isa.RB[0]),
                 mul=AluOp("fmul", isa.R0, source, isa.RB[0]))


def test_fill_three_slots_by_hoisting_independent_counter():
    first = mac(isa.RA[3], isa.RA[0])
    second = mac(isa.RA[0], isa.RA[1])
    third = mac(isa.RA[1], isa.RA[2])
    increment = Instr(add=AluOp("add", isa.RA[5], isa.RA[5], Imm(1)))
    last = Instr(add=AluOp("fadd", isa.RA[2], isa.R0, isa.RB[0]))
    cmp, br = compare(), branch()
    body = [first, second, third, increment, last, cmp]
    result = fill_delay_slots(body, br)
    assert result == [increment, first, cmp, br, second, third, last]
    # The counter still has its required intervening instruction, and the
    # multiply/add dependency chain keeps its original arithmetic order.
    assert result.index(cmp) - result.index(increment) == 2
    assert body == [first, second, third, increment, last, cmp]


@pytest.mark.parametrize("dependency", ["raw", "war", "waw"])
def test_cannot_move_arithmetic_across_compare_dependency(dependency):
    if dependency == "raw":
        arithmetic = Instr(add=AluOp("add", isa.RA[0], isa.RA[0], Imm(1)))
        cmp = compare(source=isa.RA[0])
        body = [arithmetic, Instr.nop(), cmp]
    elif dependency == "war":
        arithmetic = Instr(add=AluOp("mov", isa.R0, isa.RA[0]))
        cmp = compare(dst=isa.RA[0], source=isa.RA[1])
        body = [arithmetic, cmp]
    else:
        arithmetic = Instr(add=AluOp("mov", isa.RA[0], isa.R0))
        cmp = compare(dst=isa.RA[0], source=isa.RA[1])
        body = [arithmetic, cmp]
    br = branch()
    result = fill_delay_slots(body, br)
    assert result.index(arithmetic) < result.index(br)


def test_conditional_arithmetic_keeps_original_flags():
    conditional = Instr(add=AluOp("mov", isa.R0, isa.RA[0], cond="ifz"))
    cmp, br = compare(), branch()
    result = fill_delay_slots([conditional, cmp], br)
    assert result.index(conditional) < result.index(cmp) < result.index(br)


@pytest.mark.parametrize("instruction", [
    Instr(add=AluOp("mov", isa.TMU_S[0], isa.RA[0])),
    Instr(add=AluOp("mov", isa.RA[0], isa.R4), sig="ldtmu0"),
    Instr(add=AluOp("mov", isa.UNIF_ADDR, isa.RA[0])),
])
def test_search_does_not_move_signals_or_peripheral_accesses(instruction):
    cmp, br = compare(), branch()
    result = fill_delay_slots([instruction, cmp], br)
    assert result.index(instruction) < result.index(br)


def test_reject_shortened_regfile_latency_when_producer_cannot_hoist():
    read_old = Instr(add=AluOp("mov", isa.R3, isa.RA[0]))
    write_new = Instr(add=AluOp("add", isa.RA[0], isa.RA[1], Imm(1)))
    gap = Instr(add=AluOp("mov", isa.R1, isa.R2))
    cmp, br = compare(source=isa.RA[0]), branch()
    result = fill_delay_slots([read_old, write_new, gap, cmp], br)
    # Moving gap would require hoisting write_new across a read of the old
    # value, so the original two-cycle write/compare distance must remain.
    assert result.index(cmp) - result.index(write_new) >= 2
    assert result.index(read_old) < result.index(write_new)
    assert result.index(gap) < result.index(br)


def test_reject_shortened_accumulator_rotation_latency():
    write = Instr(add=AluOp("mov", isa.R0, isa.RA[0]))
    gap = Instr(add=AluOp("mov", isa.R1, isa.R2))
    rotate_flags = Instr(mul=AluOp("mov", isa.R3, isa.R0, rot=1), setf=True)
    br = branch()
    result = fill_delay_slots([write, gap, rotate_flags], br)
    assert result.index(rotate_flags) - result.index(write) >= 2
    assert result.index(gap) < result.index(br)


def test_latency_repair_never_hoists_a_slot_instruction_across_branch():
    producer = Instr(add=AluOp("mov", isa.RA[0], isa.R1))
    gap = Instr(sig="ldtmu0")
    consumer = Instr(add=AluOp("mov", isa.R0, isa.RA[0]))
    cmp, br = compare(), branch()
    result = fill_delay_slots([producer, gap, consumer, cmp], br)
    # Moving producer and consumer together removes the intervening signal.
    # Repairing their RAW gap across the branch would alter the slot count.
    assert len(result) - result.index(br) - 1 == 3
    assert result.index(producer) < result.index(br) < result.index(consumer)
    assert result.index(consumer) - result.index(producer) >= 2


@pytest.mark.parametrize("indirect", [False, True])
def test_indirect_and_link_branches_keep_all_original_instructions_before_branch(indirect):
    body = [Instr(add=AluOp("mov", isa.ACC[i], isa.RA[i])) for i in range(3)]
    br = branch()
    if indirect:
        br.breg = isa.RA[0]
    else:
        br.add = AluOp("mov", isa.RA[0], isa.NOP)
    result = fill_delay_slots(body, br)
    assert result[:4] == body + [br]
    assert all(ins.add is None and ins.mul is None for ins in result[-3:])


def test_fully_occupied_trailing_slots_keep_existing_fast_path():
    cmp, br = compare(), branch()
    trailing = [Instr(add=AluOp("mov", isa.ACC[i], isa.RA[i])) for i in range(3)]
    assert fill_delay_slots([cmp, *trailing], br) == [cmp, br, *trailing]
