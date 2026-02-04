"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        SIMD Vectorized kernel with VLIW packing.

        Process 8 elements at a time. Pack independent operations into same cycle.
        Target: ~24 cycles per iteration = ~12,288 cycles total.

        Resources per cycle:
        - 6 VALU slots (vector ops on 8 elements)
        - 12 ALU slots (scalar ops)
        - 2 Load slots
        - 2 Store slots
        - 1 Flow slot (jumps, selects)
        """
        def emit(instr):
            self.instrs.append(instr)

        # ============ SCRATCH ALLOCATION ============

        # Vector registers (8 elements each)
        v_idx = self.alloc_scratch("v_idx", VLEN)
        v_val = self.alloc_scratch("v_val", VLEN)
        v_node = self.alloc_scratch("v_node", VLEN)
        v_addr = self.alloc_scratch("v_addr", VLEN)
        v_tmp1 = self.alloc_scratch("v_tmp1", VLEN)
        v_tmp2 = self.alloc_scratch("v_tmp2", VLEN)

        # Vector constants
        v_forest_p = self.alloc_scratch("v_forest_p", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)

        # Hash constants (6 stages × 2 constants each)
        v_hc1 = [self.alloc_scratch(f"v_hc1_{i}", VLEN) for i in range(6)]
        v_hc3 = [self.alloc_scratch(f"v_hc3_{i}", VLEN) for i in range(6)]

        # Scalar registers
        s_tmp = self.alloc_scratch("s_tmp")
        s_batch = self.alloc_scratch("s_batch")
        s_idx_ptr = self.alloc_scratch("s_idx_ptr")
        s_val_ptr = self.alloc_scratch("s_val_ptr")
        s_save_idx = self.alloc_scratch("s_save_idx")  # saved ptr for store
        s_save_val = self.alloc_scratch("s_save_val")
        s_n_nodes = self.alloc_scratch("s_n_nodes")
        s_forest_p = self.alloc_scratch("s_forest_p")
        s_idx_base = self.alloc_scratch("s_idx_base")
        s_val_base = self.alloc_scratch("s_val_base")
        s_cond = self.alloc_scratch("s_cond")

        # Scalar constants
        s_zero = self.scratch_const(0)
        s_one = self.scratch_const(1)
        s_two = self.scratch_const(2)
        s_vlen = self.scratch_const(VLEN)

        num_batches = (batch_size // VLEN) * rounds  # 32 * 16 = 512
        s_total = self.scratch_const(num_batches)
        s_bpr = self.scratch_const(batch_size // VLEN)  # 32 batches per round

        # ============ INITIALIZATION ============

        # Load header values from memory (addresses 1, 4, 5, 6)
        emit({"load": [("const", s_tmp, 1)]})
        emit({"load": [("load", s_n_nodes, s_tmp)]})
        emit({"load": [("const", s_tmp, 4)]})
        emit({"load": [("load", s_forest_p, s_tmp)]})
        emit({"load": [("const", s_tmp, 5)]})
        emit({"load": [("load", s_idx_base, s_tmp)]})
        emit({"load": [("const", s_tmp, 6)]})
        emit({"load": [("load", s_val_base, s_tmp)]})

        # Broadcast scalars to vectors (pack 2 per cycle)
        emit({"valu": [("vbroadcast", v_forest_p, s_forest_p), ("vbroadcast", v_n_nodes, s_n_nodes)]})
        emit({"valu": [("vbroadcast", v_zero, s_zero), ("vbroadcast", v_one, s_one)]})
        emit({"valu": [("vbroadcast", v_two, s_two)]})

        # Broadcast hash constants (2 per cycle)
        for i in range(6):
            c1 = self.scratch_const(HASH_STAGES[i][1])
            c3 = self.scratch_const(HASH_STAGES[i][4])
            emit({"valu": [("vbroadcast", v_hc1[i], c1), ("vbroadcast", v_hc3[i], c3)]})

        emit({"flow": [("pause",)]})

        # Initialize pointers and batch counter
        emit({"alu": [
            ("+", s_idx_ptr, s_idx_base, s_zero),
            ("+", s_val_ptr, s_val_base, s_zero)
        ]})
        emit({"load": [("const", s_batch, 0)]})

        # ============ MAIN LOOP ============
        loop_start = len(self.instrs)

        # --- Cycle 1: Load idx and val vectors ---
        emit({"load": [("vload", v_idx, s_idx_ptr), ("vload", v_val, s_val_ptr)]})

        # --- Cycle 2: Compute gather addresses + save pointers for later store ---
        emit({
            "valu": [("+", v_addr, v_forest_p, v_idx)],
            "alu": [
                ("+", s_save_idx, s_idx_ptr, s_zero),
                ("+", s_save_val, s_val_ptr, s_zero)
            ]
        })

        # --- Cycles 3-6: Gather node values + loop bookkeeping ---
        # Cycle 3: gather[0:2] + batch++ + advance pointers
        emit({
            "load": [("load", v_node + 0, v_addr + 0), ("load", v_node + 1, v_addr + 1)],
            "alu": [
                ("+", s_batch, s_batch, s_one),
                ("+", s_idx_ptr, s_idx_ptr, s_vlen),
                ("+", s_val_ptr, s_val_ptr, s_vlen)
            ]
        })

        # Cycle 4: gather[2:4] + compute wrap condition (batch % bpr)
        emit({
            "load": [("load", v_node + 2, v_addr + 2), ("load", v_node + 3, v_addr + 3)],
            "alu": [("%", s_tmp, s_batch, s_bpr)]
        })

        # Cycle 5: gather[4:6] + check if wrap needed (tmp == 0)
        emit({
            "load": [("load", v_node + 4, v_addr + 4), ("load", v_node + 5, v_addr + 5)],
            "alu": [("==", s_tmp, s_tmp, s_zero)]
        })

        # Cycle 6: gather[6:8] + apply wrap to idx_ptr
        emit({
            "load": [("load", v_node + 6, v_addr + 6), ("load", v_node + 7, v_addr + 7)],
            "flow": [("select", s_idx_ptr, s_tmp, s_idx_base, s_idx_ptr)]
        })

        # --- Cycle 7: XOR + apply wrap to val_ptr ---
        emit({
            "valu": [("^", v_val, v_val, v_node)],
            "flow": [("select", s_val_ptr, s_tmp, s_val_base, s_val_ptr)]
        })

        # --- Cycles 8-19: Hash computation (6 stages × 2 cycles) ---
        for i in range(6):
            op1, _, op2, op3, _ = HASH_STAGES[i]
            # Parallel ops: tmp1 = op1(val, c1), tmp2 = op3(val, c3)
            emit({"valu": [(op1, v_tmp1, v_val, v_hc1[i]), (op3, v_tmp2, v_val, v_hc3[i])]})
            # Combine: val = op2(tmp1, tmp2)
            emit({"valu": [(op2, v_val, v_tmp1, v_tmp2)]})

        # --- Cycles 20-23: Index computation (optimized - no vselect!) ---
        # Formula: new_idx = 2*idx + (1 if even else 2) = 2*idx + 2 - is_even
        # where is_even = (val & 1) == 0

        # Cycle 20: val & 1, idx << 1 (parallel)
        emit({"valu": [
            ("&", v_tmp1, v_val, v_one),   # tmp1 = val & 1 (0 if even, 1 if odd)
            ("<<", v_idx, v_idx, v_one)    # idx = idx << 1
        ]})

        # Cycle 21: is_even test + idx += 2 (parallel - no dependency!)
        emit({
            "valu": [
                ("==", v_tmp1, v_tmp1, v_zero),  # tmp1 = (val&1)==0 (1 if even, 0 if odd)
                ("+", v_idx, v_idx, v_two)       # idx = 2*old_idx + 2
            ],
            "alu": [("<", s_cond, s_batch, s_total)]  # loop condition
        })

        # Cycle 22: idx -= is_even (gives 2*idx+1 if even, 2*idx+2 if odd)
        emit({"valu": [("-", v_idx, v_idx, v_tmp1)]})

        # Cycle 23: check bounds (idx < n_nodes) - tmp1 = 1 if valid, 0 if overflow
        emit({"valu": [("<", v_tmp1, v_idx, v_n_nodes)]})

        # Cycle 24: Wrap index using multiply (no vselect needed!)
        # idx * 1 = idx (valid), idx * 0 = 0 (overflow)
        emit({"valu": [("*", v_idx, v_idx, v_tmp1)]})

        # Cycle 25: Store both + jump
        emit({
            "store": [("vstore", s_save_idx, v_idx), ("vstore", s_save_val, v_val)],
            "flow": [("cond_jump", s_cond, loop_start)]
        })

        emit({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
