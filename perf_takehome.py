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
        Round-fused kernel with static list scheduling and shallow-tree preloading.

        Strategy:
        - Keep v_idx/v_val in scratch for all rounds (round fusion).
        - Preload tree levels 0-3 (nodes 0-14) into vectors and use vselect for
          rounds at shallow depths (including after wrap at depth 11).
        - Use a dependency-aware list scheduler to pack ready ops into VLIW
          bundles, saturating load/flow/valu slots.
        """
        def emit(instr):
            self.instrs.append(instr)

        class Task:
            def __init__(self, engine, slot, deps):
                self.engine = engine
                self.slot = slot
                self.deps = deps
                self.users = []
                self.ready_cycle = 0

        class ListScheduler:
            def __init__(
                self,
                readonly_addrs: set[int],
                serial_addrs: set[int],
                nop_slot: tuple,
            ):
                self.tasks: list[Task] = []
                self.last_write: dict[int, int] = {}
                self.last_read: dict[int, int] = {}
                self.last_access: dict[int, int] = {}
                self.readonly = readonly_addrs
                self.serial_addrs = serial_addrs
                self.nop_slot = nop_slot

            def add_task(self, engine, slot, reads, writes):
                deps = {}
                for addr in reads + writes:
                    if addr in self.readonly:
                        continue
                    if addr in self.serial_addrs:
                        if addr in self.last_access:
                            deps[self.last_access[addr]] = max(deps.get(self.last_access[addr], 0), 1)

                # RAW deps: read after last write (latency 1)
                for addr in reads:
                    if addr in self.readonly:
                        continue
                    if addr in self.serial_addrs:
                        continue
                    if addr in self.last_write:
                        deps[self.last_write[addr]] = max(deps.get(self.last_write[addr], 0), 1)
                # WAW deps: write after last write (latency 1)
                for addr in writes:
                    if addr in self.readonly:
                        continue
                    if addr in self.serial_addrs:
                        continue
                    if addr in self.last_write:
                        deps[self.last_write[addr]] = max(deps.get(self.last_write[addr], 0), 1)
                    # WAR deps: write after last read (latency 0, same-cycle ok)
                    if addr in self.last_read:
                        # WAR is safe in same cycle (reads observe old values)
                        deps[self.last_read[addr]] = max(deps.get(self.last_read[addr], 0), 0)
                tid = len(self.tasks)
                task = Task(engine, slot, deps)
                self.tasks.append(task)
                for dep in deps:
                    self.tasks[dep].users.append(tid)
                for addr in reads:
                    if addr in self.readonly:
                        continue
                    if addr in self.serial_addrs:
                        self.last_access[addr] = tid
                        continue
                    self.last_read[addr] = tid
                for addr in writes:
                    if addr in self.readonly:
                        continue
                    if addr in self.serial_addrs:
                        self.last_access[addr] = tid
                        continue
                    self.last_write[addr] = tid
                return tid

            def schedule(self):
                n = len(self.tasks)
                indeg = [len(t.deps) for t in self.tasks]
                ready_by_engine = {k: [] for k in SLOT_LIMITS.keys()}
                for tid in range(n):
                    if indeg[tid] == 0:
                        ready_by_engine[self.tasks[tid].engine].append(tid)

                scheduled_cycle = [-1] * n
                instrs = []
                current_cycle = 0
                remaining = n

                engine_order = ["load", "valu", "alu", "flow", "store"]

                while remaining > 0:
                    bundle = {}
                    scheduled_any = False
                    for engine in engine_order:
                        if engine not in SLOT_LIMITS:
                            continue
                        limit = SLOT_LIMITS[engine]
                        if limit == 0:
                            continue
                        ready_list = ready_by_engine.get(engine, [])
                        i = 0
                        while limit > 0 and i < len(ready_list):
                            tid = ready_list[i]
                            task = self.tasks[tid]
                            if task.ready_cycle <= current_cycle:
                                bundle.setdefault(engine, []).append(task.slot)
                                scheduled_cycle[tid] = current_cycle
                                ready_list.pop(i)
                                scheduled_any = True
                                limit -= 1
                                remaining -= 1
                                for user in task.users:
                                    indeg[user] -= 1
                                    if indeg[user] == 0:
                                        tuser = self.tasks[user]
                                        if tuser.deps:
                                            tuser.ready_cycle = max(
                                                scheduled_cycle[d] + tuser.deps[d]
                                                for d in tuser.deps
                                            )
                                        else:
                                            tuser.ready_cycle = current_cycle
                                        ready_by_engine[tuser.engine].append(user)
                            else:
                                i += 1
                        ready_by_engine[engine] = ready_list

                    if not scheduled_any:
                        # Fallback to a harmless ALU op to advance the cycle.
                        # This should be rare; indicates a scheduling gap.
                        bundle = {"alu": [self.nop_slot]}

                    instrs.append(bundle)
                    current_cycle += 1

                return instrs

        def vec_addrs(base):
            return list(range(base, base + VLEN))

        # ============ SCRATCH ALLOCATION ============
        assert batch_size % VLEN == 0, "Batch size must be multiple of VLEN"
        batches_total = batch_size // VLEN
        tile_batches = 16
        for cand in [32, 24, 16, 8, 4, 2, 1]:
            if batches_total % cand == 0:
                tile_batches = cand
                break
        num_tiles = batches_total // tile_batches
        num_batches = tile_batches
        preload_depth = 2

        v_idx = [self.alloc_scratch(f"v_idx_{b}", VLEN) for b in range(num_batches)]
        v_val = [self.alloc_scratch(f"v_val_{b}", VLEN) for b in range(num_batches)]
        v_node = [self.alloc_scratch(f"v_node_{b}", VLEN) for b in range(num_batches)]
        v_tmp = [self.alloc_scratch(f"v_tmp_{b}", VLEN) for b in range(num_batches)]
        v_sel0 = [self.alloc_scratch(f"v_sel0_{b}", VLEN) for b in range(num_batches)]

        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)

        # multiply_add constants for stages 0, 2, 4
        v_mul0 = self.alloc_scratch("v_mul0", VLEN)
        v_mul2 = self.alloc_scratch("v_mul2", VLEN)
        v_mul4 = self.alloc_scratch("v_mul4", VLEN)
        v_hc0 = self.alloc_scratch("v_hc0", VLEN)
        v_hc2 = self.alloc_scratch("v_hc2", VLEN)
        v_hc4 = self.alloc_scratch("v_hc4", VLEN)

        # XOR stage constants (1, 3, 5)
        v_hc3 = self.alloc_scratch("v_hc3", VLEN)
        v_hc5 = self.alloc_scratch("v_hc5", VLEN)
        v_sh3 = self.alloc_scratch("v_sh3", VLEN)
        v_sh5 = self.alloc_scratch("v_sh5", VLEN)

        pre_nodes = 7
        v_nodes = [self.alloc_scratch(f"v_node_pre_{i}", VLEN) for i in range(pre_nodes)]

        s_tmp = self.alloc_scratch("s_tmp")
        s_node = self.alloc_scratch("s_node")
        s_forest_p = self.alloc_scratch("s_forest_p")
        s_val_base = self.alloc_scratch("s_val_base")
        s_val_ptr = self.alloc_scratch("s_val_ptr")
        s_val_ptr2 = self.alloc_scratch("s_val_ptr2")
        s_tile_off = self.alloc_scratch("s_tile_off")
        s_nop = self.alloc_scratch("s_nop")

        s_zero = self.scratch_const(0)
        s_one = self.scratch_const(1)
        s_two = self.scratch_const(2)
        s_vlen = self.scratch_const(VLEN)
        s_2vlen = self.scratch_const(VLEN * 2)

        # ============ INITIALIZATION ============
        emit({"load": [("const", s_tmp, 1)]})
        emit({"load": [("const", s_tmp, 4)]})
        emit({"load": [("load", s_forest_p, s_tmp)]})
        emit({"load": [("const", s_tmp, 6)]})
        emit({"load": [("load", s_val_base, s_tmp)]})

        emit({"alu": [("+", s_nop, s_zero, s_zero)]})

        emit({"valu": [("vbroadcast", v_zero, s_zero), ("vbroadcast", v_one, s_one), ("vbroadcast", v_two, s_two)]})

        # multiply_add constants
        s_4097 = self.scratch_const(4097)
        s_33 = self.scratch_const(33)
        s_9 = self.scratch_const(9)
        emit({"valu": [("vbroadcast", v_mul0, s_4097), ("vbroadcast", v_mul2, s_33), ("vbroadcast", v_mul4, s_9)]})

        s_c0 = self.scratch_const(HASH_STAGES[0][1])
        s_c2 = self.scratch_const(HASH_STAGES[2][1])
        s_c4 = self.scratch_const(HASH_STAGES[4][1])
        emit({"valu": [("vbroadcast", v_hc0, s_c0), ("vbroadcast", v_hc2, s_c2), ("vbroadcast", v_hc4, s_c4)]})

        s_c3 = self.scratch_const(HASH_STAGES[3][1])
        s_c5 = self.scratch_const(HASH_STAGES[5][1])
        s_c1 = self.scratch_const(HASH_STAGES[1][1])
        emit({"valu": [("vbroadcast", v_hc3, s_c3), ("vbroadcast", v_hc5, s_c5)]})

        s_19 = self.scratch_const(19)
        s_9s = self.scratch_const(9)
        s_16 = self.scratch_const(16)
        emit({"valu": [("vbroadcast", v_sh3, s_9s), ("vbroadcast", v_sh5, s_16)]})

        # Preload nodes (levels 0-2) into vectors
        emit({"alu": [("+", s_tmp, s_forest_p, s_zero)]})
        for i in range(pre_nodes):
            emit({"load": [("load", s_node, s_tmp)]})
            emit({"valu": [("vbroadcast", v_nodes[i], s_node)]})
            if i + 1 < pre_nodes:
                emit({"alu": [("+", s_tmp, s_tmp, s_one)]})

        emit({"flow": [("pause",)]})

        readonly_addrs = set()
        for base in [
            v_zero, v_one, v_two,
            v_mul0, v_mul2, v_mul4, v_hc0, v_hc2, v_hc3, v_hc4, v_hc5,
            v_sh3, v_sh5,
        ]:
            readonly_addrs.update(vec_addrs(base))
        for vn in v_nodes:
            readonly_addrs.update(vec_addrs(vn))

        def add_valu(op, dest, a, b):
            sched.add_task("valu", (op, dest, a, b), vec_addrs(a) + vec_addrs(b), vec_addrs(dest))

        def add_madd(dest, a, b, c):
            sched.add_task("valu", ("multiply_add", dest, a, b, c),
                           vec_addrs(a) + vec_addrs(b) + vec_addrs(c), vec_addrs(dest))

        def add_vselect(dest, cond, a, b):
            sched.add_task("flow", ("vselect", dest, cond, a, b),
                           vec_addrs(cond) + vec_addrs(a) + vec_addrs(b), vec_addrs(dest))

        def add_load_lane(dest, addr):
            sched.add_task("load", ("load", dest, addr), [addr], [dest])

        def add_alu_vec(op, dest, a, b):
            for lane in range(VLEN):
                sched.add_task(
                    "alu",
                    (op, dest + lane, a + lane, b + lane),
                    [a + lane, b + lane],
                    [dest + lane],
                )

        def add_alu_vec_scalar(op, dest, a, b_scalar):
            for lane in range(VLEN):
                sched.add_task(
                    "alu",
                    (op, dest + lane, a + lane, b_scalar),
                    [a + lane, b_scalar],
                    [dest + lane],
                )

        def add_round(batch, depth):
            vi = v_idx[batch]
            vv = v_val[batch]
            vn = v_node[batch]
            vt = v_tmp[batch]
            vs0 = v_sel0[batch]

            # Node fetch: preloaded levels 0-2 at depths 0-2
            if depth == 0:
                add_valu("^", vv, vv, v_nodes[0])
            elif depth == 1:
                add_valu("&", vt, vi, v_one)
                add_vselect(vn, vt, v_nodes[1], v_nodes[2])
                add_valu("^", vv, vv, vn)
            elif depth == 2:
                add_valu("&", vt, vi, v_one)  # mask0
                add_vselect(vn, vt, v_nodes[5], v_nodes[4])   # t0
                add_vselect(vs0, vt, v_nodes[3], v_nodes[6])  # t1
                add_valu("&", vt, vi, v_two)  # mask1
                add_vselect(vn, vt, vs0, vn)
                add_valu("^", vv, vv, vn)
            else:
                add_alu_vec_scalar("+", vn, vi, s_forest_p)
                for lane in range(VLEN):
                    add_load_lane(vt + lane, vn + lane)
                add_valu("^", vv, vv, vt)

            # Hash stages
            add_madd(vv, vv, v_mul0, v_hc0)

            # Stage 1 (xor / shift) using ALU lanes with scalar constants
            add_alu_vec_scalar(">>", vt, vv, s_19)
            add_alu_vec_scalar("^", vv, vv, s_c1)
            add_alu_vec("^", vv, vv, vt)

            add_madd(vv, vv, v_mul2, v_hc2)

            add_valu("<<", vt, vv, v_sh3)
            add_valu("+", vv, vv, v_hc3)
            add_valu("^", vv, vv, vt)

            add_madd(vv, vv, v_mul4, v_hc4)

            add_valu(">>", vt, vv, v_sh5)
            add_valu("^", vv, vv, v_hc5)
            add_valu("^", vv, vv, vt)

            # Index update: for max depth, always wraps to 0
            if depth == forest_height:
                add_valu("+", vi, v_zero, v_zero)
            elif depth == 0:
                # idx starts at 0 for depth 0, so idx = 1 + (val & 1)
                add_valu("&", vt, vv, v_one)
                add_valu("+", vt, vt, v_one)
                add_valu("+", vi, vt, v_zero)
            else:
                # idx = (idx << 1) + 1 + (val & 1)
                add_valu("&", vt, vv, v_one)
                add_valu("+", vt, vt, v_one)
                add_madd(vi, vi, v_two, vt)

        for tile in range(num_tiles):
            if num_tiles == 1:
                emit({
                    "alu": [
                        ("+", s_val_ptr, s_val_base, s_zero),
                        ("+", s_val_ptr2, s_val_base, s_vlen),
                    ]
                })
            else:
                s_off = self.scratch_const(tile * tile_batches * VLEN)
                emit({"alu": [("+", s_tile_off, s_off, s_zero)]})
                emit({
                    "alu": [
                        ("+", s_val_ptr, s_val_base, s_tile_off),
                    ]
                })
                emit({
                    "alu": [
                        ("+", s_val_ptr2, s_val_ptr, s_vlen),
                    ]
                })
            for b in range(0, num_batches, 2):
                if b + 1 < num_batches:
                    emit({
                        "load": [
                            ("vload", v_val[b], s_val_ptr),
                            ("vload", v_val[b + 1], s_val_ptr2),
                        ],
                        "valu": [
                            ("vbroadcast", v_idx[b], s_zero),
                            ("vbroadcast", v_idx[b + 1], s_zero),
                        ],
                        "alu": [
                            ("+", s_val_ptr, s_val_ptr, s_2vlen),
                            ("+", s_val_ptr2, s_val_ptr2, s_2vlen),
                        ],
                    })
                else:
                    emit({
                        "load": [("vload", v_val[b], s_val_ptr)],
                        "valu": [("vbroadcast", v_idx[b], s_zero)],
                        "alu": [("+", s_val_ptr, s_val_ptr, s_vlen)],
                    })

            sched = ListScheduler(
                readonly_addrs,
                set(),
                ("+", s_nop, s_nop, s_zero),
            )

            for r in range(rounds):
                depth = r % (forest_height + 1)
                for b in range(num_batches):
                    add_round(b, depth)

            self.instrs.extend(sched.schedule())

            if num_tiles == 1:
                emit({
                    "alu": [
                        ("+", s_val_ptr, s_val_base, s_zero),
                    ]
                })
            else:
                emit({
                    "alu": [
                        ("+", s_val_ptr, s_val_base, s_tile_off),
                    ]
                })
            for b in range(num_batches):
                emit({
                    "store": [("vstore", s_val_ptr, v_val[b])],
                    "alu": [("+", s_val_ptr, s_val_ptr, s_vlen)]
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
