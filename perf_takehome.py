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
import os
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
        self,
        forest_height: int,
        n_nodes: int,
        batch_size: int,
        rounds: int,
        use_state_machine: bool = False,
    ):
        """
        Round-fused kernel with static list scheduling and shallow-tree preloading.

        Strategy:
        - Keep v_idx/v_val in scratch for all rounds (round fusion).
        - Preload tree levels 0-3 (nodes 0-14) into vectors and use vselect for
          rounds at shallow depths.
        - Use a dependency-aware list scheduler to pack ready ops into VLIW
          bundles, saturating load/flow/valu slots.
        """
        use_state_machine = use_state_machine or os.getenv("STATE_MACHINE") == "1"
        use_aggressive = os.getenv("AGGRESSIVE") == "1"
        use_pipeline = os.getenv("PIPELINE") == "1"
        use_static = os.getenv("STATIC", "1") == "1"

        def emit(instr):
            self.instrs.append(instr)

        class Task:
            def __init__(self, engine, slot, deps, tag, stage=None, batch_round=None):
                self.engine = engine
                self.slot = slot
                self.deps = deps
                self.tag = tag
                self.stage = stage
                self.batch_round = batch_round
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

            def add_task(self, engine, slot, reads, writes, tag="other", stage=None, batch_round=None):
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
                task = Task(engine, slot, deps, tag, stage=stage, batch_round=batch_round)
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
                        bundle = {"alu": [self.nop_slot]}

                    instrs.append(bundle)
                    current_cycle += 1

                return instrs

        class StaticScheduler(ListScheduler):
            """
            Offline list scheduler: place each task at the earliest cycle
            with an available slot for its engine, based on dependency readiness.
            """
            def schedule(self):
                import heapq

                n = len(self.tasks)
                indeg = [len(t.deps) for t in self.tasks]
                scheduled_cycle = [-1] * n

                # Track per-engine slot usage per cycle
                usage = {engine: [] for engine in SLOT_LIMITS.keys()}

                # Compute a simple critical-path height for tie-breaking
                indeg2 = indeg[:]
                topo = []
                queue = [tid for tid in range(n) if indeg2[tid] == 0]
                while queue:
                    tid = queue.pop()
                    topo.append(tid)
                    for user in self.tasks[tid].users:
                        indeg2[user] -= 1
                        if indeg2[user] == 0:
                            queue.append(user)
                height = [1] * n
                for tid in reversed(topo):
                    if self.tasks[tid].users:
                        height[tid] = 1 + max(height[u] for u in self.tasks[tid].users)

                # Ready queue ordered by earliest dependency-ready cycle, then critical-path height,
                # then engine priority (favor constrained engines).
                engine_prio = {
                    "flow": 0,
                    "load": 1,
                    "valu": 2,
                    "alu": 3,
                    "store": 4,
                    "debug": 5,
                }
                ready = []
                for tid in range(n):
                    if indeg[tid] == 0:
                        prio = engine_prio.get(self.tasks[tid].engine, 9)
                        heapq.heappush(ready, (0, -height[tid], prio, tid))

                while ready:
                    ready_cycle, _neg_h, _prio, tid = heapq.heappop(ready)
                    task = self.tasks[tid]
                    engine = task.engine
                    limit = SLOT_LIMITS.get(engine, 0)
                    cycle = ready_cycle
                    if limit == 0:
                        # Should not happen, but avoid infinite loop
                        cycle = ready_cycle
                    while True:
                        if cycle >= len(usage[engine]):
                            usage[engine].extend([0] * (cycle - len(usage[engine]) + 1))
                        if usage[engine][cycle] < limit:
                            usage[engine][cycle] += 1
                            scheduled_cycle[tid] = cycle
                            break
                        cycle += 1

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
                                tuser.ready_cycle = 0
                            prio = engine_prio.get(self.tasks[user].engine, 9)
                            heapq.heappush(ready, (tuser.ready_cycle, -height[user], prio, user))

                max_cycle = max(scheduled_cycle) if scheduled_cycle else -1
                instrs = []
                for _ in range(max_cycle + 1):
                    instrs.append({})

                for tid, cycle in enumerate(scheduled_cycle):
                    if cycle < 0:
                        continue
                    task = self.tasks[tid]
                    instrs[cycle].setdefault(task.engine, []).append(task.slot)

                # Fill any completely empty cycles with a harmless ALU op
                for i, bundle in enumerate(instrs):
                    if not bundle:
                        instrs[i] = {"alu": [self.nop_slot]}

                return instrs

        class StateMachineScheduler(ListScheduler):
            def __init__(
                self,
                readonly_addrs: set[int],
                serial_addrs: set[int],
                nop_slot: tuple,
            ):
                super().__init__(readonly_addrs, serial_addrs, nop_slot)
                self.stage_tasks = defaultdict(list)
                self.stage_counts = defaultdict(int)
                self.batch_stage = {}
                self.batch_max_stage = {}

            def add_task(self, engine, slot, reads, writes, tag="other", stage=None, batch_round=None):
                tid = super().add_task(engine, slot, reads, writes, tag=tag, stage=stage, batch_round=batch_round)
                if stage is not None and batch_round is not None:
                    self.stage_tasks[(batch_round, stage)].append(tid)
                    self.stage_counts[(batch_round, stage)] += 1
                    if batch_round not in self.batch_stage:
                        self.batch_stage[batch_round] = stage
                    else:
                        self.batch_stage[batch_round] = min(self.batch_stage[batch_round], stage)
                    self.batch_max_stage[batch_round] = max(self.batch_max_stage.get(batch_round, stage), stage)
                return tid

            def _stage_ready(self, task: Task) -> bool:
                if task.stage is None or task.batch_round is None:
                    return True
                return self.batch_stage.get(task.batch_round, task.stage) == task.stage

            def _advance_stage(self, batch_round):
                next_stage = self.batch_stage.get(batch_round, 0) + 1
                while next_stage <= self.batch_max_stage.get(batch_round, -1):
                    if self.stage_counts.get((batch_round, next_stage), 0) > 0:
                        self.batch_stage[batch_round] = next_stage
                        return
                    next_stage += 1
                self.batch_stage[batch_round] = next_stage

            def schedule(self):
                n = len(self.tasks)
                indeg = [len(t.deps) for t in self.tasks]
                ready_by_engine = {k: [] for k in SLOT_LIMITS.keys()}
                for tid in range(n):
                    if indeg[tid] == 0 and self._stage_ready(self.tasks[tid]):
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
                            if task.ready_cycle <= current_cycle and self._stage_ready(task):
                                bundle.setdefault(engine, []).append(task.slot)
                                scheduled_cycle[tid] = current_cycle
                                ready_list.pop(i)
                                scheduled_any = True
                                limit -= 1
                                remaining -= 1
                                if task.stage is not None and task.batch_round is not None:
                                    key = (task.batch_round, task.stage)
                                    self.stage_counts[key] -= 1
                                    if self.stage_counts[key] == 0:
                                        self._advance_stage(task.batch_round)
                                        next_stage = self.batch_stage.get(task.batch_round, task.stage + 1)
                                        for tid2 in self.stage_tasks.get((task.batch_round, next_stage), []):
                                            if indeg[tid2] == 0 and self._stage_ready(self.tasks[tid2]):
                                                ready_by_engine[self.tasks[tid2].engine].append(tid2)
                                for user in task.users:
                                    indeg[user] -= 1
                                    if indeg[user] == 0 and self._stage_ready(self.tasks[user]):
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
        if use_aggressive and batches_total == 32:
            tile_sizes = [24, 8]
            tile_batches = 24
        else:
            for cand in [32, 24, 16, 8, 4, 2, 1]:
                if batches_total % cand == 0:
                    tile_batches = cand
                    break
            tile_sizes = [tile_batches] * (batches_total // tile_batches)
        num_tiles = len(tile_sizes)
        num_batches = max(tile_sizes)
        preload_depth = 3

        v_idx = [self.alloc_scratch(f"v_idx_{b}", VLEN) for b in range(num_batches)]
        v_val = [self.alloc_scratch(f"v_val_{b}", VLEN) for b in range(num_batches)]
        v_tmp = [self.alloc_scratch(f"v_tmp_{b}", VLEN) for b in range(num_batches)]
        v_sel0 = [self.alloc_scratch(f"v_sel0_{b}", VLEN) for b in range(num_batches)]
        if use_pipeline:
            v_node = None
            v_sel1 = None
            v_sel1_shared = self.alloc_scratch("v_sel1_shared", VLEN)
            v_sel2_shared = self.alloc_scratch("v_sel2_shared", VLEN)
        else:
            v_node = [self.alloc_scratch(f"v_node_{b}", VLEN) for b in range(num_batches)]
            if use_aggressive:
                v_sel1 = [self.alloc_scratch(f"v_sel1_{b}", VLEN) for b in range(num_batches)]
                v_sel1_shared = None
                v_sel2_shared = self.alloc_scratch("v_sel2_shared", VLEN)
            else:
                v_sel1 = None
                v_sel1_shared = self.alloc_scratch("v_sel1_shared", VLEN)
                v_sel2_shared = None

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
        # XOR stage shift constants (3, 5)
        v_sh3 = self.alloc_scratch("v_sh3", VLEN)
        v_sh5 = self.alloc_scratch("v_sh5", VLEN)
        if use_pipeline:
            v_hc1 = self.alloc_scratch("v_hc1", VLEN)
            v_sh1 = self.alloc_scratch("v_sh1", VLEN)
            v_forest_p = self.alloc_scratch("v_forest_p", VLEN)
        else:
            v_hc1 = None
            v_sh1 = None
            v_forest_p = None

        pre_nodes = 15
        v_nodes = [self.alloc_scratch(f"v_node_pre_{i}", VLEN) for i in range(pre_nodes)]

        s_tmp = self.alloc_scratch("s_tmp")
        s_node = self.alloc_scratch("s_node")
        s_forest_p = self.alloc_scratch("s_forest_p")
        s_val_base = self.alloc_scratch("s_val_base")
        s_val_ptr = self.alloc_scratch("s_val_ptr")
        s_val_ptr2 = self.alloc_scratch("s_val_ptr2")
        s_tile_off = self.alloc_scratch("s_tile_off") if num_tiles > 1 else None
        s_nop = self.alloc_scratch("s_nop")

        s_zero = self.scratch_const(0)
        s_one = self.scratch_const(1)
        s_three = self.scratch_const(3)
        s_seven = self.scratch_const(7)
        s_vlen = self.scratch_const(VLEN)
        s_2vlen = self.scratch_const(VLEN * 2)

        # ============ INITIALIZATION ============
        emit({"load": [("const", s_tmp, 1)]})
        emit({"load": [("const", s_tmp, 4)]})
        emit({"load": [("load", s_forest_p, s_tmp)]})
        emit({"load": [("const", s_tmp, 6)]})
        emit({"load": [("load", s_val_base, s_tmp)]})

        emit({"alu": [("+", s_nop, s_zero, s_zero)]})

        emit({
            "valu": [
                ("vbroadcast", v_zero, s_zero),
                ("vbroadcast", v_one, s_one),
            ]
        })
        emit({"valu": [("+", v_two, v_one, v_one)]})
        if use_pipeline:
            emit({"valu": [("vbroadcast", v_forest_p, s_forest_p)]})

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
        if use_pipeline:
            emit({"valu": [("vbroadcast", v_hc1, s_c1), ("vbroadcast", v_sh1, s_19)]})
        s_9s = self.scratch_const(9)
        s_16 = self.scratch_const(16)
        emit({"valu": [("vbroadcast", v_sh3, s_9s), ("vbroadcast", v_sh5, s_16)]})

        # Preload nodes: vectors for 0-14 (pipelined)
        emit({"alu": [("+", s_tmp, s_forest_p, s_zero)]})
        emit({"load": [("load", s_node, s_tmp)]})
        for i in range(pre_nodes - 1):
            emit({
                "valu": [("vbroadcast", v_nodes[i], s_node)],
                "alu": [("+", s_tmp, s_tmp, s_one)],
            })
            emit({"load": [("load", s_node, s_tmp)]})
        emit({"valu": [("vbroadcast", v_nodes[pre_nodes - 1], s_node)]})

        emit({"flow": [("pause",)]})

        readonly_addrs = set()
        for base in [
            v_zero, v_one, v_two,
            v_mul0, v_mul2, v_mul4, v_hc0, v_hc2, v_hc3, v_hc4, v_hc5,
            v_sh3, v_sh5,
        ]:
            readonly_addrs.update(vec_addrs(base))
        if use_pipeline:
            readonly_addrs.update(vec_addrs(v_hc1))
            readonly_addrs.update(vec_addrs(v_sh1))
            readonly_addrs.update(vec_addrs(v_forest_p))
        for vn in v_nodes:
            readonly_addrs.update(vec_addrs(vn))

        def add_valu(op, dest, a, b, tag="other", stage=None, batch_round=None):
            sched.add_task(
                "valu",
                (op, dest, a, b),
                vec_addrs(a) + vec_addrs(b),
                vec_addrs(dest),
                tag=tag,
                stage=stage,
                batch_round=batch_round,
            )

        def add_madd(dest, a, b, c, tag="other", stage=None, batch_round=None):
            sched.add_task(
                "valu",
                ("multiply_add", dest, a, b, c),
                vec_addrs(a) + vec_addrs(b) + vec_addrs(c),
                vec_addrs(dest),
                tag=tag,
                stage=stage,
                batch_round=batch_round,
            )

        def add_vselect(dest, cond, a, b, tag="other", stage=None, batch_round=None):
            sched.add_task(
                "flow",
                ("vselect", dest, cond, a, b),
                vec_addrs(cond) + vec_addrs(a) + vec_addrs(b),
                vec_addrs(dest),
                tag=tag,
                stage=stage,
                batch_round=batch_round,
            )

        def add_select_valu(dest, a, b, mask, tmp):
            # dest = a ^ (mask & (a ^ b))
            add_valu("^", tmp, a, b)
            add_valu("&", tmp, tmp, mask)
            add_valu("^", dest, a, tmp)

        def add_mask_from_bit(dest, bit_vec, tag="select", stage=None, batch_round=None):
            add_valu("-", dest, v_zero, bit_vec, tag=tag, stage=stage, batch_round=batch_round)

        def add_select_masked(dest, a, b, mask, tmp, tag="select", stage=None, batch_round=None):
            # dest = b ^ (mask & (a ^ b))  (mask=-1 selects a)
            add_valu("^", tmp, a, b, tag=tag, stage=stage, batch_round=batch_round)
            add_valu("&", tmp, tmp, mask, tag=tag, stage=stage, batch_round=batch_round)
            add_valu("^", dest, b, tmp, tag=tag, stage=stage, batch_round=batch_round)

        def add_load_lane(dest, addr, tag="other", stage=None, batch_round=None):
            sched.add_task(
                "load",
                ("load", dest, addr),
                [addr],
                [dest],
                tag=tag,
                stage=stage,
                batch_round=batch_round,
            )

        def add_alu_vec(op, dest, a, b, tag="other", stage=None, batch_round=None):
            for lane in range(VLEN):
                sched.add_task(
                    "alu",
                    (op, dest + lane, a + lane, b + lane),
                    [a + lane, b + lane],
                    [dest + lane],
                    tag=tag,
                    stage=stage,
                    batch_round=batch_round,
                )

        def add_alu_vec_scalar(op, dest, a, b_scalar, tag="other", stage=None, batch_round=None):
            for lane in range(VLEN):
                sched.add_task(
                    "alu",
                    (op, dest + lane, a + lane, b_scalar),
                    [a + lane, b_scalar],
                    [dest + lane],
                    tag=tag,
                    stage=stage,
                    batch_round=batch_round,
                )

        def add_round(batch, depth, batch_round=None):
            vi = v_idx[batch]
            vv = v_val[batch]
            vn = v_node[batch] if v_node is not None else None
            vt = v_tmp[batch]
            vs0 = v_sel0[batch]
            br = batch_round
            st_fetch = 0
            st_xor = 1
            st_h0 = 2
            st_s1 = 3
            st_h2 = 4
            st_h3 = 5
            st_h4 = 6
            st_h5 = 7
            st_idx = 8

            # Node fetch: preloaded levels 0-3 at depths 0-3
            if use_pipeline:
                if depth == 0:
                    add_valu("^", vv, vv, v_nodes[0], tag="select", stage=st_xor, batch_round=br)
                elif depth == 1:
                    add_valu("&", vs0, vi, v_one, tag="select", stage=st_fetch, batch_round=br)
                    add_vselect(vt, vs0, v_nodes[1], v_nodes[2], tag="select", stage=st_fetch, batch_round=br)
                    add_valu("^", vv, vv, vt, tag="hash", stage=st_xor, batch_round=br)
                elif depth == 2:
                    vs1 = v_sel1_shared
                    add_alu_vec_scalar("-", vt, vi, s_three, tag="select", stage=st_fetch, batch_round=br)
                    add_valu("&", vs0, vt, v_one, tag="select", stage=st_fetch, batch_round=br)  # mask0
                    add_valu("&", vs1, vt, v_two, tag="select", stage=st_fetch, batch_round=br)  # mask1
                    add_vselect(vt, vs0, v_nodes[4], v_nodes[3], tag="select", stage=st_fetch, batch_round=br)   # pair0
                    add_vselect(vs0, vs0, v_nodes[6], v_nodes[5], tag="select", stage=st_fetch, batch_round=br)  # pair1
                    add_vselect(vt, vs1, vs0, vt, tag="select", stage=st_fetch, batch_round=br)
                    add_valu("^", vv, vv, vt, tag="hash", stage=st_xor, batch_round=br)
                elif depth == 3:
                    vs1 = v_sel1_shared
                    vs2 = v_sel2_shared
                    # quad0 (nodes 7-10) -> vt
                    add_alu_vec_scalar("-", vt, vi, s_seven, tag="select", stage=st_fetch, batch_round=br)
                    add_valu("&", vs0, vt, v_one, tag="select", stage=st_fetch, batch_round=br)  # mask0
                    add_valu("&", vs2, vt, v_two, tag="select", stage=st_fetch, batch_round=br)  # mask1
                    add_vselect(vt, vs0, v_nodes[8], v_nodes[7], tag="select", stage=st_fetch, batch_round=br)   # pair0
                    add_vselect(vs1, vs0, v_nodes[10], v_nodes[9], tag="select", stage=st_fetch, batch_round=br)  # pair1
                    add_vselect(vt, vs2, vs1, vt, tag="select", stage=st_fetch, batch_round=br)  # quad0

                    # quad1 (nodes 11-14) -> vs1
                    add_alu_vec_scalar("-", vs1, vi, s_seven, tag="select", stage=st_fetch, batch_round=br)
                    add_valu("&", vs0, vs1, v_one, tag="select", stage=st_fetch, batch_round=br)  # mask0
                    add_valu("&", vs2, vs1, v_two, tag="select", stage=st_fetch, batch_round=br)  # mask1
                    add_vselect(vs1, vs0, v_nodes[12], v_nodes[11], tag="select", stage=st_fetch, batch_round=br)  # pair2
                    add_vselect(vs0, vs0, v_nodes[14], v_nodes[13], tag="select", stage=st_fetch, batch_round=br)  # pair3
                    add_vselect(vs1, vs2, vs0, vs1, tag="select", stage=st_fetch, batch_round=br)  # quad1

                    # mask2 = (off >> 2) & 1
                    add_alu_vec_scalar("-", vs0, vi, s_seven, tag="select", stage=st_fetch, batch_round=br)
                    add_valu(">>", vs0, vs0, v_one, tag="select", stage=st_fetch, batch_round=br)
                    add_valu(">>", vs0, vs0, v_one, tag="select", stage=st_fetch, batch_round=br)
                    add_valu("&", vs0, vs0, v_one, tag="select", stage=st_fetch, batch_round=br)
                    add_vselect(vt, vs0, vs1, vt, tag="select", stage=st_fetch, batch_round=br)
                    add_valu("^", vv, vv, vt, tag="hash", stage=st_xor, batch_round=br)
                else:
                    # Depths 4+: load from memory using ALU address computation
                    add_valu("+", vt, vi, v_forest_p, tag="load", stage=st_fetch, batch_round=br)
                    for lane in range(VLEN):
                        add_load_lane(vt + lane, vt + lane, tag="load", stage=st_fetch, batch_round=br)
                    add_valu("^", vv, vv, vt, tag="hash", stage=st_xor, batch_round=br)
            else:
                if depth == 0:
                    add_valu("^", vv, vv, v_nodes[0], tag="select", stage=st_xor, batch_round=br)
                elif depth == 1:
                    add_valu("&", vt, vi, v_one, tag="select", stage=st_fetch, batch_round=br)
                    add_vselect(vn, vt, v_nodes[1], v_nodes[2], tag="select", stage=st_fetch, batch_round=br)
                    add_valu("^", vv, vv, vn, tag="hash", stage=st_xor, batch_round=br)
                elif depth == 2:
                    # off = (idx + 1) & 3, select nodes[3..6] by off bits
                    add_valu("+", vt, vi, v_one, tag="select", stage=st_fetch, batch_round=br)
                    add_valu("&", vs0, vt, v_one, tag="select", stage=st_fetch, batch_round=br)  # mask0 (bit0)
                    add_vselect(vn, vs0, v_nodes[4], v_nodes[3], tag="select", stage=st_fetch, batch_round=br)   # pair0: 3/4
                    add_vselect(vs0, vs0, v_nodes[6], v_nodes[5], tag="select", stage=st_fetch, batch_round=br)  # pair1: 5/6
                    add_valu("&", vt, vt, v_two, tag="select", stage=st_fetch, batch_round=br)  # mask1 (bit1)
                    add_vselect(vn, vt, vs0, vn, tag="select", stage=st_fetch, batch_round=br)
                    add_valu("^", vv, vv, vn, tag="hash", stage=st_xor, batch_round=br)
                elif depth == 3:
                    # off = idx - 7, select nodes[7..14] by off bits
                    if use_aggressive:
                        vs1 = v_sel1[batch]
                        vs2 = v_sel2_shared
                        # quad0 (nodes 7-10)
                        add_alu_vec_scalar("-", vt, vi, s_seven, tag="select", stage=st_fetch, batch_round=br)
                        add_valu("&", vs0, vt, v_one, tag="select", stage=st_fetch, batch_round=br)  # bit0
                        add_mask_from_bit(vs0, vs0, tag="select", stage=st_fetch, batch_round=br)  # mask0
                        add_select_masked(vn, v_nodes[8], v_nodes[7], vs0, vs2, tag="select", stage=st_fetch, batch_round=br)   # pair0
                        add_select_masked(vs1, v_nodes[10], v_nodes[9], vs0, vs2, tag="select", stage=st_fetch, batch_round=br)  # pair1
                        add_alu_vec_scalar("-", vt, vi, s_seven, tag="select", stage=st_fetch, batch_round=br)
                        add_valu(">>", vt, vt, v_one, tag="select", stage=st_fetch, batch_round=br)
                        add_valu("&", vt, vt, v_one, tag="select", stage=st_fetch, batch_round=br)  # bit1
                        add_mask_from_bit(vt, vt, tag="select", stage=st_fetch, batch_round=br)  # mask1
                        add_select_masked(vn, vs1, vn, vt, vs2, tag="select", stage=st_fetch, batch_round=br)  # quad0

                        # quad1 (nodes 11-14)
                        add_alu_vec_scalar("-", vt, vi, s_seven, tag="select", stage=st_fetch, batch_round=br)
                        add_valu("&", vs0, vt, v_one, tag="select", stage=st_fetch, batch_round=br)  # bit0
                        add_mask_from_bit(vs0, vs0, tag="select", stage=st_fetch, batch_round=br)  # mask0
                        add_select_masked(vs1, v_nodes[12], v_nodes[11], vs0, vs2, tag="select", stage=st_fetch, batch_round=br)  # pair2
                        add_select_masked(vs2, v_nodes[14], v_nodes[13], vs0, vt, tag="select", stage=st_fetch, batch_round=br)   # pair3
                        add_alu_vec_scalar("-", vt, vi, s_seven, tag="select", stage=st_fetch, batch_round=br)
                        add_valu(">>", vt, vt, v_one, tag="select", stage=st_fetch, batch_round=br)
                        add_valu("&", vt, vt, v_one, tag="select", stage=st_fetch, batch_round=br)  # bit1
                        add_mask_from_bit(vt, vt, tag="select", stage=st_fetch, batch_round=br)  # mask1
                        add_select_masked(vs1, vs2, vs1, vt, vs0, tag="select", stage=st_fetch, batch_round=br)  # quad1

                        # mask2 = (off >> 2) & 1  => 0 or 1
                        add_alu_vec_scalar("-", vt, vi, s_seven, tag="select", stage=st_fetch, batch_round=br)
                        add_valu(">>", vt, vt, v_one, tag="select", stage=st_fetch, batch_round=br)
                        add_valu(">>", vt, vt, v_one, tag="select", stage=st_fetch, batch_round=br)
                        add_valu("&", vt, vt, v_one, tag="select", stage=st_fetch, batch_round=br)  # bit2
                        add_mask_from_bit(vt, vt, tag="select", stage=st_fetch, batch_round=br)  # mask2
                        add_select_masked(vn, vs1, vn, vt, vs0, tag="select", stage=st_fetch, batch_round=br)
                        add_valu("^", vv, vv, vn, tag="hash", stage=st_xor, batch_round=br)
                    else:
                        vs1 = v_sel1_shared
                        # off = (idx + 1) & 7, select nodes[7..14] by off bits
                        add_valu("+", vt, vi, v_one, tag="select", stage=st_fetch, batch_round=br)
                        add_valu("&", vs0, vt, v_one, tag="select", stage=st_fetch, batch_round=br)  # mask0 (bit0)
                        add_valu("&", vs1, vt, v_two, tag="select", stage=st_fetch, batch_round=br)  # mask1 (bit1)

                        # quad0
                        add_vselect(vn, vs0, v_nodes[8], v_nodes[7], tag="select", stage=st_fetch, batch_round=br)   # pair0: 7/8
                        add_vselect(vt, vs0, v_nodes[10], v_nodes[9], tag="select", stage=st_fetch, batch_round=br)  # pair1: 9/10
                        add_vselect(vn, vs1, vt, vn, tag="select", stage=st_fetch, batch_round=br)  # quad0

                        # quad1
                        add_vselect(vt, vs0, v_nodes[12], v_nodes[11], tag="select", stage=st_fetch, batch_round=br)  # pair2: 11/12
                        add_vselect(vs0, vs0, v_nodes[14], v_nodes[13], tag="select", stage=st_fetch, batch_round=br)  # pair3: 13/14
                        add_vselect(vt, vs1, vs0, vt, tag="select", stage=st_fetch, batch_round=br)  # quad1

                        # mask2 = ((idx + 1) & 7) >> 2
                        add_valu("+", vs0, vi, v_one, tag="select", stage=st_fetch, batch_round=br)
                        add_valu(">>", vs0, vs0, v_one, tag="select", stage=st_fetch, batch_round=br)
                        add_valu(">>", vs0, vs0, v_one, tag="select", stage=st_fetch, batch_round=br)
                        add_valu("&", vs0, vs0, v_one, tag="select", stage=st_fetch, batch_round=br)
                        add_vselect(vn, vs0, vt, vn, tag="select", stage=st_fetch, batch_round=br)
                        add_valu("^", vv, vv, vn, tag="hash", stage=st_xor, batch_round=br)
                else:
                    # Depths 4+: load from memory using ALU address computation
                    add_alu_vec_scalar("+", vn, vi, s_forest_p, tag="load", stage=st_fetch, batch_round=br)
                    for lane in range(VLEN):
                        add_load_lane(vt + lane, vn + lane, tag="load", stage=st_fetch, batch_round=br)
                    add_valu("^", vv, vv, vt, tag="hash", stage=st_xor, batch_round=br)

            # Hash stages
            add_madd(vv, vv, v_mul0, v_hc0, tag="hash", stage=st_h0, batch_round=br)

            # Stage 1 (xor / shift)
            if use_pipeline:
                add_valu(">>", vt, vv, v_sh1, tag="hash", stage=st_s1, batch_round=br)
                add_valu("^", vv, vv, v_hc1, tag="hash", stage=st_s1, batch_round=br)
                add_valu("^", vv, vv, vt, tag="hash", stage=st_s1, batch_round=br)
            else:
                # Using ALU lanes with scalar constants
                add_alu_vec_scalar(">>", vt, vv, s_19, tag="hash", stage=st_s1, batch_round=br)
                add_alu_vec_scalar("^", vv, vv, s_c1, tag="hash", stage=st_s1, batch_round=br)
                add_alu_vec("^", vv, vv, vt, tag="hash", stage=st_s1, batch_round=br)

            add_madd(vv, vv, v_mul2, v_hc2, tag="hash", stage=st_h2, batch_round=br)

            add_valu("<<", vt, vv, v_sh3, tag="hash", stage=st_h3, batch_round=br)
            add_valu("+", vv, vv, v_hc3, tag="hash", stage=st_h3, batch_round=br)
            add_valu("^", vv, vv, vt, tag="hash", stage=st_h3, batch_round=br)

            add_madd(vv, vv, v_mul4, v_hc4, tag="hash", stage=st_h4, batch_round=br)

            add_valu(">>", vt, vv, v_sh5, tag="hash", stage=st_h5, batch_round=br)
            add_valu("^", vv, vv, v_hc5, tag="hash", stage=st_h5, batch_round=br)
            add_valu("^", vv, vv, vt, tag="hash", stage=st_h5, batch_round=br)

            # Index update: for max depth, always wraps to 0
            if depth == forest_height:
                add_valu("+", vi, v_zero, v_zero, tag="hash", stage=st_idx, batch_round=br)
            elif depth == 0:
                # idx starts at 0 for depth 0, so idx = 1 + (val & 1)
                add_valu("&", vt, vv, v_one, tag="hash", stage=st_idx, batch_round=br)
                add_valu("+", vi, vt, v_one, tag="hash", stage=st_idx, batch_round=br)
            else:
                # idx = (idx << 1) + 1 + (val & 1)
                add_valu("&", vt, vv, v_one, tag="hash", stage=st_idx, batch_round=br)
                add_valu("+", vt, vt, v_one, tag="hash", stage=st_idx, batch_round=br)
                add_madd(vi, vi, v_two, vt, tag="hash", stage=st_idx, batch_round=br)

        tile_base = 0
        for tile in range(num_tiles):
            tile_batches = tile_sizes[tile]
            if num_tiles == 1:
                emit({
                    "alu": [
                        ("+", s_val_ptr, s_val_base, s_zero),
                        ("+", s_val_ptr2, s_val_base, s_vlen),
                    ]
                })
            else:
                s_off = self.scratch_const(tile_base * VLEN)
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
            for b in range(0, tile_batches, 2):
                if b + 1 < tile_batches:
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

            serial_addrs = set()
            if v_sel1_shared is not None:
                serial_addrs.update(vec_addrs(v_sel1_shared))
            if v_sel2_shared is not None:
                serial_addrs.update(vec_addrs(v_sel2_shared))
            if use_static:
                Scheduler = StaticScheduler
            elif use_state_machine:
                Scheduler = StateMachineScheduler
            else:
                Scheduler = ListScheduler
            sched = Scheduler(
                readonly_addrs,
                serial_addrs,
                ("+", s_nop, s_nop, s_zero),
            )

            for r in range(rounds):
                depth = r % (forest_height + 1)
                for b in range(tile_batches):
                    add_round(b, depth, batch_round=r * tile_batches + b)

            self.instrs.extend(sched.schedule())

            if num_tiles == 1:
                emit({
                    "alu": [
                        ("+", s_val_ptr, s_val_base, s_zero),
                        ("+", s_val_ptr2, s_val_base, s_vlen),
                    ]
                })
            else:
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
            for b in range(0, tile_batches, 2):
                if b + 1 < tile_batches:
                    emit({
                        "store": [
                            ("vstore", s_val_ptr, v_val[b]),
                            ("vstore", s_val_ptr2, v_val[b + 1]),
                        ],
                        "alu": [
                            ("+", s_val_ptr, s_val_ptr, s_2vlen),
                            ("+", s_val_ptr2, s_val_ptr2, s_2vlen),
                        ]
                    })
                else:
                    emit({
                        "store": [("vstore", s_val_ptr, v_val[b])],
                        "alu": [("+", s_val_ptr, s_val_ptr, s_vlen)]
                    })

            tile_base += tile_batches

        # Post-process: pack consecutive const-loads (safe init optimization)
        packed_instrs = []
        i = 0
        while i < len(self.instrs):
            instr = self.instrs[i]
            if list(instr.keys()) == ["load"] and len(instr["load"]) == 1:
                slot0 = instr["load"][0]
                if slot0[0] != "const":
                    packed_instrs.append(instr)
                    i += 1
                    continue
                merged_loads = [slot0]
                dests = {slot0[1]}
                j = i + 1
                while j < len(self.instrs) and len(merged_loads) < 2:
                    next_instr = self.instrs[j]
                    if list(next_instr.keys()) == ["load"] and len(next_instr["load"]) == 1:
                        next_slot = next_instr["load"][0]
                        if next_slot[0] == "const" and next_slot[1] not in dests:
                            merged_loads.append(next_slot)
                            dests.add(next_slot[1])
                            j += 1
                        else:
                            break
                    else:
                        break
                packed_instrs.append({"load": merged_loads})
                i = j
            else:
                packed_instrs.append(instr)
                i += 1
        self.instrs = packed_instrs

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
