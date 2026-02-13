# PyTorch Feature Requests for Physics Simulation (mujoco-torch)

These feature requests come from porting MuJoCo's MJX (JAX-based physics
simulation) to PyTorch as [mujoco-torch](https://github.com/vmoens/mujoco-torch).
Physics simulation is a stress test for `torch.compile` and `torch.vmap`
because it combines iterative solvers, heterogeneous data structures, and
batched parallel environments.

---

## 1. `torch.while_loop` — vmap batching rule (HIGH PRIORITY)

**Problem**: `torch.while_loop` (and the underlying `WhileLoopOp` higher-order
op) has no vmap dispatch rule. Calling it inside `torch.vmap` raises:

```
KeyError(<TransformType.Vmap: 1>)
```

**Impact**: Physics solvers (CG, Newton) use while loops with convergence
checks. Under vmap (batched parallel simulation), we must fall back to a
fixed-iteration `for` loop that always runs `max_iter` steps. The MuJoCo
solver has `iterations=100` and `ls_iterations=50` but typically converges in
3–5 iterations. The fixed loop does ~20–30x more work than necessary.

**What JAX does**: `jax.lax.while_loop` has vmap batching rules that convert it
to a masked loop: run until all batch elements converge, using `jnp.where` to
freeze finished elements. This is transparent to user code.

**Desired behavior**:

```python
def cond(state):
    return state.error > tolerance  # per-element bool under vmap

def body(state):
    return solver_step(state)

# Should work under torch.vmap — runs until all elements converge
result = torch.while_loop(cond, body, (initial_state,))
```

**Current workaround** (in mujoco-torch):

```python
if _inside_vmap():
    val = carried_inputs
    for _ in range(max_iter):       # always runs ALL iterations
        val = body_fn(*val)
    return val
```

**References**:
- JAX batching rule: `jax/_src/lax/control_flow/loops.py` → `_while_loop_batching_rule`
- PyTorch while_loop: `torch/_higher_order_ops/while_loop.py`

---

## 2. `torch.while_loop` — dynamic shapes support

**Problem**: Using `torch.compile(fn, dynamic=True)` with `torch.while_loop`
triggers an inductor lowering error:

```
torch._inductor.exc.InductorError: LoweringException: KeyError: 'unbacked_bindings'
```

**Impact**: Dynamic shapes would allow a single compiled graph to handle
different model configurations without recompilation. Currently we must use
`dynamic=False` (the default) which means a separate compilation per model.

**Desired behavior**: `torch.compile(dynamic=True)` should work with
`while_loop` (and other higher-order ops like `scan`, `cond`).

---

## 3. `torch._higher_order_ops.scan` — heterogeneous iteration shapes

**Problem**: The `scan` higher-order op requires all iterations to process
tensors of identical shapes. In physics simulation, we scan over joint groups
where each group has different tensor shapes (e.g., 3-DOF ball joints vs 1-DOF
hinge joints). This means `scan` cannot be used, and the entire function must
be excluded from compilation.

**Context**: MuJoCo models organize joints, actuators, and bodies into groups
by type. A scan over these groups calls the same function `f` but with
different-sized input slices:

```python
# Pseudocode for what scan.flat does:
for group in joint_groups:
    # group.qpos might be (N_ball, 4) or (N_hinge, 1) — different shapes!
    result = vmap(f)(group.qpos, group.qvel, ...)
    results.append(result)
```

**What JAX does**: `jax.lax.scan` has the same homogeneous-shape restriction,
but JAX's XLA compiler can trace through Python for-loops efficiently via
`jax.jit`, so the for-loop fallback compiles into a single fused graph.

**Desired behavior** (one of):

a. A `scan` variant that supports heterogeneous shapes across iterations
   (padding + masking internally), or

b. `torch.compile` should be able to trace a Python for-loop where each
   iteration calls `vmap(f)` with different shapes, producing a single fused
   graph (like XLA does for JAX). Today, dynamo recompiles per iteration due
   to shape changes.

**Current workaround**: `@torch.compiler.disable` on the entire scan function,
causing a graph break.

---

## 4. `torch.vmap` — non-tensor pytree leaves in output

**Problem**: When a vmapped function returns a pytree that contains non-tensor
values (e.g., Python ints, strings, or custom non-tensor wrappers), vmap errors:

```
ValueError: vmap must only return Tensors, got type <class 'int'>
```

**Impact**: Physics simulation data structures (like tensordict's
`TensorClass`) may store metadata as non-tensor fields (constraint counts,
flags, etc.). These are identical across batch elements and should pass through
vmap unchanged.

**What JAX does**: `jax.vmap` treats non-array pytree leaves as auxiliary data
in the tree spec. They pass through the transform unchanged and appear
identically in every output element.

**Desired behavior**:

```python
@dataclass
class SimState:
    qpos: torch.Tensor    # batched
    ncon: int              # metadata, same for all batch elements

def step(state):
    return SimState(qpos=state.qpos + 1, ncon=state.ncon)

# Non-tensor fields should pass through unchanged
result = torch.vmap(step)(batched_state)
assert isinstance(result.ncon, int)  # not batched
```

**Current workaround**: Convert all non-tensor fields to 0-d tensors so vmap
can batch/unbatch them. This is fragile and adds unnecessary overhead.

---

## 5. `torch.compile` — `__getattribute__` causes whole-frame graph breaks

**Problem**: If a class defines `__getattribute__`, dynamo inserts a graph break
on *every* attribute access on instances of that class — even for attributes
that don't hit the custom logic. If the function body has enough such accesses,
dynamo gives up and skips the entire frame:

```
torch.compile will skip tracing the frame _advance and fall back to eager.
The graph break occurred in: _mj_getattribute
  return object.__getattribute__(self, name)
```

**Impact**: In mujoco-torch, a `__getattribute__` override was used to resolve
a single field name collision (`Model.names` vs `TensorDict.names`). This
caused *every* attribute access on `Model` to trigger a graph break, making
`torch.compile` skip core integration functions entirely.

**Desired behavior**: Dynamo should be able to inline/trace simple
`__getattribute__` implementations, or at least only break on the specific
attributes that hit custom logic rather than breaking on all accesses.

**Current workaround**: Replace `__getattribute__` with a targeted `@property`
descriptor for the specific colliding field. This works but isn't always
possible (e.g., when the set of affected attributes isn't known at class
definition time).

---

## Summary and prioritization

| # | Feature | Impact | Workaround cost | Status |
|---|---------|--------|-----------------|--------|
| 1 | `while_loop` vmap rule | **High** — 20-30x wasted solver compute | Fixed-iter for-loop | **DONE** — masked loop batching rule added |
| 2 | `while_loop` dynamic shapes | Medium — recompile per model | Use `dynamic=False` | **DONE** — already works, regression tests added |
| 3 | `scan` heterogeneous shapes | **High** — graph break on all scan ops | `@torch.compiler.disable` | OPEN — requires padding/masking scan variant or dynamo dynamic-shape for-loop support |
| 4 | `vmap` non-tensor leaves | Medium — forces int→tensor conversion | 0-d tensor workaround | **DONE** — non-tensor leaves pass through unchanged |
| 5 | `__getattribute__` tracing | Medium — forces API redesign | `@property` per field | **DONE** — dynamo now traces `__getattribute__` on nn.Module |
