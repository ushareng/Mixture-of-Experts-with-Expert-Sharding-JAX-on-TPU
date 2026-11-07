# Mixture-of-Experts with Expert Sharding — JAX on TPU

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ushareng/Mixture-of-Experts-with-Expert-Sharding-JAX-on-TPU/blob/main/expert_sharding_moe_AX_TPU%20(2).ipynb)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

A self-contained Colab notebook that builds a **sparse Mixture-of-Experts (MoE) layer** from scratch in pure JAX and shards its experts across TPU cores (**expert parallelism**). It implements the layer two ways, GSPMD sharding annotations and `shard_map` with explicit `all_to_all`, and checks that both match a dense reference before training a small MoE network.

No Flax, no framework: just `jax`, `optax` and `matplotlib`.

---

## Highlights

- **2-D device mesh** `('data', 'expert')`. Tokens are split across every device, and the expert weights are split along the expert dimension.
- **Top-k router** (Switch / GShard style):
  - fixed expert capacity, with overflow tokens dropped
  - priority filling: every token's 1st choice gets a slot before anyone's 2nd choice
  - load-balancing auxiliary loss and router z-loss
  - router computed in float32
- **Two expert-parallel implementations:**
  - **A. GSPMD**: write the math with `einsum`s, add `with_sharding_constraint`, and let XLA insert the `all-to-all`s.
  - **B. `shard_map`**: write the per-device program by hand with explicit `jax.lax.all_to_all` for dispatch and combine.
- **Correctness checks:** both implementations are compared against a dense "run every expert on every token" reference, on forward outputs, aux losses **and gradients**.
- **Compiled-code inspection:** counts the collectives in the compiled HLO to confirm the expected communication pattern.
- **Training demo:** a residual MoE network on a synthetic task where expert specialization is visible, with plots of loss, load balance, dropped tokens and a routing heatmap.
- **Benchmark:** GSPMD vs `shard_map` training-step time, tokens/s and expert-matmul TFLOP/s.

## How the expert sharding works

| Tensor | Shape | Sharding |
|---|---|---|
| Activations / tokens | `[G, S, M]` | `G` split over `('data', 'expert')` |
| Expert input weights `w_in` | `[E, M, H]` | `E` split over `'expert'` |
| Expert output weights `w_out` | `[E, H, M]` | `E` split over `'expert'` |
| Router, embedding, head | — | replicated |

One MoE layer:

```
tokens [G, S, M]            (sharded by group)
   │  route + one-hot dispatch
   ▼
expert buffers [E, G, C, M] ── all-to-all #1 ──▶ each device holds only its own experts' tokens
   │  local expert FFN:  gelu(x · w_in) · w_out
   ▼
expert outputs [E, G, C, M] ── all-to-all #2 ──▶ back to group sharding
   │  weighted combine
   ▼
tokens [G, S, M]
```

Each expert's buffer holds `C = ceil(top_k · S · capacity_factor / E)` tokens per group, which keeps every shape static for XLA/TPU.

## Quick start

1. Click **Open in Colab** above.
2. **Runtime → Change runtime type → TPU**.
3. **Runtime → Run all**.

The notebook picks its model size from the backend: bfloat16 and larger layers on TPU, a small float32 model on CPU.

### Choosing a runtime

| Runtime | Devices | What you get |
|---|---|---|
| TPU **v2-8** | 8 | Real 8-way expert sharding (1 expert per core) |
| TPU **v5e-1 / v6e-1** | 1 | Everything runs, but there is nothing to shard across (all experts on one chip) |
| Any runtime with `FORCE_CPU_SIMULATION = True` | 8 virtual CPU devices | Real sharding and real `all-to-all`s, slow |

To see the sharding on any runtime, set this in the **first cell**, before JAX is imported:

```python
FORCE_CPU_SIMULATION = True
```

## Results

### TPU v5e-1 (1 chip, JAX 0.11.2)

Correctness:

```
no-drop  | max|A-ref| = 1.19e-07 | max|B-ref| = 1.19e-07 | drop frac = 0.0
capacity | loss A = 0.214370  loss B = 0.214370 | drop frac = 0.031
✅ All correctness checks passed.
```

Training (2 MoE layers, d_model 512, d_ff 2048, 8 experts, top-2, 33.6M params, 2,048 tokens/step):

| step | MSE | load-balance loss | dropped |
|---:|---:|---:|---:|
| 0 | 1.786 | 1.052 | 7.7% |
| 100 | 0.026 | 1.028 | 0.5% |
| 300 | 0.0126 | 1.013 | 0.1% |
| 599 | 0.0073 | 1.012 | 0.2% |

Benchmark (forward + backward + AdamW, bfloat16):

| Implementation | ms / step | tokens / s | expert-matmul throughput |
|---|---:|---:|---:|
| GSPMD | 2.49 | 0.82 M | 51.7 TFLOP/s |
| `shard_map` | 2.51 | 0.82 M | 51.4 TFLOP/s |

### 8 simulated devices (`FORCE_CPU_SIMULATION = True`)

Each device holds exactly one expert, and the compiled forward pass contains the two expected all-to-alls:

```
Mesh('data': 1, 'expert': 8)
Collectives in compiled FORWARD:
  GSPMD     : {'all-to-all': 2, 'all-gather': 1, ...}
  shard_map : {'all-to-all': 2, 'all-gather': 0, ...}
```

Gradients from the two implementations agree to within 1e-9.

## Notebook outline

1. **Device mesh**: build the `('data', 'expert')` mesh.
2. **Configuration and parameters**: `MoEConfig`, expert capacity, parameter `PartitionSpec`s.
3. **Router**: top-k gating, capacity, dispatch/combine tensors, aux losses.
4. **Expert parallelism**: (A) GSPMD, (B) `shard_map`.
5. **Correctness checks**: dense reference, A vs B forward and gradients.
6. **Sharding and HLO inspection**: which device holds which expert, and a count of collectives.
7. **Training**: MoE network on 8-cluster synthetic regression.
8. **Plots**: loss, load-balance loss, dropped tokens, expert-specialization heatmap, per-expert load.
9. **Benchmark**: GSPMD vs `shard_map`.

## Key configuration

```python
MoEConfig(
    d_model=512,
    d_ff=2048,
    num_experts=8,
    top_k=2,
    capacity_factor=1.25,     # 1.0–1.25 for training, >=2 for eval
    aux_loss_coef=1e-2,       # load-balancing loss weight
    z_loss_coef=1e-3,         # router z-loss weight
    compute_dtype=jnp.bfloat16,
)
```

## Notes

- **Precision in tests:** on TPU, float32 matmuls default to a single bfloat16 pass, which is only accurate to about 1e-2. The correctness cell sets `jax_default_matmul_precision="highest"` for the comparison and resets it afterwards.
- **GSPMD vs `shard_map`:** GSPMD is shorter and composes with other shardings. `shard_map` gives exact control over communication, for example to overlap the all-to-all with compute.
- **Many experts:** one-hot `einsum` dispatch costs `O(S·E·C·M)`. For 64+ experts, switch to sort/gather dispatch, or to `jax.lax.ragged_dot` with `ragged_all_to_all` for dropless MoE.
- **TPU pods:** the same code works on multi-host slices. Call `jax.distributed.initialize()`, build global batches with `jax.make_array_from_process_local_data`, and place the `expert` axis on the fastest interconnect dimension.
- **Tensor parallelism inside experts:** add a `model` mesh axis and shard `w_in` as `P('expert', None, 'model')` and `w_out` as `P('expert', 'model', None)`.

## Requirements

- JAX ≥ 0.4.30 (tested on 0.11.2 on TPU and 0.12 on CPU). Preinstalled on Colab TPU runtimes.
- `optax` (installed by the notebook if missing)
- `matplotlib`, `numpy`

## References

- Lepikhin et al., *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding*, 2020.
- Fedus, Zoph & Shazeer, *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*, 2021.
- Zoph et al., *ST-MoE: Designing Stable and Transferable Sparse Expert Models*, 2022.
- JAX docs: [Distributed arrays and automatic parallelization](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html), [`shard_map`](https://docs.jax.dev/en/latest/notebooks/shard_map.html).

## License

[Apache 2.0](LICENSE)
