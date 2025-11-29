# ANNOTATED EXAMPLE 2: Mixtral 8x7B FSDP+EP Training

**Source:** `build_and_train_models/sm-distributed_model_parallel_v2/mixtral/sm-fsdp-ep_train_mixtral.ipynb`

**Purpose:** This annotation explains Expert Parallelism (EP) for Mixture-of-Experts (MoE) models like Mixtral, showing how it differs from Tensor Parallelism and why it's essential for training MoE architectures efficiently.

---

## Overview: What Makes Mixtral Different?

**Mixtral 8x7B Architecture:**
- **Total Parameters:** 46.7B parameters
- **Active Parameters per Token:** ~12.9B (only 2 of 8 experts activated per token)
- **Architecture:** 8 expert networks per layer, router network selects top-2 experts per token
- **Challenge:** Experts must be distributed across GPUs to fit in memory

**Why Expert Parallelism?**
- **Tensor Parallelism (TP):** Splits individual tensors across GPUs → Not optimal for MoE (experts are independent)
- **Expert Parallelism (EP):** Assigns different experts to different GPUs → Natural fit for MoE architecture
- **Hybrid:** Combine FSDP (for non-expert layers) + EP (for expert layers)

---

## SECTION 1: Mixtral Model Architecture Primer

### Mixture-of-Experts (MoE) Basics

**Standard Transformer Layer:**
```
Input → Attention → FFN (single network) → Output
```

**MoE Transformer Layer (Mixtral):**
```
Input → Attention → Router + 8 Expert FFNs → Output
                       ↓
            Selects top-2 experts per token
            Computes weighted combination
```

**Key Components:**
1. **Router Network:** Learns which experts to activate for each token
2. **Expert Networks:** 8 independent FFN networks (each ~7B parameters)
3. **Load Balancing:** Ensures all experts get roughly equal usage

**Memory Challenge:**
- 8 experts × 7B params = 56B params just for experts (per layer!)
- 32 layers × 56B = Cannot fit on single GPU even with BF16

**Solution: Expert Parallelism**
- Distribute experts across GPUs
- Use All-to-All communication to route tokens to correct expert GPUs
- Combine results with another All-to-All

---

## SECTION 2: Environment Setup (Same as Llama Example)

```python
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.pytorch import PyTorch

role = get_execution_role()
sagemaker_session = sagemaker.session.Session()
default_bucket = sagemaker_session.default_bucket()
```

**ANNOTATION:**
- Setup identical to Llama example
- See Example 1 for detailed annotation of these SDK calls

---

## SECTION 3: Data Preparation (GLUE/SST2)

```python
# Load dataset
raw_datasets = load_dataset("glue", "sst2")

# Load Mixtral tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

# Tokenize with block_size=2048 (Mixtral's max context)
block_size = 2048
lm_datasets = tokenized_datasets.map(
    functools.partial(group_texts, block_size),
    batched=True,
)

# Upload to S3
train_dataset.to_json("./training.json")
training_dataset_location = f"s3://{default_bucket}/dataset/train/"
os.system(f"aws s3 cp ./training.json {training_dataset_location}")
```

**ANNOTATION - Mixtral-Specific Considerations:**

**Tokenizer Differences:**
- **Mixtral vocab:** 32,000 tokens (vs Llama 3.1: 128,256)
- **Context length:** 2048 default (vs Llama 3.1: 4096+)
- **Tokenizer type:** SentencePiece (similar to Llama 2)

**Why smaller block_size (2048) matters for MoE:**
- Longer sequences = more tokens = more expert routing overhead
- All-to-All communication scales with sequence length
- Shorter sequences reduce EP communication bottleneck

---

## SECTION 4: Distributed Training Configuration

### Key Hyperparameters for Expert Parallelism

```python
expert_parallel_degree = 2   # NEW: Expert parallelism degree
hybrid_shard_degree = 32     # FSDP sharding across 32 GPUs
activation_loading_horizon = 2
offload_activations = False  # Usually False for MoE (already memory efficient)

hyperparameters = {
    # MoE-specific settings
    "moe": 1,                           # Enable MoE mode
    "moe_load_balancing": "sinkhorn",   # Load balancing algorithm
    "moe_all_to_all_dispatcher": 1,     # Use All-to-All for expert routing

    # Model architecture
    "model_type": "mixtral",
    "hidden_width": 4096,
    "num_layers": 32,
    "num_heads": 32,
    "num_key_value_heads": 8,           # GQA: Grouped Query Attention
    "num_local_experts": 8,             # 8 experts per MoE layer

    # Training config
    "train_batch_size": 2,
    "max_steps": 15,
    "bf16": 1,

    # FSDP settings
    "sharding_strategy": "hybrid_shard",
    "auto_wrap_policy": "transformer_auto_wrap_policy",
    "activation_checkpointing": 1,
}
```

### DEEP ANNOTATION: Expert Parallelism Parameters

#### **expert_parallel_degree**
```python
expert_parallel_degree = 2
```

**What it does:**
- Distributes the 8 experts across `expert_parallel_degree` GPUs
- Each GPU holds `8 / 2 = 4 experts`

**Example with EP=2, 64 GPUs (8 nodes × 8 GPUs):**
```
GPU Group 0-1 (EP group 0):
  - GPU 0: Experts 0, 1, 2, 3
  - GPU 1: Experts 4, 5, 6, 7

GPU Group 2-3 (EP group 1):
  - GPU 2: Experts 0, 1, 2, 3
  - GPU 3: Experts 4, 5, 6, 7

... (32 EP groups total)
```

**Communication pattern:**
```
Forward pass for a batch of tokens:
1. Router decides which experts to use for each token
2. All-to-All: Tokens sent to GPUs with their assigned experts
3. Expert computation: Each GPU processes its tokens
4. All-to-All: Results sent back to original GPUs
```

**Why EP=2 vs higher:**
- **EP=2:** Each GPU stores 4 experts
  - Memory per GPU: 4 × 7B = ~28B params (for experts)
  - Communication: All-to-All across 2 GPUs (fast)
- **EP=8:** Each GPU stores 1 expert
  - Memory per GPU: 1 × 7B = ~7B params
  - Communication: All-to-All across 8 GPUs (slower)
  - Use when: Memory constrained OR very large expert models

**Recommended values:**
- **EP=1:** No expert parallelism (all experts on each GPU) - only for small MoE
- **EP=2:** Good balance of memory savings and communication
- **EP=4:** More memory savings, acceptable communication overhead
- **EP=8:** Maximum memory savings, use for very large experts
- **EP > 8:** Not recommended (inter-node All-to-All is very slow)

#### **hybrid_shard_degree = 32**
```python
hybrid_shard_degree = 32  # FSDP shards across 32 GPUs
```

**What it does:**
- FSDP shards non-expert parameters across 32 GPUs
- Non-expert params: Attention, LayerNorm, embeddings, router

**Math with 64 total GPUs, EP=2, HSD=32:**
```
Total GPUs: 64
EP groups: 64 / 2 = 32 EP groups
Each EP group has: 2 GPUs sharing expert distribution

FSDP sharding happens within all 64 GPUs (HSD=32 applies to effective world)
Actually: Since EP=2, FSDP sees world_size = 64, HSD=32 means:
  - Shard across 32 "effective" units
  - Each unit consists of inter-EP replication
```

**Why HSD=32 for 64 GPUs:**
- Total GPUs: 8 instances × 8 GPUs = 64
- HSD=32: Shard non-expert params across 32 groups
- Each group: 2 GPUs with replicated non-expert params
- Balances: Memory savings vs communication cost

**Memory distribution per GPU (approximate):**
```
Non-expert params (attention, etc.): ~15B / 32 = ~470M per GPU
Expert params: 8 experts / 2 = 4 experts × 7B = ~28B per GPU
Total per GPU: ~28.5B parameters (fits in 40GB A100 with BF16 + activations)
```

#### **moe_load_balancing: "sinkhorn"**
```python
"moe_load_balancing": "sinkhorn"
```

**What it does:**
- Ensures tokens are distributed evenly across experts
- Prevents: Some experts overloaded, others idle

**Why load balancing matters:**
```
Without load balancing:
  Expert 0: 80% of tokens → GPU 0 bottleneck
  Expert 1: 5% of tokens → GPU 1 idle
  ... unbalanced computation

With Sinkhorn balancing:
  Expert 0: ~12.5% of tokens (equal distribution)
  Expert 1: ~12.5% of tokens
  ... balanced across all 8 experts
```

**Load Balancing Algorithms:**

1. **"sinkhorn"** (used here):
   - Iterative algorithm to balance token-to-expert assignments
   - Ensures: Each expert gets roughly equal number of tokens
   - Cost: Slight routing overhead
   - Best for: Training (ensures all experts learn equally)

2. **"aux_loss"** (alternative):
   - Adds auxiliary loss term to encourage balanced routing
   - Router learns to balance over time
   - Best for: Inference (no extra routing compute)

3. **"none"**:
   - No load balancing
   - Can lead to expert collapse (some experts never used)

**Impact on training:**
- Prevents expert collapse (all experts learn)
- Improves convergence (no wasted capacity)
- Slightly slower than no balancing (~5% overhead)

#### **moe_all_to_all_dispatcher: 1**
```python
"moe_all_to_all_dispatcher": 1
```

**What it does:**
- Enables NCCL All-to-All communication for expert routing
- Alternative: Sequential send/recv (slower)

**All-to-All Communication Pattern:**
```
Before All-to-All (tokens on GPU 0):
  Token 0 → needs Expert 5 (on GPU 1)
  Token 1 → needs Expert 2 (on GPU 0)
  Token 2 → needs Expert 7 (on GPU 1)

After All-to-All:
  GPU 0 receives: Tokens assigned to Experts 0-3
  GPU 1 receives: Tokens assigned to Experts 4-7

Each GPU processes its tokens through its experts

After 2nd All-to-All: Results returned to original GPUs
```

**Performance:**
- All-to-All: Single collective communication (fast)
- Sequential: Multiple point-to-point sends (slow)
- Speedup: ~2-5x for expert routing

**Requirement:**
- NCCL >= 2.12 for efficient All-to-All
- InfiniBand or EFA for multi-node (P4d has EFA)

---

## SECTION 5: Instance Configuration for Multi-Node Training

```python
instance_type = "ml.p4d.24xlarge"  # 8x A100 40GB GPUs
instance_count = 8                  # 8 nodes = 64 GPUs total
processes_per_host = 8              # 8 processes (1 per GPU)
```

**ANNOTATION - Why 64 GPUs for Mixtral 8x7B:**

**Memory Requirements:**
```
Mixtral 8x7B parameters: ~46.7B
  - BF16: 46.7B × 2 bytes = 93.4 GB (model weights alone)
  - Optimizer (AdamW): 2× weights for momentum + variance = 186.8 GB
  - Gradients: Same as weights = 93.4 GB
  - Activations (batch_size=2, seq=2048): ~50 GB per GPU

Total without optimization: ~280 GB minimum
```

**With FSDP+EP optimization (HSD=32, EP=2):**
```
Per-GPU memory:
  - Model weights: 46.7B / 32 ≈ 1.5B × 2 bytes = 3 GB
  - Optimizer states: 3 GB × 2 = 6 GB
  - Gradients: 3 GB
  - Activations: 50 GB (not sharded by FSDP)
  - Overhead: 5 GB

Total per GPU: ~67 GB... still too much for 40GB GPU!
```

**But wait - Expert Parallelism saves the day:**
```
With EP=2:
  - Non-expert params sharded (FSDP): ~15B / 32 ≈ 470M per GPU
  - Expert params (local): 4 experts × 7B = 28B per GPU (not sharded)

Per-GPU with EP:
  - Non-expert (FSDP): 470M × 2 bytes × 4 (model + opt + grad) ≈ 4 GB
  - Experts (local): 28B × 2 bytes × 4 ≈ 224 GB... still too big!

Actually: Experts also get FSDP sharding:
  - 28B / (64/2) = 28B / 32 ≈ 875M per GPU for expert params
  - Total params per GPU: 470M + 875M ≈ 1.35B
  - With optimizer & gradients: 1.35B × 4 × 2 bytes ≈ 10.8 GB
  - Plus activations: ~30 GB (reduced with checkpointing)
  - Total: ~40 GB → Fits in A100 40GB!
```

**Scaling Rules:**
- **16 GPUs (2 nodes):** Possible with activation offloading + smaller batch
- **32 GPUs (4 nodes):** Comfortable fit
- **64 GPUs (8 nodes):** Used here for faster training
- **128+ GPUs:** For Mixtral 8x22B or larger batch sizes

**Instance Type Comparison:**

| Instance | GPUs | GPU Mem | Use Case |
|----------|------|---------|----------|
| ml.p4d.24xlarge | 8 | 40 GB | Mixtral 8x7B (needs ≥4 nodes) |
| ml.p4de.24xlarge | 8 | 80 GB | Mixtral 8x7B (can use 2 nodes) |
| ml.p5.48xlarge | 8 | 80 GB | Mixtral 8x7B (2 nodes) or Mixtral 8x22B |

---

## SECTION 6: PyTorch Estimator with Expert Parallelism

```python
smp_estimator = PyTorch(
    entry_point="train.py",
    source_dir="../shared-scripts",
    hyperparameters=hyperparameters,
    role=role,

    # Multi-node configuration
    instance_type="ml.p4d.24xlarge",
    instance_count=8,              # 8 nodes
    volume_size=400,

    # Checkpointing
    checkpoint_s3_uri=checkpoint_s3_uri,

    # DISTRIBUTED CONFIGURATION - EP + FSDP
    distribution={
        "torch_distributed": {"enabled": True},
        "smdistributed": {
            "modelparallel": {
                "enabled": True,
                "parameters": {
                    "expert_parallel_degree": 2,      # NEW: Expert Parallelism
                    "hybrid_shard_degree": 32,        # FSDP sharding
                    "sm_activation_offloading": False, # Usually off for MoE
                    "activation_loading_horizon": 2,
                },
            }
        },
    },

    py_version="py311",
    framework_version="2.4.1",
    output_path=s3_output_bucket,
    sagemaker_session=sagemaker_session,
    base_job_name=base_job_name,
)
```

### DEEP ANNOTATION: EP vs TP Configuration

**Comparison: Expert Parallelism vs Tensor Parallelism**

| Feature | Expert Parallelism (EP) | Tensor Parallelism (TP) |
|---------|------------------------|-------------------------|
| **Use Case** | MoE models (Mixtral, Switch) | Dense models (Llama, GPT) |
| **What's Split** | Expert networks across GPUs | Individual tensor dimensions |
| **Communication** | All-to-All (token routing) | All-Reduce (tensor ops) |
| **Granularity** | Expert-level (7B chunks) | Layer-level (matrix slices) |
| **Best Degree** | 2-4 (match # experts / GPU memory) | 2-8 (intra-node only) |
| **Memory Pattern** | Uneven (experts + non-experts) | Even (all params split equally) |
| **Compute Pattern** | Sparse (only active experts) | Dense (all params used) |

**When to use EP:**
```python
# MoE model (Mixtral, Switch Transformer)
distribution = {
    "smdistributed": {
        "modelparallel": {
            "parameters": {
                "expert_parallel_degree": 2,  # Use EP
                "hybrid_shard_degree": 32,
            }
        }
    }
}
```

**When to use TP:**
```python
# Dense model (Llama, GPT-NeoX)
distribution = {
    "smdistributed": {
        "modelparallel": {
            "parameters": {
                "tensor_parallel_degree": 2,  # Use TP
                "hybrid_shard_degree": 4,
            }
        }
    }
}
```

**Can you use both?**
- **Theoretically:** Yes, EP for expert layers + TP for attention
- **In practice:** Rarely used (too complex, marginal benefits)
- **SageMaker MP v2:** Supports EP or TP, not both simultaneously

---

## SECTION 7: Container Behavior with Multi-Node EP

### Multi-Node Startup Sequence

**When you call `estimator.fit()`:**

1. **SageMaker provisions 8 instances**
   ```
   algo-1: ml.p4d.24xlarge (8 GPUs) → Rank 0-7
   algo-2: ml.p4d.24xlarge (8 GPUs) → Rank 8-15
   algo-3: ml.p4d.24xlarge (8 GPUs) → Rank 16-23
   ...
   algo-8: ml.p4d.24xlarge (8 GPUs) → Rank 56-63
   ```

2. **Each instance runs torchrun:**
   ```bash
   # On algo-1 (master node):
   torchrun \
     --nproc_per_node=8 \
     --nnodes=8 \
     --node_rank=0 \
     --master_addr=algo-1 \
     --master_port=7777 \
     train.py --expert_parallel_degree=2 ...

   # On algo-2:
   torchrun \
     --nproc_per_node=8 \
     --nnodes=8 \
     --node_rank=1 \
     --master_addr=algo-1 \
     --master_port=7777 \
     train.py --expert_parallel_degree=2 ...

   # ... same for algo-3 through algo-8
   ```

3. **Environment variables set per process:**
   ```bash
   # Process on algo-1, GPU 0:
   WORLD_SIZE=64
   RANK=0
   LOCAL_RANK=0
   MASTER_ADDR=algo-1
   MASTER_PORT=7777

   # Process on algo-2, GPU 3:
   WORLD_SIZE=64
   RANK=11
   LOCAL_RANK=3
   MASTER_ADDR=algo-1
   MASTER_PORT=7777
   ```

### EP Group Formation

**Inside train.py, SageMaker MP creates EP groups:**

```python
import sagemaker_model_parallel.torch as smp

smp.init()  # Reads distribution config

# With EP=2, creates 32 EP groups:
# EP Group 0: Ranks [0, 1]     (algo-1 GPU 0-1)
# EP Group 1: Ranks [2, 3]     (algo-1 GPU 2-3)
# EP Group 2: Ranks [4, 5]     (algo-1 GPU 4-5)
# ...
# EP Group 31: Ranks [62, 63]  (algo-8 GPU 6-7)

# Each EP group handles same expert distribution:
# Rank 0: Experts 0-3
# Rank 1: Experts 4-7
# Rank 2: Experts 0-3
# Rank 3: Experts 4-7
# ... (pattern repeats)
```

**FSDP groups (HSD=32) span across EP groups:**
```python
# FSDP shard group: Every 2 ranks (matching EP degree)
# FSDP Shard 0: Rank 0, 2, 4, ..., 62 (32 ranks)
# FSDP Shard 1: Rank 1, 3, 5, ..., 63 (32 ranks)

# This means:
# - Non-expert params sharded across all 64 GPUs (grouped by 32)
# - Expert params local to EP groups (4 experts per GPU)
```

### Expert Routing During Forward Pass

**Step-by-step for a single MoE layer:**

```
1. Input tokens on each GPU (batch_size=2, seq_len=2048)
   Rank 0: 2×2048 = 4096 tokens
   Rank 1: 4096 tokens
   ... (each rank has its tokens)

2. Router Network (runs on each GPU)
   - Computes scores for all 8 experts per token
   - Selects top-2 experts per token
   - Load balancing ensures even distribution

3. All-to-All: Route tokens to expert GPUs
   Example for Rank 0's tokens:
     Tokens [0-1023] → assigned to Experts 0,1 → stay on Rank 0
     Tokens [1024-2047] → assigned to Experts 4,5 → send to Rank 1
     Tokens [2048-3071] → assigned to Experts 2,3 → stay on Rank 0
     Tokens [3072-4095] → assigned to Experts 6,7 → send to Rank 1

   After All-to-All:
     Rank 0 receives: All tokens assigned to Experts 0-3 (from all ranks)
     Rank 1 receives: All tokens assigned to Experts 4-7 (from all ranks)

4. Expert Computation (parallel on each GPU)
   Rank 0: Process tokens through Experts 0, 1, 2, 3
   Rank 1: Process tokens through Experts 4, 5, 6, 7

5. All-to-All: Return results to original GPUs
   Rank 0 sends results back to original token owners
   Rank 1 sends results back to original token owners

6. Combine expert outputs (weighted by router scores)
```

**Communication cost:**
- 2× All-to-All per MoE layer (forward)
- 2× All-to-All per MoE layer (backward)
- 32 MoE layers × 4 All-to-Alls = 128 All-to-Alls per forward+backward
- Critical: Use fast interconnect (EFA on P4d: 400 Gbps)

---

## SECTION 8: Training Script Integration

**How train.py uses EP (simplified):**

```python
import sagemaker_model_parallel.torch as smp
import torch
from transformers import MixtralForCausalLM

# 1. Initialize SMP
smp.init()

# 2. Load Mixtral model
model = MixtralForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-v0.1",
    torch_dtype=torch.bfloat16,
)

# 3. Wrap with DistributedModel (automatically handles EP)
model = smp.DistributedModel(
    model,
    expert_parallel_degree=2,      # Experts distributed across 2 GPUs
    hybrid_shard_degree=32,        # FSDP sharding
    trace_device="cuda",
)

# 4. Model automatically:
#    - Detects MoE layers (MixtralSparseMoeBlock)
#    - Applies EP to expert layers
#    - Applies FSDP to non-expert layers
#    - Sets up All-to-All communication groups

# 5. Training loop (same as dense models!)
@smp.step
def train_step(batch):
    outputs = model(**batch)
    loss = outputs.loss
    model.backward(loss)
    return loss

for batch in dataloader:
    loss = train_step(batch)
    optimizer.step()

    # EP routing happens automatically inside forward/backward
```

**What SMP does automatically for MoE:**
1. Detects `nn.ModuleList` of expert FFNs
2. Distributes experts across EP groups
3. Inserts All-to-All communication before/after expert computation
4. Handles load balancing (if enabled)
5. Manages gradient synchronization across EP groups

---

## SECTION 9: Performance Considerations

### Communication Overhead: EP vs TP

**Tensor Parallelism (dense models):**
```
Per layer:
  - Column-parallel: 1 All-Reduce
  - Row-parallel: 1 All-Reduce
Total: 2 All-Reduces × 32 layers = 64 All-Reduces
Data size: ~4 GB (activation size)
```

**Expert Parallelism (MoE models):**
```
Per MoE layer:
  - Forward All-to-All (token routing): Send all tokens
  - Forward All-to-All (gather results): Receive all results
  - Backward All-to-All (gradient routing): 2× more
Total: 4 All-to-Alls × 32 layers = 128 All-to-Alls
Data size: batch_size × seq_len × hidden_dim × num_active_experts
  = 2 × 2048 × 4096 × 2 × 2 bytes = 134 MB per All-to-All
```

**Why MoE can still be faster:**
- Only 2 of 8 experts active → 75% less compute per token
- Experts computed in parallel → high GPU utilization
- Larger batch sizes possible (fewer active params)

### Instance Type Selection for MoE

**Network bandwidth critical for EP:**

| Instance | Network | Intra-Node | Inter-Node | Best For |
|----------|---------|------------|------------|----------|
| ml.p4d.24xlarge | 4× 100 Gbps EFA | NVLink (600 GB/s) | EFA (50 GB/s per link) | Mixtral 8x7B |
| ml.p4de.24xlarge | 4× 100 Gbps EFA | NVLink (600 GB/s) | EFA (50 GB/s per link) | Mixtral 8x7B (80GB GPU) |
| ml.p5.48xlarge | 8× 100 Gbps EFA | NVSwitch (900 GB/s) | EFA (100 GB/s per link) | Mixtral 8x22B |

**Why P5 is better for large MoE:**
- 2× network bandwidth (800 Gbps vs 400 Gbps)
- More GPU memory (80 GB vs 40 GB)
- Faster NVSwitch for intra-node All-to-All

### Tuning EP Degree

**Rule of thumb:**
```
expert_parallel_degree = num_local_experts / experts_per_GPU

Target experts_per_GPU based on expert size:
  - 7B expert: 2-4 experts per GPU
  - 3B expert: 4-8 experts per GPU
  - 1B expert: 8+ experts per GPU

Mixtral 8x7B: 8 experts / 4 per GPU = EP=2 ✓
```

**Trade-offs:**

| EP Degree | Memory per GPU | Communication | Use When |
|-----------|----------------|---------------|----------|
| EP=1 | All 8 experts | No All-to-All | Small experts, 80GB GPUs |
| EP=2 | 4 experts | 2-GPU All-to-All | **Recommended for Mixtral 8x7B** |
| EP=4 | 2 experts | 4-GPU All-to-All | Memory constrained, 40GB GPUs |
| EP=8 | 1 expert | 8-GPU All-to-All | Very large experts, last resort |

---

## SECTION 10: Resuming from Checkpoint

```python
# Enable checkpoint resume
hyperparameters["resume_from_checkpoint"] = "/opt/ml/checkpoints/mixtral-10steps"

smp_estimator = PyTorch(
    # ... same config as before ...
    checkpoint_s3_uri=checkpoint_s3_uri,  # Same path as previous job
)

smp_estimator.fit(inputs=data_channels)
```

**ANNOTATION - EP Checkpoint Structure:**

**Checkpoints saved per EP group:**
```
/opt/ml/checkpoints/mixtral-10steps/
├── ep_rank_0/
│   ├── model_shard_0.pt   # FSDP shard 0, Experts 0-3
│   ├── model_shard_2.pt   # FSDP shard 2, Experts 0-3
│   ├── ...
│   └── model_shard_62.pt  # FSDP shard 62, Experts 0-3
├── ep_rank_1/
│   ├── model_shard_1.pt   # FSDP shard 1, Experts 4-7
│   ├── model_shard_3.pt   # FSDP shard 3, Experts 4-7
│   ├── ...
│   └── model_shard_63.pt  # FSDP shard 63, Experts 4-7
├── optimizer_shards/
│   └── ... (optimizer states per shard)
└── metadata.json
```

**Resume requirements:**
- **MUST match:** `expert_parallel_degree`, `hybrid_shard_degree`, `instance_count`
- **CAN change:** `batch_size`, `learning_rate`, `max_steps`
- **Why:** Checkpoint sharding pattern depends on parallelism config

**Failure modes:**
```
# Original job: EP=2, HSD=32, 64 GPUs
# Resume with: EP=4 → ERROR (expert distribution mismatch)
# Resume with: 32 GPUs → ERROR (FSDP shard count mismatch)
# Resume with: batch_size=4 → OK (doesn't affect sharding)
```

---

## SECTION 11: Key Takeaways for MoE Training

### 1. **Expert Parallelism is Essential for MoE**
- Distributes independent expert networks across GPUs
- Reduces memory per GPU without splitting tensors
- Uses All-to-All for token routing (different from All-Reduce in TP)

### 2. **Load Balancing Prevents Expert Collapse**
```python
"moe_load_balancing": "sinkhorn"  # Ensures all experts used equally
```
- Without balancing: Some experts over-trained, others ignored
- With balancing: Even training, better model quality

### 3. **Communication Pattern is Different**
**Dense Models (TP):**
- All-Reduce: Combine tensor slices
- Happens at every linear layer
- Data size: Activation dimensions

**MoE Models (EP):**
- All-to-All: Route tokens to experts
- Happens only at MoE layers
- Data size: Token embeddings × active experts

### 4. **Scaling Recommendations**

| Model | Experts | Expert Size | Min GPUs | Recommended EP | Recommended HSD |
|-------|---------|-------------|----------|----------------|-----------------|
| Mixtral 8x7B | 8 | 7B | 32 | 2 | 16-32 |
| Mixtral 8x22B | 8 | 22B | 64 | 2-4 | 32-64 |
| Switch-C (2048 experts) | 2048 | 1B | 256 | 256 | 1 |

### 5. **Instance Type Matters**
- **P4d:** Good for Mixtral 8x7B (400 Gbps network)
- **P4de:** Better (80GB GPU, same network)
- **P5:** Best (800 Gbps network, 80GB GPU) for Mixtral 8x22B

### 6. **FSx Recommended for Multi-Node**
- Fast checkpoint writes (100+ GB/s)
- Shared access across all nodes
- Critical for 64+ GPU training jobs

---

## Comparison: This Example vs Llama Example

| Aspect | Mixtral (EP) | Llama (TP) |
|--------|--------------|------------|
| **Parallelism Strategy** | Expert Parallelism | Tensor Parallelism |
| **Config Parameter** | `expert_parallel_degree=2` | `tensor_parallel_degree=2` |
| **What's Split** | 8 experts → 2 GPUs | Tensor dimensions |
| **Communication** | All-to-All (routing) | All-Reduce (combining) |
| **Memory Pattern** | Experts (local) + Non-experts (FSDP) | All params split equally |
| **Nodes Required** | 8 (64 GPUs) | 1 (8 GPUs) for 8B |
| **Model Size** | 46.7B total, ~13B active | 8B total, 8B active |
| **Use Case** | Sparse MoE models | Dense transformers |

---

## Next Steps

1. **Try different EP degrees:**
   - EP=4 for more memory savings
   - EP=1 if using p4de/p5 instances (80GB GPUs)

2. **Enable FP8 training:**
   - See `sm-fsdp-ep-fp8_train_mixtral.ipynb`
   - Requires P5 instances (H100 GPUs)
   - Further reduces memory by 50%

3. **Scale to Mixtral 8x22B:**
   - Increase `instance_count=16` (128 GPUs)
   - Use `expert_parallel_degree=4`
   - Enable activation offloading

4. **Production deployment:**
   - See TGI/LMI serving examples
   - MoE models need special serving config
   - Load balancing also important at inference

---

**End of Annotated Example 2**

*This annotation explained Expert Parallelism for MoE models, showing how it differs from Tensor Parallelism and why it's critical for training models like Mixtral efficiently. See other examples for FP8 training, fine-tuning workflows, and serving patterns.*
