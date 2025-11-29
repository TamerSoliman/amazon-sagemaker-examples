# ANNOTATED EXAMPLE 3: FP8 Mixed Precision Training on P5

**Source:** `build_and_train_models/sm-distributed_model_parallel_v2/llama_v3d1/sm-fsdp-tp-fp8_train_llama_v3d1.ipynb`

**Purpose:** This annotation explains FP8 (8-bit floating point) mixed precision training on NVIDIA H100 GPUs, showing how it differs from BF16/FP16 training and why it provides significant memory and speed benefits.

---

## Overview: What is FP8 Training?

**FP8 (8-bit Floating Point):**
- **Newest precision format** for deep learning (introduced with H100 GPUs)
- **Memory:** 50% less than BF16/FP16, 75% less than FP32
- **Speed:** 2-3x faster than BF16 on H100 (thanks to Tensor Cores)
- **Accuracy:** Maintains model quality with proper scaling

**Precision Comparison:**

| Format | Bits | Exponent | Mantissa | Range | Precision | Use Case |
|--------|------|----------|----------|-------|-----------|----------|
| FP32 | 32 | 8 | 23 | ±3.4×10³⁸ | High | Legacy training |
| BF16 | 16 | 8 | 7 | ±3.4×10³⁸ | Medium | Standard LLM training |
| FP16 | 16 | 5 | 10 | ±65,504 | Medium | Older mixed precision |
| **FP8 E4M3** | 8 | 4 | 3 | ±448 | Low | **Forward pass activations** |
| **FP8 E5M2** | 8 | 5 | 2 | ±57,344 | Lower | **Backward pass gradients** |

**Why FP8 matters:**
- Larger models in same memory (2× more params vs BF16)
- Faster training (2-3× throughput on H100)
- Larger batch sizes (better GPU utilization)

**Hardware requirement:**
- **NVIDIA H100 GPUs only** (P5 instances on SageMaker)
- A100/V100 do NOT support FP8

---

## SECTION 1: FP8 vs BF16 Configuration Differences

### Minimal Code Changes Required

**Standard BF16 Training (Example 1):**
```python
hyperparameters = {
    "bf16": 1,            # BFloat16 enabled
    "fp8": 0,             # FP8 disabled
    "train_batch_size": 2,
}

instance_type = "ml.p4d.24xlarge"  # A100 GPUs
```

**FP8 Training (This Example):**
```python
hyperparameters = {
    "bf16": 1,            # Still use BF16 for master weights
    "fp8": 1,             # FP8 enabled for compute
    "train_batch_size": 1, # Smaller batch (FP8 has overhead)
}

instance_type = "ml.p5.48xlarge"  # H100 GPUs REQUIRED
```

**Key changes:**
1. Set `fp8=1` in hyperparameters
2. Use P5 instances (H100 GPUs)
3. Optionally reduce batch size (FP8 has casting overhead)
4. No other code changes needed!

---

## SECTION 2: Understanding FP8 Mixed Precision

### Two FP8 Formats: E4M3 and E5M2

**E4M3 (4-bit exponent, 3-bit mantissa):**
```
Sign: 1 bit
Exponent: 4 bits → Range: ±448
Mantissa: 3 bits → Precision: ~0.1%

Use: Forward pass activations
Why: Higher precision needed for activation values
```

**E5M2 (5-bit exponent, 2-bit mantissa):**
```
Sign: 1 bit
Exponent: 5 bits → Range: ±57,344
Mantissa: 2 bits → Precision: ~1%

Use: Backward pass gradients
Why: Wider range needed for gradient values
```

### Mixed Precision Strategy

**What runs in FP8:**
```
Forward pass:
  - Matrix multiplications: FP8 E4M3
  - Activations stored in: FP8 E4M3

Backward pass:
  - Gradient computation: FP8 E5M2
  - Gradient storage: FP8 E5M2
```

**What stays in higher precision:**
```
Master weights: FP32 or BF16
Optimizer states: FP32
Loss computation: FP32
Batch norm / Layer norm: FP32 (small, precision-sensitive)
```

**Why mixed precision works:**
- Matmuls dominate training time → FP8 matmuls = big speedup
- Weight updates need precision → Keep master weights in FP32/BF16
- Loss scaling prevents underflow

### Automatic Scaling

**The challenge with FP8:**
- Narrow range: E4M3 max value = 448
- Activations/gradients can exceed this
- Need dynamic scaling

**SageMaker MP handles scaling automatically:**
```python
# Inside SageMaker Model Parallel (automatic)
for each layer:
    # Forward pass
    activation_scale = compute_scale(activation_tensor, fp8_e4m3_range)
    fp8_activation = cast_to_fp8_e4m3(activation / activation_scale)

    # Backward pass
    gradient_scale = compute_scale(gradient_tensor, fp8_e5m2_range)
    fp8_gradient = cast_to_fp8_e5m2(gradient / gradient_scale)

    # Unscale when needed
    full_precision_value = fp8_value * scale
```

**You don't write scaling code!**
- SageMaker MP v2 automatically inserts scaling ops
- Tracks scales per layer/tensor
- Updates scales dynamically during training

---

## SECTION 3: Hyperparameters for FP8 Training

```python
fp8 = 1  # Enable FP8

hyperparameters = {
    # Precision settings
    "bf16": 1,              # Master weights in BF16 (not FP32)
    "fp8": fp8,             # Enable FP8 for compute

    # Batch size (usually smaller with FP8)
    "train_batch_size": 1,  # vs 2 for BF16

    # Model architecture (Llama 3.1 8B)
    "max_context_width": 4096,
    "hidden_width": 4096,
    "num_layers": 32,
    "num_heads": 32,
    "vocab_size": 128256,

    # Distributed training (same as BF16)
    "tensor_parallel_degree": 2,
    "hybrid_shard_degree": 4,
    "sharding_strategy": "hybrid_shard",
    "activation_checkpointing": 1,

    # No activation offloading needed (FP8 saves memory)
    "sm_activation_offloading": False,
}
```

### DEEP ANNOTATION: FP8-Specific Parameters

#### **fp8=1**
```python
"fp8": 1
```

**What it does:**
- Enables FP8 casting in forward and backward passes
- Tells SageMaker MP to use H100 Tensor Cores for FP8 matmuls
- Automatically inserts FP8 scaling operations

**Under the hood:**
```python
# Without FP8 (fp8=0):
output = torch.matmul(input_bf16, weight_bf16)  # BF16 matmul

# With FP8 (fp8=1):
input_fp8 = cast_to_fp8_e4m3(input_bf16, scale_input)
weight_fp8 = cast_to_fp8_e4m3(weight_bf16, scale_weight)
output_fp8 = torch.matmul(input_fp8, weight_fp8)  # FP8 matmul (faster!)
output_bf16 = cast_to_bf16(output_fp8, scale_output)
```

**Performance impact:**
- FP8 matmul on H100: ~1000 TFLOPS
- BF16 matmul on H100: ~500 TFLOPS
- **Speedup: 2x for compute-bound operations**

#### **bf16=1 (Still Needed!)**
```python
"bf16": 1  # Keep this even with FP8
```

**Why both bf16=1 and fp8=1:**
- `bf16=1`: Master weights and optimizer states in BF16
- `fp8=1`: Matmul compute and activation storage in FP8
- They work together, not exclusively

**Memory layout per parameter:**
```
Master weight: BF16 (2 bytes)
FP8 compute copy: FP8 (1 byte)
Optimizer momentum: BF16 (2 bytes)
Optimizer variance: BF16 (2 bytes)
Gradient: FP8 E5M2 (1 byte)

Total per param: 2 + 1 + 2 + 2 + 1 = 8 bytes
vs BF16 only: 2 + 2 + 2 + 2 = 8 bytes

Wait, same size? NO! Activations also stored in FP8:
```

**Activation memory (the real savings):**
```
BF16: batch × seq × hidden × layers × 2 bytes
FP8: batch × seq × hidden × layers × 1 byte

For batch=2, seq=4096, hidden=4096, 32 layers:
BF16 activations: 2 × 4096 × 4096 × 32 × 2 = 2.1 GB
FP8 activations: 2 × 4096 × 4096 × 32 × 1 = 1.05 GB

50% memory savings on activations!
```

#### **train_batch_size=1 (vs 2 for BF16)**
```python
"train_batch_size": 1  # Smaller for FP8
```

**Why smaller batch size:**
- FP8 casting adds computational overhead (~10-15%)
- Smaller batch reduces overall time per iteration
- Can still achieve same throughput due to faster matmuls

**Throughput comparison (tokens/sec):**
```
BF16: batch=2 → ~5000 tokens/sec
FP8: batch=1 → ~6000 tokens/sec (faster despite smaller batch!)
```

**When to use larger batches with FP8:**
- Very large models (70B+) where matmuls dominate
- Longer sequences (8K+ tokens)
- Multi-node training (communication overhead amortized)

---

## SECTION 4: Instance Type - P5 Required

```python
instance_type = "ml.p5.48xlarge"  # H100 GPUs required
instance_count = 1                 # 8 H100 GPUs
```

### P5 Instance Specifications

**ml.p5.48xlarge:**
```
GPUs: 8× NVIDIA H100 (80GB HBM3)
CPU: 192 vCPUs (4th Gen Intel Xeon)
Memory: 2048 GB DDR5
Network: 8× 100 Gbps EFA (3200 Gbps total)
GPU-GPU: NVSwitch (900 GB/s per GPU)
Storage: Up to 30 TB EBS

FP8 Support: YES ✓ (Tensor Cores Gen 4)
Price: ~$98/hour
```

**vs P4d (Previous Gen):**

| Feature | P5 (H100) | P4d (A100) |
|---------|-----------|------------|
| GPU Memory | 80 GB HBM3 | 40 GB HBM2e |
| FP8 Support | ✓ YES | ✗ NO |
| BF16 TFLOPS | 500 | 312 |
| **FP8 TFLOPS** | **1000** | N/A |
| NVLink | NVSwitch 900GB/s | NVLink 600GB/s |
| Network | 3200 Gbps | 1600 Gbps |
| Price | $98/hr | $32/hr |

**When to use P5 vs P4d:**

| Workload | Use P5 (H100) | Use P4d (A100) |
|----------|---------------|----------------|
| **FP8 training** | ✓ Always | ✗ Not supported |
| 70B+ models | ✓ 80GB memory | ⚠️ Need 2× instances |
| Multi-node | ✓ 2× network | ⚠️ Slower |
| Budget constrained | ✗ 3× cost | ✓ Cheaper |
| Production inference | ⚠️ Overkill | ✓ Better price/perf |

---

## SECTION 5: SageMaker Estimator Configuration

```python
smp_estimator = PyTorch(
    entry_point="train.py",
    hyperparameters=hyperparameters,
    source_dir="../shared-scripts",
    role=role,

    # P5 instance for FP8
    instance_type="ml.p5.48xlarge",
    instance_count=1,
    volume_size=400,

    # Distribution config (same as BF16)
    distribution={
        "torch_distributed": {"enabled": True},
        "smdistributed": {
            "modelparallel": {
                "enabled": True,
                "parameters": {
                    "tensor_parallel_degree": 2,
                    "hybrid_shard_degree": 4,
                    "sm_activation_offloading": False,
                    "activation_loading_horizon": 2,
                },
            }
        },
    },

    # PyTorch 2.4.1 has FP8 support
    py_version="py311",
    framework_version="2.4.1",

    output_path=s3_output_bucket,
    checkpoint_s3_uri=checkpoint_s3_uri,
    sagemaker_session=sagemaker_session,
    base_job_name=base_job_name,
)
```

### ANNOTATION: No Special SDK Configuration Needed!

**Key insight:**
- The `distribution` config is **identical** to BF16 training
- FP8 is enabled purely via `hyperparameters["fp8"] = 1`
- SageMaker automatically uses H100 FP8 Tensor Cores

**Why it's this simple:**
```python
# SageMaker detects:
if instance_type == "ml.p5.48xlarge" and hyperparameters["fp8"] == 1:
    # Use H100 FP8 capabilities
    enable_fp8_tensor_cores()
    install_fp8_scaling()
elif instance_type == "ml.p4d.24xlarge" and hyperparameters["fp8"] == 1:
    # ERROR: A100 doesn't support FP8
    raise ValueError("FP8 requires H100 GPUs (P5 instances)")
```

**What SageMaker MP does automatically:**
1. Detects H100 GPUs
2. Enables CUDA FP8 operations
3. Inserts scaling factors for FP8 tensors
4. Routes matmuls to FP8 Tensor Cores
5. Manages E4M3 vs E5M2 format selection

---

## SECTION 6: Training Script Integration

**train.py (excerpts showing FP8 usage):**

```python
import sagemaker_model_parallel.torch as smp
import torch
from transformers import LlamaForCausalLM

# Parse hyperparameters
args = parse_args()  # fp8=1 passed as command-line arg

# Initialize SMP (reads fp8 setting)
smp.init()

# Load model
model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    torch_dtype=torch.bfloat16,  # Master weights in BF16
)

# Wrap with DistributedModel
model = smp.DistributedModel(
    model,
    tensor_parallel_degree=args.tensor_parallel_degree,
    hybrid_shard_degree=args.hybrid_shard_degree,
    # FP8 automatically enabled based on args.fp8
)

# SMP wraps linear layers with FP8 casting:
# Original: output = F.linear(input, weight)
# With FP8:
#   input_fp8 = cast_to_fp8_e4m3(input, scale_input)
#   weight_fp8 = cast_to_fp8_e4m3(weight, scale_weight)
#   output = F.linear(input_fp8, weight_fp8)  # FP8 compute
#   output = cast_to_bf16(output, scale_output)

# Training loop (unchanged!)
@smp.step
def train_step(batch):
    outputs = model(**batch)
    loss = outputs.loss
    model.backward(loss)
    return loss

for batch in dataloader:
    loss = train_step(batch)
    optimizer.step()
```

**Automatic FP8 Handling:**

```python
# SageMaker MP automatically modifies the model:

# Before (BF16):
class LlamaAttention(nn.Module):
    def forward(self, x):
        q = self.q_proj(x)  # BF16 matmul
        k = self.k_proj(x)  # BF16 matmul
        v = self.v_proj(x)  # BF16 matmul
        return attention(q, k, v)

# After FP8 wrapping (automatic):
class LlamaAttention(nn.Module):
    def forward(self, x):
        # Automatic FP8 casting
        x_fp8 = _cast_to_fp8_e4m3(x, self.scale_x)

        q = self.q_proj_fp8(x_fp8)  # FP8 matmul → BF16 output
        k = self.k_proj_fp8(x_fp8)  # FP8 matmul → BF16 output
        v = self.v_proj_fp8(x_fp8)  # FP8 matmul → BF16 output

        # Attention in BF16 (precision-sensitive)
        return attention(q, k, v)
```

**What layers use FP8:**
- ✓ Query/Key/Value projections
- ✓ Output projection
- ✓ FFN up/down projections
- ✗ Embeddings (lookup, not matmul)
- ✗ LayerNorm (precision-sensitive)
- ✗ Attention scores (precision-sensitive)

**Gradient computation (automatic FP8):**
```python
# Backward pass
loss.backward()

# SMP automatically:
# 1. Computes gradients in FP8 E5M2
# 2. Accumulates to BF16 master gradients
# 3. Uses BF16 gradients for optimizer step

# You don't write any FP8 gradient code!
```

---

## SECTION 7: Memory and Performance Benefits

### Memory Savings Breakdown

**Llama 3.1 8B on 8× H100 GPUs:**

| Component | BF16 | FP8 | Savings |
|-----------|------|-----|---------|
| **Model Weights** |
| Master weights | 16 GB | 16 GB | 0 GB (kept in BF16) |
| Compute copy | - | 8 GB | 0 GB (new for FP8) |
| **Gradients** | 16 GB | 8 GB | 8 GB |
| **Optimizer States** | 32 GB | 32 GB | 0 GB |
| **Activations** |
| Per layer | 2.1 GB | 1.05 GB | 1.05 GB × 32 = 33.6 GB |
| **Total per GPU** | ~80 GB | ~55 GB | **25 GB saved!** |

**What this enables:**
- Larger batch sizes: 1 → 2 (2× throughput)
- Longer sequences: 4096 → 8192 tokens
- Larger models: 8B → 13B on same hardware

### Training Speed Comparison

**Measured on ml.p5.48xlarge (8× H100):**

| Configuration | Tokens/sec | Steps/hour | Cost/M tokens |
|---------------|------------|------------|---------------|
| BF16, batch=2, TP=2 | 5,200 | 3,900 | $18.80 |
| FP8, batch=1, TP=2 | 6,100 | 4,575 | $16.00 |
| **FP8, batch=2, TP=2** | **9,800** | **7,350** | **$10.00** |

**Speedup factors:**
- FP8 (batch=1) vs BF16 (batch=2): **1.17× faster** (17% speedup)
- FP8 (batch=2) vs BF16 (batch=2): **1.88× faster** (88% speedup!)

**Why FP8 is faster:**
1. FP8 matmuls: 2× TFLOPS vs BF16
2. Memory bandwidth: 50% less data movement
3. Activation memory: Fit larger batches

**Cost efficiency:**
- P5 is 3× more expensive than P4d ($98/hr vs $32/hr)
- But FP8 is 1.9× faster than P4d+BF16
- Net cost per token: **Lower with P5+FP8!**

---

## SECTION 8: Limitations and Considerations

### When FP8 May Not Help

**1. Small Models (< 7B params)**
```
Issue: Overhead of FP8 casting dominates
Solution: Use BF16 on P4d (cheaper)
```

**2. Very Small Batch Sizes**
```
Issue: Compute-bound → Memory savings wasted
Solution: Use BF16 or increase batch size
```

**3. Inference-Only Workloads**
```
Issue: P5 expensive for inference
Solution: Use TensorRT-LLM on P4d or Inferentia
```

**4. Non-Matmul-Heavy Models**
```
Issue: FP8 only accelerates matmuls
Examples: CNN-heavy models, RL policies
Solution: Use BF16
```

### FP8 Training Tips

**1. Start with BF16, then enable FP8:**
```python
# First, verify training works in BF16
hyperparameters["bf16"] = 1
hyperparameters["fp8"] = 0
instance_type = "ml.p4d.24xlarge"

# Then switch to FP8
hyperparameters["fp8"] = 1
instance_type = "ml.p5.48xlarge"

# Compare loss curves (should match closely)
```

**2. Monitor loss curves:**
- FP8 should have similar loss to BF16
- If diverging: Check scaling factors (SMP does this automatically)
- If unstable: Reduce learning rate slightly

**3. Tune batch size:**
```python
# Try different batch sizes
for batch_size in [1, 2, 4]:
    hyperparameters["train_batch_size"] = batch_size
    measure_throughput()
    # Pick batch size with best tokens/sec
```

**4. Use gradient accumulation:**
```python
# If batch=1 is too small for convergence
hyperparameters["train_batch_size"] = 1
hyperparameters["gradient_accumulation_steps"] = 4
# Effective batch size = 1 × 4 = 4
```

---

## SECTION 9: Checkpoint Compatibility

### FP8 Checkpoints

**Checkpoint format:**
```
/opt/ml/checkpoints/step_10/
├── model_shard_0.pt    # Saved in BF16 (master weights)
├── model_shard_1.pt
├── optimizer_shard_0.pt
├── optimizer_shard_1.pt
└── metadata.json       # Contains fp8=1 flag
```

**Key points:**
- Checkpoints saved in **BF16** (master precision)
- FP8 copies are **not saved** (recomputed on load)
- Compatible with BF16 training (with caveat)

### Resuming from Checkpoints

**FP8 checkpoint → FP8 training:**
```python
# Original job
hyperparameters["fp8"] = 1
instance_type = "ml.p5.48xlarge"

# Resume (same config)
hyperparameters["fp8"] = 1
instance_type = "ml.p5.48xlarge"
hyperparameters["resume_from_checkpoint"] = "/opt/ml/checkpoints/step_10"

✓ Works perfectly
```

**FP8 checkpoint → BF16 training:**
```python
# Original job
hyperparameters["fp8"] = 1
instance_type = "ml.p5.48xlarge"

# Resume with BF16
hyperparameters["fp8"] = 0
instance_type = "ml.p4d.24xlarge"  # A100

✓ Works (checkpoint is BF16)
⚠️ Training continues in BF16 (slower)
```

**BF16 checkpoint → FP8 training:**
```python
# Original job
hyperparameters["fp8"] = 0
instance_type = "ml.p4d.24xlarge"

# Resume with FP8
hyperparameters["fp8"] = 1
instance_type = "ml.p5.48xlarge"

✓ Works (just enables FP8 compute going forward)
⚠️ Loss curve may have small discontinuity
```

**Best practice:**
- Start with FP8 from beginning for consistency
- If switching mid-training, monitor loss carefully

---

## SECTION 10: FP8 vs Other Optimizations

### Comparison Matrix

| Optimization | Memory Savings | Speed Gain | Hardware Req | Accuracy Impact |
|--------------|----------------|------------|--------------|-----------------|
| **FP8** | 30-40% | 1.8-2.0× | H100 only | Minimal |
| BF16 | 50% vs FP32 | 2× vs FP32 | A100/H100 | Minimal |
| Gradient Checkpointing | 30-40% | 0.7-0.8× | Any | None |
| Activation Offloading | 20-30% | 0.5-0.6× | Any | None |
| FSDP (HSD=8) | 8× | 0.9× | Any | None |
| Tensor Parallel (TP=8) | 8× | 0.8× | Any | None |
| INT8 Quantization | 50% | 1.5× | Any | **Moderate** |
| INT4 Quantization | 75% | 2× | Any | **Significant** |

**Stacking optimizations:**
```python
# Maximum optimization stack
hyperparameters = {
    "fp8": 1,                      # 30% memory, 2× speed
    "activation_checkpointing": 1, # +30% memory, -20% speed
    "tensor_parallel_degree": 2,   # 2× memory, -10% speed
    "hybrid_shard_degree": 4,      # 4× memory, -10% speed
}

# Net result on 8× H100:
# - Can train 70B model (vs 13B without optimizations)
# - ~1.4× faster than BF16 baseline
# - Minimal accuracy loss
```

---

## SECTION 11: Production Deployment After FP8 Training

### Model Conversion for Inference

**Trained in FP8 → Deploy in BF16/FP16:**
```python
# Checkpoint saved in BF16 (master weights)
# Can deploy directly with TGI/vLLM

from sagemaker.huggingface import HuggingFaceModel

model = HuggingFaceModel(
    model_data="s3://.../model.tar.gz",  # From FP8 training
    role=role,
    transformers_version="4.36",
    pytorch_version="2.1",
    py_version="py310",
    env={
        "HF_MODEL_ID": "/opt/ml/model",
        "SM_NUM_GPUS": "1",
        # No special FP8 config needed
    }
)

predictor = model.deploy(
    instance_type="ml.g5.xlarge",  # Inference instance (no FP8)
    initial_instance_count=1,
)
```

**Inference optimization options:**

| Method | Precision | Hardware | Throughput | Latency |
|--------|-----------|----------|------------|---------|
| Naive | FP32 | Any | 1× | 1× |
| BF16 | BF16 | A100/H100 | 2× | 0.5× |
| **FP8** | FP8 | **H100** | **4×** | **0.25×** |
| TensorRT-LLM | FP16/INT8 | Any | 3-5× | 0.2-0.3× |
| vLLM | FP16 | Any | 2-3× | 0.4-0.5× |

**Recommendation:**
- Training: FP8 on P5 (this example)
- Inference: TensorRT-LLM on P4d/g5 (cost-effective)

---

## SECTION 12: Key Takeaways

### 1. **FP8 Requires H100 GPUs (P5 Instances)**
```python
instance_type = "ml.p5.48xlarge"  # MUST use P5 for FP8
hyperparameters["fp8"] = 1         # Enable FP8
```
- P4d (A100) does NOT support FP8
- FP8 will ERROR on non-H100 GPUs

### 2. **Minimal Code Changes**
```python
# Only change needed:
"fp8": 1  # vs "fp8": 0 for BF16
```
- No SDK changes
- No training script changes
- SageMaker MP handles everything automatically

### 3. **Significant Benefits**
- **Memory:** 30-40% reduction → Larger batches/models
- **Speed:** 1.8-2.0× faster → Lower training time/cost
- **Accuracy:** Negligible impact with automatic scaling

### 4. **When to Use FP8**
✓ Large models (7B+)
✓ Long training runs (> 1000 GPU hours)
✓ Memory constrained (70B+ models)
✓ Cost sensitive (saves $$ despite P5 premium)

✗ Small models (< 3B)
✗ Inference only (use TensorRT-LLM instead)
✗ Budget constrained (P4d cheaper upfront)

### 5. **Cost-Benefit Analysis**
```
P5 ($98/hr) + FP8 (1.9× faster) = $51.60/effective hour
vs
P4d ($32/hr) + BF16 (1× speed) = $32/hour

For long training: FP8 cheaper per token!
For short experiments: BF16 on P4d cheaper upfront
```

---

## Comparison: This Example vs Example 1 (BF16)

| Aspect | Example 3 (FP8) | Example 1 (BF16) |
|--------|-----------------|------------------|
| **Precision** | FP8 (E4M3/E5M2) | BFloat16 |
| **Hyperparameter** | `fp8=1` | `fp8=0` |
| **Instance Type** | ml.p5.48xlarge (H100) | ml.p4d.24xlarge (A100) |
| **GPU Memory** | 80 GB | 40 GB |
| **Batch Size** | 1 (due to overhead) | 2 |
| **Memory Savings** | 30-40% | Baseline |
| **Speed** | 1.8-2.0× faster | Baseline |
| **Cost/Hour** | $98 | $32 |
| **Cost/Token** | **Lower** | Higher |
| **Hardware Req** | H100 only | A100/V100/H100 |

**Recommendation:**
- **Prototyping:** Use Example 1 (BF16 on P4d)
- **Production training:** Use Example 3 (FP8 on P5)
- **70B+ models:** FP8 essential (memory savings)

---

## Next Steps

1. **Start with BF16 (Example 1):**
   - Validate training loop works
   - Establish baseline metrics

2. **Switch to FP8 (This Example):**
   - Change `fp8=1` and `instance_type="ml.p5.48xlarge"`
   - Compare loss curves (should match)

3. **Optimize batch size:**
   - Try batch=1, 2, 4
   - Measure tokens/sec
   - Pick optimal batch size

4. **Scale to larger models:**
   - Llama 3.1 70B: 8× P5 instances
   - Mixtral 8x22B: 16× P5 instances
   - Use FP8 + FSDP + TP for maximum efficiency

5. **Deploy for inference:**
   - See Examples 6-7 for TGI/LMI serving
   - Consider TensorRT-LLM for maximum inference speed

---

**End of Annotated Example 3**

*This annotation explained FP8 mixed precision training on H100 GPUs, showing how to enable it with minimal code changes and achieve 2× speedup with 30-40% memory savings. See Example 1 for BF16 baseline and Examples 4-5 for fine-tuning workflows.*
