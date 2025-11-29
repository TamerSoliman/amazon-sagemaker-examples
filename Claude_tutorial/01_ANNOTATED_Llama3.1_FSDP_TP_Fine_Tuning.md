# ANNOTATED EXAMPLE 1: Llama 3.1 FSDP+TP Fine-Tuning

**Source:** `build_and_train_models/sm-distributed_model_parallel_v2/llama_v3d1/sm-fsdp-tp_finetuning_llama_v3d1.ipynb`

**Purpose:** This annotation explains every SageMaker SDK call and distributed training configuration for fine-tuning Llama 3.1 with FSDP (Fully Sharded Data Parallel) + TP (Tensor Parallelism).

---

## Overview

This notebook demonstrates:
- **What:** Fine-tuning Llama 3.1 (8B or 70B) on GLUE/SST2 dataset
- **How:** Using SageMaker Model Parallelism v2 with hybrid parallelism (FSDP + Tensor Parallelism)
- **Why:** To efficiently train large models across multiple GPUs/nodes by combining data and tensor parallelism

---

## SECTION 1: Environment Setup and Authentication

### Cell: SageMaker SDK Initialization

```python
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.pytorch import PyTorch

# WHAT: Retrieve the IAM role that SageMaker will assume during training
role = get_execution_role()
```

**ANNOTATION:**
- **What it does:** Gets the IAM execution role attached to your SageMaker notebook instance or local environment
- **Why it matters:** This role needs permissions to:
  - Read/write to S3 buckets (for data and model artifacts)
  - Create and manage SageMaker training jobs
  - Pull Docker images from ECR
  - Write logs to CloudWatch
- **Container impact:** This role is passed to the training container, allowing it to access AWS resources

```python
# WHAT: Get AWS account ID and region information
client = boto3.client("sts")
account = client.get_caller_identity()["Account"]

session = boto3.session.Session()
region = session.region_name
```

**ANNOTATION:**
- **What it does:** Retrieves your AWS account ID and current region
- **Why it matters:** Used to construct S3 paths and ensure resources are in the same region
- **Best practice:** Training jobs must be in the same region as your S3 data buckets

```python
# WHAT: Create SageMaker session and get default bucket
sagemaker_session = sagemaker.session.Session(boto_session=session)
default_bucket = sagemaker_session.default_bucket()
```

**ANNOTATION:**
- **What it does:**
  - `Session`: Manages interactions with SageMaker APIs
  - `default_bucket()`: Creates/retrieves S3 bucket named `sagemaker-{region}-{account}`
- **Why it matters:** This bucket stores:
  - Training data inputs
  - Model checkpoints
  - Final model artifacts
  - Training logs and metrics
- **Container impact:** This bucket path is mapped to `/opt/ml/` directories inside the container

---

## SECTION 2: Data Preparation and S3 Upload

### Cell: Load and Tokenize Dataset

```python
from datasets import load_dataset
from transformers import AutoTokenizer

# WHAT: Load GLUE/SST2 dataset from HuggingFace
raw_datasets = load_dataset("glue", "sst2")
```

**ANNOTATION:**
- **What it does:** Downloads the GLUE SST2 sentiment classification dataset
- **Dataset structure:**
  - Training: ~67K examples
  - Validation: ~870 examples (notebook creates custom split)
- **Why this dataset:** Small enough for demo but realistic for fine-tuning workflow

```python
# WHAT: Load tokenizer for Llama 3.1
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL, **tokenizer_kwargs)
```

**ANNOTATION:**
- **What it does:** Loads the Llama 3.1 tokenizer (BPE-based, vocab size 128,256)
- **Critical parameter:** `PRETRAINED_MODEL` must be set to:
  - HuggingFace model: `"meta-llama/Llama-3.1-8B"` (requires HF access token)
  - OR FSx path: `"/fsx/.../Llama-3.1-8B"`
- **Why tokenizer matters:** Converts text to token IDs that the model expects
- **Container usage:** Tokenization happens CLIENT-SIDE, tokenized data is uploaded to S3

```python
# WHAT: Tokenize and group texts into chunks of max_context_width (4096)
tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    num_proc=1,
    remove_columns=column_names,
)

lm_datasets = tokenized_datasets.map(
    functools.partial(group_texts, 4096),  # max_context_width for Llama 3.1
    batched=True,
)
```

**ANNOTATION:**
- **What it does:**
  1. Tokenizes all text into token IDs
  2. Groups tokens into chunks of 4096 (Llama 3.1's max context length)
  3. Creates `labels` by copying `input_ids` (for causal LM training)
- **Why chunk size matters:**
  - Llama 3.1 supports up to 128K context with RoPE scaling
  - This example uses 4096 for faster training
  - Larger chunks = more memory usage per GPU
- **Data flow:** Processed locally → Saved to JSON → Uploaded to S3

### Cell: Upload to S3

```python
# WHAT: Save tokenized datasets as JSON files
train_dataset.to_json("./training.json")
eval_dataset.to_json("./validation.json")

# WHAT: Define S3 paths for training and validation data
training_dataset_location = f"s3://{default_bucket}/{default_bucket_prefix_path}/dataset/train/"
validation_dataset_location = f"s3://{default_bucket}/{default_bucket_prefix_path}/dataset/validation/"

# WHAT: Upload JSON files to S3
os.system(f"aws s3 cp ./training.json {training_dataset_location}")
os.system(f"aws s3 cp ./validation.json {validation_dataset_location}")
```

**ANNOTATION:**
- **What it does:** Uploads preprocessed training/validation data to S3
- **S3 structure:**
  ```
  s3://sagemaker-{region}-{account}/dataset/
  ├── train/
  │   └── training.json
  └── validation/
      └── validation.json
  ```
- **Why S3:**
  - Durable storage accessible by all training instances
  - SageMaker automatically downloads from S3 to container's `/opt/ml/input/data/`
- **Data format:** JSON lines format with `input_ids` and `labels` fields
- **Container mapping (later):** These S3 paths → `/opt/ml/input/data/train/` and `/opt/ml/input/data/test/`

---

## SECTION 3: Define Data Channels (S3 vs FSx)

### Option A: S3 Data Channels

```python
# WHAT: Create SageMaker TrainingInput objects pointing to S3 data
train = sagemaker.inputs.TrainingInput(
    s3_train_bucket,
    distribution="FullyReplicated",  # Copy data to ALL nodes
    s3_data_type="S3Prefix"          # Treat as directory prefix
)

test = sagemaker.inputs.TrainingInput(
    s3_test_bucket,
    distribution="FullyReplicated",
    s3_data_type="S3Prefix"
)

data_channels = {"train": train, "test": test}
```

**ANNOTATION - TrainingInput Configuration:**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `s3_train_bucket` | `s3://.../dataset/train/` | S3 URI for training data |
| `distribution` | `"FullyReplicated"` | **CRITICAL:** Copy full dataset to EVERY training instance |
| `s3_data_type` | `"S3Prefix"` | Download all objects under this prefix |

**Distribution Strategies:**
1. **`FullyReplicated`** (used here):
   - Copies complete dataset to `/opt/ml/input/data/train/` on EVERY instance
   - Use when: Dataset is small enough to fit on each instance's EBS volume
   - Advantage: Fast local access, no coordination needed
   - Trade-off: N instances = N copies of data downloaded

2. **`ShardedByS3Key`** (alternative):
   - Distributes different S3 objects to different instances
   - Use when: Dataset is too large to fit on single instance
   - Requires: Training script to handle partial data per instance

**Container File System Mapping:**
```
S3: s3://bucket/dataset/train/training.json
  ↓ (SageMaker downloads during job start)
Container: /opt/ml/input/data/train/training.json
```

### Option B: FSx Lustre Data Channels (Recommended for Large Models)

```python
from sagemaker.inputs import FileSystemInput

# WHAT: Create FSx data input for high-performance file system access
train = FileSystemInput(
    file_system_id="fs-0abc123...",      # FSx file system ID
    file_system_type="FSxLustre",        # Lustre for HPC workloads
    directory_path="/3x5lhbmv",          # Mount path on FSx
    file_system_access_mode="rw",        # Read-write access
)

data_channels = {"train": train, "test": train}
```

**ANNOTATION - FSx Configuration:**

**Why FSx over S3 for large-scale training:**
1. **Performance:**
   - FSx: Sub-millisecond latency, 100s GB/s throughput
   - S3: Higher latency, better for cold storage
2. **Checkpointing:**
   - FSx: Fast checkpoint writes during training (critical for large models)
   - S3: Slower writes, better for final artifacts
3. **Shared access:**
   - Multiple training instances can read/write simultaneously
   - Essential for distributed checkpointing

**Network Requirements:**
- FSx must be in a **VPC private subnet** with internet gateway
- Training instances must use same **security group** and **subnet** as FSx
- This is configured later in the Estimator via `security_group_ids` and `subnets` kwargs

**Container Mounting:**
```
FSx: /3x5lhbmv/datasets/c4/en/hf-tokenized/llama/train
  ↓ (Mounted directly to container)
Container: /opt/ml/input/data/train/datasets/...
```

---

## SECTION 4: Distributed Training Configuration (THE CRITICAL PART)

### Hyperparameters: Distributed Training Settings

```python
tensor_parallel_degree = 2
hybrid_shard_degree = 4
offload_activations = True
activation_loading_horizon = 2
```

**ANNOTATION - Parallelism Strategy:**

#### **Tensor Parallel Degree (TP)**
```python
tensor_parallel_degree = 2  # Split model tensors across 2 GPUs
```

**What it does:**
- Splits individual model layers (weights, activations) across `TP` GPUs
- Example: For a 4096x4096 weight matrix, each GPU holds 4096x2048

**Why use it:**
- Enables fitting models larger than single GPU memory
- Reduces per-GPU memory by factor of ~TP

**Communication pattern:**
- **All-Reduce** after each tensor operation
- Requires LOW latency → Use within single node only (TP ≤ 8)

**Example with TP=2 on 8 GPUs:**
```
Node 1: GPU 0-1 (TP group 1), GPU 2-3 (TP group 2), ...
        Each pair shares model layer tensors
```

**Recommended values:**
- TP=1: No tensor parallelism (model fits on 1 GPU)
- TP=2: Halve memory per GPU
- TP=4: Quarter memory per GPU
- TP=8: Use all GPUs in a node for one model copy
- TP>8: NOT recommended (inter-node TP is slow)

#### **Hybrid Shard Degree (HSD)**
```python
hybrid_shard_degree = 4  # FSDP shards across 4 GPUs
```

**What it does:**
- Controls FSDP sharding granularity
- Shards model parameters, gradients, optimizer states across `HSD` GPUs
- Replicates across groups

**Math with 8 GPUs total, TP=2, HSD=4:**
```
Total GPUs = 8
TP groups = 8 / 2 = 4 groups of 2 GPUs each
Each TP group is further FSDP-sharded with HSD=4

Actually: With TP=2, effective world size for FSDP = 8/2 = 4
So HSD=4 means FULL_SHARD across all 4 TP groups
```

**How HSD affects memory:**
- HSD=0: Fallback to PyTorch native FSDP (usually FULL_SHARD across all GPUs)
- HSD=1: No sharding (replicates model everywhere) - highest memory
- HSD=8: Shard within 8-GPU node, replicate across nodes - balanced
- HSD=world_size: Full sharding across all GPUs - lowest memory per GPU

**Communication pattern:**
- **All-Gather** before forward/backward (fetch sharded params)
- **Reduce-Scatter** after backward (shard gradients)
- With HSD=8, communication only within node (fast)

**Recommended strategy:**
- Start with smallest HSD that doesn't OOM
- If OOM → Increase HSD to shard more aggressively
- For multi-node: HSD=8 keeps expensive comms within node

#### **Activation Offloading**
```python
offload_activations = True  # Offload activations to CPU
activation_loading_horizon = 2  # Pre-fetch 2 layers ahead
```

**What it does:**
- Offloads activations from GPU → CPU during forward pass
- Pre-fetches them back to GPU before backward pass
- Requires `activation_checkpointing=1` to be effective

**Why use it:**
- Activations grow with batch_size × sequence_length × hidden_dim
- For Llama 70B, activations can exceed 100GB per GPU
- Offloading trades GPU memory for CPU-GPU bandwidth

**SageMaker Enhancement (`sm_activation_offloading`):**
- Standard PyTorch: Offload synchronously (GPU waits)
- SageMaker: **Async prefetching** with `activation_loading_horizon`
  - Loads activations for layer N+2 while computing layer N
  - Hides CPU-GPU transfer latency
  - Requires tuning horizon (2-4 typical)

**Trade-offs:**
- ✅ Enables larger batch sizes or models
- ❌ Adds CPU-GPU PCIe bandwidth overhead
- ❌ Only beneficial for models ≥20B params or when hitting OOM

**When to use:**
- Model ≥20B parameters
- Getting OOM errors with reasonable batch size
- Have CPU memory headroom

### Hyperparameters: Model and Training Configuration

```python
hyperparameters = {
    # Llama 3.1 specific
    "rope_scaling_type": "llama3",
    "rotary_emb_base": 500000,
    "vocab_size": 128256,

    # Training hyperparameters
    "train_batch_size": 2,
    "val_batch_size": 4,
    "max_steps": 50,
    "lr": 0.0001,
    "warmup": 0.0032,

    # Memory optimization
    "activation_checkpointing": 1,
    "bf16": 1,
    "fp8": 0,

    # Distributed settings
    "sharding_strategy": "hybrid_shard",
    "auto_wrap_policy": "transformer_auto_wrap_policy",

    # Checkpointing
    "checkpoint_freq": 50,
    "checkpoint_dir": "/opt/ml/checkpoints",
    "num_kept_checkpoints": 2,

    # Model size (8B config)
    "max_context_width": 4096,
    "hidden_width": 4096,
    "num_layers": 32,
    "num_heads": 32,
    "llama_intermediate_size": 14336,
}
```

**ANNOTATION - Key Hyperparameters:**

#### **RoPE Scaling (Context Length Extension)**
```python
"rope_scaling_type": "llama3"
"rotary_emb_base": 500000  # Theta value
```

**What it does:**
- Llama 3.1 uses RoPE (Rotary Position Embeddings) for position encoding
- `rotary_emb_base` controls frequency basis (higher = better long context)
- Llama 3.1 default: 500,000 (vs 10,000 for Llama 2)

**Impact:**
- Enables training/inference up to 128K context length
- This example uses 4096 for speed, but model supports longer

#### **Precision Settings**
```python
"bf16": 1       # BFloat16 mixed precision
"fp8": 0        # FP8 disabled (see separate example)
```

**What it does:**
- `bf16=1`: Train in BFloat16 (16-bit) instead of FP32 (32-bit)
  - Reduces memory by ~50%
  - Maintains FP32 range (better than FP16 for LLMs)
  - Native support on A100/H100 GPUs
- `fp8=0`: FP8 (8-bit) disabled here
  - FP8 available on H100 GPUs
  - See `sm-fsdp-tp-fp8_train_llama_v3d1.ipynb` for FP8 example

#### **Activation Checkpointing**
```python
"activation_checkpointing": 1
```

**What it does:**
- Doesn't store ALL activations during forward pass
- Recomputes activations during backward pass
- Trades compute for memory

**Impact:**
- Reduces memory by ~30-40%
- Increases training time by ~20-30%
- Essential for large models

**How it works:**
```
Without checkpointing:
  Forward: Store all activations → Backward: Use stored activations

With checkpointing:
  Forward: Store only checkpoint activations → Backward: Recompute missing activations
```

#### **Sharding Strategy**
```python
"sharding_strategy": "hybrid_shard"
"auto_wrap_policy": "transformer_auto_wrap_policy"
```

**What it does:**
- `hybrid_shard`: Use FSDP hybrid sharding (controlled by `hybrid_shard_degree`)
- `transformer_auto_wrap_policy`: Automatically wrap each Transformer layer as FSDP unit
  - Granularity: One FSDP unit per Transformer block
  - Alternative: `size_based_auto_wrap_policy` (wraps by parameter count)

**Why transformer wrapping:**
- Natural boundary for LLMs (each layer is ~same size)
- Balanced communication vs computation
- Works well with activation checkpointing

#### **Checkpointing Configuration**
```python
"checkpoint_freq": 50           # Save every 50 steps
"checkpoint_dir": "/opt/ml/checkpoints"
"num_kept_checkpoints": 2       # Keep only 2 most recent
```

**What it does:**
- Saves model state every 50 training steps
- Stores in `/opt/ml/checkpoints/` (container path)

**Where checkpoints go:**

**With S3 (checkpoint_s3_uri):**
```
Container: /opt/ml/checkpoints/step_50/
  ↓ (SageMaker syncs to S3 periodically)
S3: s3://bucket/.../checkpoints/step_50/
```

**With FSx (checkpoint_local_path):**
```
Container: /opt/ml/checkpoints/step_50/
  ↓ (Writes directly to FSx mount)
FSx: /3x5lhbmv/smp-v2/llama_v3/checkpointdir/step_50/
```

**Why `num_kept_checkpoints=2`:**
- Checkpoints for 70B model can be 100+ GB each
- Keeping only 2 prevents filling disk
- Deletes oldest when saving new checkpoint

#### **Model Architecture (8B vs 70B)**
```python
# 8B configuration
model_params = {
    "max_context_width": 4096,
    "hidden_width": 4096,
    "num_layers": 32,
    "num_heads": 32,
    "llama_intermediate_size": 14336,
}

# 70B configuration (commented out)
# model_params = {
#     "max_context_width": 4096,
#     "hidden_width": 8192,
#     "num_layers": 80,
#     "num_heads": 64,
#     "llama_intermediate_size": 28672,
# }
```

**ANNOTATION:**
- These parameters define the model architecture
- Must match the pretrained model you're loading
- Passed to training script to initialize model

**8B Model:**
- Parameters: ~8 billion
- GPU memory (BF16, no optimizations): ~16 GB
- Fits on: 1x A100 (40GB) with FSDP

**70B Model:**
- Parameters: ~70 billion
- GPU memory (BF16, no optimizations): ~140 GB
- Requires: Multi-GPU with FSDP+TP (8x A100 minimum)

### Fine-tuning Configuration

```python
# WHAT: Enable fine-tuning from pretrained model
if use_fsx:
    hyperparameters["hf_pretrained_model_name_or_dir"] = f"{SM_TRAIN_DIR}{PRETRAINED_DIR}"
else:
    hyperparameters["hf_pretrained_model_name_or_dir"] = PRETRAINED_MODEL
```

**ANNOTATION:**
- **What it does:** Activates fine-tuning mode in `train.py`
- **When set:** Script calls `AutoModelForCausalLM.from_pretrained()`
- **When not set:** Script initializes random weights (pretraining from scratch)

**Two options for pretrained model:**

**Option 1: HuggingFace Hub**
```python
"hf_pretrained_model_name_or_dir": "meta-llama/Llama-3.1-8B"
```
- Downloads from HuggingFace during training job start
- Requires: HF access token for gated models (Llama)
- Download time: ~5 min for 8B, ~30 min for 70B
- Each instance downloads independently (inefficient for multi-node)

**Option 2: FSx Pre-downloaded Model**
```python
"hf_pretrained_model_name_or_dir": "/opt/ml/input/data/train/models/Llama-3.1-8B"
```
- Model already stored on FSx
- All instances access shared FSx copy (faster)
- Recommended for multi-node training

**Container behavior:**
```python
# Inside train.py (simplified)
if "hf_pretrained_model_name_or_dir" in hyperparameters:
    # Fine-tuning: Load pretrained weights
    model = AutoModelForCausalLM.from_pretrained(
        hyperparameters["hf_pretrained_model_name_or_dir"],
        torch_dtype=torch.bfloat16,
    )
else:
    # Pretraining: Random initialization
    config = LlamaConfig(**model_params)
    model = LlamaForCausalLM(config)
```

---

## SECTION 5: SageMaker Estimator - The Core SDK Object

```python
smp_estimator = PyTorch(
    entry_point="train.py",
    hyperparameters=hyperparameters,
    source_dir=os.path.join(os.getcwd(), "../shared-scripts"),
    role=role,

    # Checkpointing
    checkpoint_s3_uri=checkpoint_s3_uri if not use_fsx else None,
    checkpoint_local_path=hyperparameters["checkpoint_dir"] if use_fsx else None,

    # Instance configuration
    instance_type="ml.p4d.24xlarge",
    volume_size=400,
    instance_count=1,

    # Session
    sagemaker_session=sagemaker_session,

    # DISTRIBUTED TRAINING CONFIGURATION (CRITICAL!)
    distribution={
        "torch_distributed": {"enabled": True},
        "smdistributed": {
            "modelparallel": {
                "enabled": True,
                "parameters": {
                    "tensor_parallel_degree": 2,
                    "hybrid_shard_degree": 4,
                    "sm_activation_offloading": True,
                    "activation_loading_horizon": 2,
                },
            }
        },
    },

    # Framework
    py_version="py311",
    framework_version="2.4.1",

    # Output
    output_path=s3_output_bucket,
    max_run=86400,

    # Other
    debugger_hook_config=False,
    base_job_name=base_job_name,
    metric_definitions=metric_definitions,

    # FSx networking (if using FSx)
    **kwargs,  # Contains security_group_ids and subnets
)
```

### DEEP ANNOTATION: PyTorch Estimator Parameters

#### **Entry Point and Source Code**
```python
entry_point="train.py"
source_dir="../shared-scripts"
```

**What happens:**
1. SageMaker packages `../shared-scripts/` directory into a tar.gz
2. Uploads to S3: `s3://bucket/.../source/sourcedir.tar.gz`
3. Training container downloads and extracts to `/opt/ml/code/`
4. Runs: `python /opt/ml/code/train.py` with hyperparameters as args

**Directory structure in container:**
```
/opt/ml/code/
├── train.py              (entry point)
├── arguments.py
├── checkpoints.py
├── data_utils.py
├── fsdp_utils.py
├── train_lib.py
└── ... (all files from source_dir)
```

**Hyperparameters passed as:**
```bash
python train.py \
  --train_batch_size 2 \
  --lr 0.0001 \
  --tensor_parallel_degree 2 \
  ...
```

#### **Checkpointing Configuration**

**Option 1: S3 Checkpointing**
```python
checkpoint_s3_uri="s3://bucket/experiments/smp_fsdp-llama-checkpoints/..."
checkpoint_local_path=None
```

**How it works:**
- Container writes checkpoints to `/opt/ml/checkpoints/`
- SageMaker **periodically** syncs `/opt/ml/checkpoints/` → S3 checkpoint_s3_uri
- Sync frequency: Every ~5 minutes (not configurable)
- On job end: Final sync ensures all checkpoints uploaded

**Resuming training:**
```python
# In train.py
if os.path.exists("/opt/ml/checkpoints/latest"):
    # SageMaker downloaded previous checkpoints from S3
    load_checkpoint("/opt/ml/checkpoints/latest")
```

**Limitation:**
- Slow for large checkpoints (70B = 140GB checkpoint)
- Sync lag can lose recent checkpoints if job crashes

**Option 2: FSx Checkpointing (Recommended)**
```python
checkpoint_s3_uri=None
checkpoint_local_path="/opt/ml/checkpoints"
```

**How it works:**
- `/opt/ml/checkpoints/` is mounted to FSx
- Writes go directly to FSx (no S3 sync)
- Much faster (100s GB/s vs S3's ~1 GB/s)

**Why it's better:**
- Immediate persistence (no sync lag)
- Fast writes don't slow down training
- Shared across instances for distributed checkpointing

#### **Instance Configuration**

```python
instance_type="ml.p4d.24xlarge"
volume_size=400
instance_count=1
```

**Instance Type Breakdown:**

| Instance | GPUs | GPU Type | GPU Memory | CPU | System RAM | Network |
|----------|------|----------|------------|-----|------------|---------|
| ml.p4d.24xlarge | 8 | A100 | 40GB each | 96 vCPUs | 1152 GB | 4x 100 Gbps |
| ml.p4de.24xlarge | 8 | A100 | **80GB** each | 96 vCPUs | 1152 GB | 4x 100 Gbps |
| ml.p5.48xlarge | 8 | H100 | 80GB each | 192 vCPUs | 2048 GB | 8x 100 Gbps |

**Choosing instance type:**
- **8B model:** 1x p4d.24xlarge sufficient
- **70B model:** ≥8x p4d.24xlarge (64 GPUs total) OR ≥2x p5.48xlarge

**Volume Size:**
- `volume_size=400`: Attaches 400 GB EBS volume to `/opt/ml/` (each instance)
- Used for: Training data, checkpoints (if not FSx), Docker layers
- Min for 70B: 500 GB (model + checkpoints + data)

**Instance Count:**
```python
instance_count=1  # 1 node = 8 GPUs
```

**Multi-node example:**
```python
instance_count=8  # 8 nodes = 64 GPUs total
```

**What changes with multi-node:**
- SageMaker sets `WORLD_SIZE=64`, `RANK=[0-63]`
- Requires: `torch_distributed` backend (NCCL for multi-node)
- Network: Uses EFA (Elastic Fabric Adapter) for low-latency GPU-GPU comms

#### **Distribution Configuration - THE HEART OF DISTRIBUTED TRAINING**

```python
distribution={
    "torch_distributed": {"enabled": True},
    "smdistributed": {
        "modelparallel": {
            "enabled": True,
            "parameters": {
                "tensor_parallel_degree": 2,
                "hybrid_shard_degree": 4,
                "sm_activation_offloading": True,
                "activation_loading_horizon": 2,
            },
        }
    },
}
```

**CRITICAL ANNOTATION - How This Works:**

#### **torch_distributed: PyTorch DDP Backend**
```python
"torch_distributed": {"enabled": True}
```

**What it does:**
- SageMaker runs training with `torchrun` (PyTorch distributed launcher)
- Sets environment variables for distributed training:
  ```bash
  WORLD_SIZE=8          # Total GPUs
  RANK=0-7              # GPU rank
  LOCAL_RANK=0-7        # GPU rank on node
  MASTER_ADDR=algo-1    # Hostname of rank 0
  MASTER_PORT=7777      # Port for rendezvous
  ```

**Container startup command:**
```bash
torchrun \
  --nproc_per_node=8 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=algo-1 \
  --master_port=7777 \
  /opt/ml/code/train.py <hyperparameters>
```

**What this enables:**
- Each GPU runs a separate process
- Processes communicate via NCCL (NVIDIA Collective Communications Library)
- Essential for FSDP/DDP

#### **smdistributed.modelparallel: SageMaker's Distributed Training Library**
```python
"smdistributed": {
    "modelparallel": {
        "enabled": True,
        "parameters": { ... }
    }
}
```

**What it does:**
- Installs SageMaker Model Parallel library v2 in container
- Library provides:
  - `smp.DistributedModel()`: Wrapper for FSDP+TP
  - `smp.DistributedOptimizer()`: Sharded optimizer
  - `smp.step()`: Distributed training step with gradient sync
- Configured via `parameters` dict

**Container environment variables set:**
```bash
SMP_TENSOR_PARALLEL_DEGREE=2
SMP_HYBRID_SHARD_DEGREE=4
SMP_ACTIVATION_OFFLOADING=1
SMP_ACTIVATION_LOADING_HORIZON=2
```

**How train.py uses this:**
```python
import sagemaker_model_parallel.torch as smp

smp.init()  # Initialize distributed training

# Wrap model
model = smp.DistributedModel(model,
    tensor_parallel_degree=2,
    hybrid_shard_degree=4,
)

# Wrap optimizer
optimizer = smp.DistributedOptimizer(optimizer)

# Training step
@smp.step
def train_step(batch):
    outputs = model(batch)
    loss = outputs.loss
    model.backward(loss)
    return loss
```

**What happens under the hood:**

1. **Model Sharding (FSDP):**
   - Each Transformer layer is wrapped as FSDP unit
   - Parameters sharded across `hybrid_shard_degree=4` GPUs
   - Before forward: All-gather parameters
   - After backward: Reduce-scatter gradients

2. **Tensor Parallelism:**
   - Attention/FFN layers split across `tensor_parallel_degree=2` GPUs
   - Column-parallel linear: Each GPU computes half of output
   - Row-parallel linear: All-reduce to combine outputs

3. **Activation Offloading:**
   - Forward: Offload activations to CPU
   - Backward: Prefetch activations from CPU (horizon=2 layers ahead)

**GPU allocation with TP=2, HSD=4, 8 GPUs:**
```
GPU 0-1: TP group 0 (share tensors), FSDP rank 0-1
GPU 2-3: TP group 1 (share tensors), FSDP rank 2-3
GPU 4-5: TP group 2 (share tensors), FSDP rank 4-5
GPU 6-7: TP group 3 (share tensors), FSDP rank 6-7

FSDP shards model across 4 groups (HSD=4)
Within each group, model tensors split across 2 GPUs (TP=2)
```

#### **Framework Version**
```python
py_version="py311"
framework_version="2.4.1"
```

**What it does:**
- SageMaker uses pre-built Docker image: `763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.4.1-gpu-py311`
- Image includes:
  - PyTorch 2.4.1
  - CUDA 12.1
  - SageMaker Model Parallel v2
  - Python 3.11

**Custom image alternative:**
```python
image_uri="123456.dkr.ecr.us-west-2.amazonaws.com/my-custom-pytorch:latest"
# Don't set framework_version if using custom image
```

#### **Output Path**
```python
output_path="s3://bucket/smp-fsdp/llama_v3-outputdir/"
```

**What it does:**
- Container saves final model artifacts to `/opt/ml/model/`
- SageMaker uploads `/opt/ml/model/` → `output_path` after training completes

**What's in output:**
```
/opt/ml/model/
├── model.safetensors    (or pytorch_model.bin)
├── config.json
├── tokenizer.json
├── tokenizer_config.json
└── ...

↓ Uploaded to ↓

s3://bucket/smp-fsdp/llama_v3-outputdir/{job-name}/output/model.tar.gz
```

**Note:**
- This is the FINAL model, not checkpoints
- Checkpoints go to `checkpoint_s3_uri` or FSx
- Model is packaged as tar.gz for deployment

#### **FSx Networking Parameters**
```python
kwargs = {
    "security_group_ids": ["sg-abc123..."],
    "subnets": ["subnet-xyz789..."],
}
```

**Why needed for FSx:**
- Training instances must be in same VPC/subnet as FSx
- Security group must allow:
  - Inbound: Port 988 (Lustre), 1021-1023
  - Outbound: All traffic (for S3, ECR, internet)

**Without FSx:**
- These parameters not needed
- SageMaker uses default VPC with internet access

---

## SECTION 6: Launch Training Job

```python
smp_estimator.fit(inputs=data_channels)
```

**ANNOTATION - What Happens When You Call fit():**

### Step 1: Job Submission (Client-Side)
1. **Upload source code:**
   ```
   Tar: ../shared-scripts/ → /tmp/source.tar.gz
   Upload: /tmp/source.tar.gz → s3://bucket/.../source/sourcedir.tar.gz
   ```

2. **Create training job:**
   ```python
   boto3.client("sagemaker").create_training_job(
       TrainingJobName="smp-8b-p4d24x-hs4-aoTrue-bs2-2024-11-23-01-23-45",
       RoleArn=role,
       AlgorithmSpecification={
           "TrainingImage": "763104351884.dkr.ecr...pytorch-training:2.4.1-gpu-py311",
           "TrainingInputMode": "File",
       },
       InputDataConfig=[
           {
               "ChannelName": "train",
               "DataSource": {
                   "S3DataSource": {
                       "S3Uri": "s3://.../dataset/train/",
                       "S3DataDistributionType": "FullyReplicated",
                   }
               },
           },
           # ... test channel ...
       ],
       OutputDataConfig={"S3OutputPath": "s3://.../outputdir/"},
       ResourceConfig={
           "InstanceType": "ml.p4d.24xlarge",
           "InstanceCount": 1,
           "VolumeSizeInGB": 400,
       },
       HyperParameters=hyperparameters,
       # ... distribution config, VPC config, etc ...
   )
   ```

3. **Wait and stream logs:**
   - `fit()` blocks and streams CloudWatch logs to notebook
   - Can Ctrl+C to detach (job continues running)

### Step 2: Instance Provisioning (SageMaker-Side)
1. **Launch EC2 instances:** 1x ml.p4d.24xlarge
2. **Attach 400 GB EBS volume** to `/opt/ml/`
3. **Pull Docker image:** pytorch-training:2.4.1-gpu-py311
4. **Mount FSx** (if configured): FSx mount → `/opt/ml/input/data/train/`

### Step 3: Data Download
1. **Download training data** (if S3):
   ```
   S3: s3://.../dataset/train/training.json
    ↓
   Container: /opt/ml/input/data/train/training.json
   ```

2. **Download source code:**
   ```
   S3: s3://.../source/sourcedir.tar.gz
    ↓
   Container: /opt/ml/code/ (extracted)
   ```

3. **Download checkpoints** (if resuming):
   ```
   S3: s3://.../checkpoints/step_100/
    ↓
   Container: /opt/ml/checkpoints/step_100/
   ```

### Step 4: Container Startup
**Container entrypoint runs:**
```bash
torchrun \
  --nproc_per_node=8 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=algo-1 \
  --master_port=7777 \
  /opt/ml/code/train.py \
    --train_batch_size 2 \
    --lr 0.0001 \
    --tensor_parallel_degree 2 \
    --hybrid_shard_degree 4 \
    --max_steps 50 \
    ... (all hyperparameters)
```

**8 processes spawn** (one per GPU):
```
RANK=0, LOCAL_RANK=0, CUDA_VISIBLE_DEVICES=0 → train.py
RANK=1, LOCAL_RANK=1, CUDA_VISIBLE_DEVICES=1 → train.py
...
RANK=7, LOCAL_RANK=7, CUDA_VISIBLE_DEVICES=7 → train.py
```

### Step 5: Training Script Execution

**Inside train.py (simplified flow):**

```python
import sagemaker_model_parallel.torch as smp
import torch
import torch.distributed as dist

# 1. Initialize distributed training
smp.init()  # Reads SMP_* environment variables
dist.init_process_group(backend="nccl")

# 2. Load data
train_dataset = load_dataset("/opt/ml/input/data/train/training.json")
train_dataloader = DataLoader(train_dataset, batch_size=2)

# 3. Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",  # or FSx path
    torch_dtype=torch.bfloat16,
)

# 4. Wrap model for distributed training
model = smp.DistributedModel(
    model,
    tensor_parallel_degree=2,
    hybrid_shard_degree=4,
    activation_checkpointing=True,
    offload_activations=True,
)

# 5. Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
optimizer = smp.DistributedOptimizer(optimizer)

# 6. Training loop
@smp.step
def train_step(batch):
    outputs = model(**batch)
    loss = outputs.loss
    model.backward(loss)
    return loss

for step, batch in enumerate(train_dataloader):
    optimizer.zero_grad()
    loss = train_step(batch)
    optimizer.step()

    # Save checkpoint every 50 steps
    if step % 50 == 0:
        smp.save_checkpoint("/opt/ml/checkpoints/", step=step)

    if step >= 50:
        break

# 7. Save final model (rank 0 only)
if smp.rank() == 0:
    model.save_pretrained("/opt/ml/model/")
```

**What SageMaker Model Parallel does:**
- **FSDP:** Shards model parameters across HSD=4 GPUs
  - All-gather params before forward/backward
  - Reduce-scatter gradients after backward
- **TP:** Splits tensors across TP=2 GPUs
  - Column-parallel: Split output dimension
  - Row-parallel: All-reduce to combine
- **Activation offloading:**
  - Offload to CPU during forward
  - Prefetch to GPU before backward (horizon=2)

### Step 6: Checkpointing (During Training)
Every 50 steps:
```python
smp.save_checkpoint("/opt/ml/checkpoints/", step=50)
```

**With S3:**
- Writes to `/opt/ml/checkpoints/step_50/`
- SageMaker syncs to S3 every ~5 min

**With FSx:**
- Writes directly to FSx mount
- Immediately visible to all nodes

**Checkpoint contents:**
```
/opt/ml/checkpoints/step_50/
├── model_0.pt        # Shard for FSDP rank 0
├── model_1.pt        # Shard for FSDP rank 1
├── model_2.pt        # Shard for FSDP rank 2
├── model_3.pt        # Shard for FSDP rank 3
├── optimizer_0.pt    # Optimizer state shard 0
├── optimizer_1.pt
├── optimizer_2.pt
├── optimizer_3.pt
└── metadata.json     # Step count, RNG state, etc.
```

### Step 7: Training Completion
1. **Save final model:**
   ```
   Container: /opt/ml/model/
   ├── pytorch_model.bin  (consolidated, not sharded)
   ├── config.json
   └── tokenizer files
   ```

2. **Upload model to S3:**
   ```
   Tar: /opt/ml/model/ → model.tar.gz
   Upload: model.tar.gz → s3://.../outputdir/{job-name}/output/model.tar.gz
   ```

3. **Final checkpoint sync** (if S3 checkpointing)

4. **Job status:** `Completed`

5. **Instance termination**

---

## SECTION 7: Container Filesystem Layout

**Complete view of `/opt/ml/` during training:**

```
/opt/ml/
├── input/
│   ├── data/
│   │   ├── train/
│   │   │   └── training.json          # From S3 or FSx
│   │   └── test/
│   │       └── validation.json        # From S3 or FSx
│   └── config/
│       ├── hyperparameters.json       # All hyperparameters
│       └── resourceconfig.json        # Instance info
├── code/
│   ├── train.py                       # Entry point
│   ├── arguments.py
│   ├── fsdp_utils.py
│   └── ... (all source files)
├── checkpoints/                       # Periodic saves
│   ├── step_50/
│   │   ├── model_0.pt
│   │   └── ...
│   └── step_100/
│       └── ...
├── model/                             # Final output
│   ├── pytorch_model.bin
│   └── config.json
└── output/
    └── failure                        # Only if job fails
```

**File ownership:**
- SageMaker downloads: input/data/, input/config/
- Your script writes: checkpoints/, model/, output/

**Mounted volumes:**
- `/opt/ml/`: 400 GB EBS volume
- `/opt/ml/input/data/train/` (if FSx): FSx Lustre mount

---

## SECTION 8: Data Flow Summary

### Training Data Flow
```
┌─────────────────────────────────────────────────────────────┐
│ CLIENT SIDE (Notebook)                                      │
├─────────────────────────────────────────────────────────────┤
│ 1. Load dataset: HuggingFace → local memory                │
│ 2. Tokenize: Text → Token IDs (client-side CPU)            │
│ 3. Save: JSON format → local disk                          │
│ 4. Upload: Local → S3 bucket                               │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ S3 STORAGE                                                  │
├─────────────────────────────────────────────────────────────┤
│ s3://sagemaker-{region}-{account}/dataset/                 │
│ ├── train/training.json                                    │
│ └── validation/validation.json                             │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ SAGEMAKER SERVICE (Job Start)                              │
├─────────────────────────────────────────────────────────────┤
│ 1. Provision ml.p4d.24xlarge instances                     │
│ 2. Attach 400 GB EBS volume                                │
│ 3. Pull PyTorch training image                             │
│ 4. Download S3 data → /opt/ml/input/data/                  │
│ 5. Download source code → /opt/ml/code/                    │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ TRAINING CONTAINER (8 GPU processes)                       │
├─────────────────────────────────────────────────────────────┤
│ Rank 0: Load /opt/ml/input/data/train/training.json       │
│ Rank 1: Load /opt/ml/input/data/train/training.json       │
│ ... (all ranks load same file)                             │
│                                                             │
│ DataLoader distributes batches across ranks:               │
│ - Rank 0: Batches 0, 8, 16, ...                           │
│ - Rank 1: Batches 1, 9, 17, ...                           │
│ - ...                                                       │
└─────────────────────────────────────────────────────────────┘
```

### Model Checkpoint Flow
```
┌─────────────────────────────────────────────────────────────┐
│ TRAINING CONTAINER (During Training)                       │
├─────────────────────────────────────────────────────────────┤
│ Step 50:                                                    │
│   smp.save_checkpoint("/opt/ml/checkpoints/", step=50)     │
│   → Writes sharded checkpoints:                            │
│     /opt/ml/checkpoints/step_50/model_{0-3}.pt            │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ CHECKPOINT BACKEND (S3 or FSx)                             │
├─────────────────────────────────────────────────────────────┤
│ S3 (checkpoint_s3_uri):                                     │
│   - SageMaker syncs /opt/ml/checkpoints/ → S3 every 5 min │
│   - Final sync on job completion                           │
│                                                             │
│ FSx (checkpoint_local_path):                                │
│   - /opt/ml/checkpoints/ mounted to FSx                    │
│   - Writes immediately visible on FSx                       │
└─────────────────────────────────────────────────────────────┘
```

### Final Model Flow
```
┌─────────────────────────────────────────────────────────────┐
│ TRAINING CONTAINER (Job End)                               │
├─────────────────────────────────────────────────────────────┤
│ Rank 0 only:                                                │
│   model.save_pretrained("/opt/ml/model/")                  │
│   → Writes consolidated model (not sharded):               │
│     /opt/ml/model/pytorch_model.bin                        │
│     /opt/ml/model/config.json                              │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ SAGEMAKER SERVICE (Job Completion)                         │
├─────────────────────────────────────────────────────────────┤
│ 1. Tar /opt/ml/model/ → model.tar.gz                       │
│ 2. Upload → s3://{output_path}/{job-name}/output/          │
│ 3. Terminate instances                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Takeaways

### 1. **Distributed Training Configuration is Hierarchical**
```
distribution = {
    torch_distributed: Enable PyTorch DDP backend (torchrun)
        ↓
    smdistributed.modelparallel: Enable SageMaker Model Parallel
        ↓
        tensor_parallel_degree: Split model tensors across GPUs
        hybrid_shard_degree: FSDP sharding level
        sm_activation_offloading: Offload activations to CPU
}
```

### 2. **Memory Optimization Techniques Stack**
- **BFloat16:** 50% memory reduction
- **FSDP (HSD=4):** 4x memory reduction for parameters/gradients
- **Tensor Parallelism (TP=2):** 2x memory reduction for activations
- **Activation Checkpointing:** 30-40% memory reduction
- **Activation Offloading:** Additional savings (with speed trade-off)

**Combined:** Can fit 70B model on 8x 40GB GPUs

### 3. **Two Deployment Patterns**
- **S3:** Easier setup, slower checkpointing, good for small models
- **FSx:** Faster checkpointing, better multi-node performance, recommended for 70B+

### 4. **Critical Paths**
- **Training data:** S3 or FSx → `/opt/ml/input/data/train/`
- **Checkpoints:** `/opt/ml/checkpoints/` → S3 or FSx
- **Final model:** `/opt/ml/model/` → `s3://{output_path}/`
- **Source code:** `source_dir/` → S3 → `/opt/ml/code/`

### 5. **Instance Selection Guide**
- **8B model:** 1x p4d.24xlarge (8 GPUs)
- **70B model:** 8x p4d.24xlarge (64 GPUs) or 2x p5.48xlarge (16 H100s)
- **FP8 training:** Requires H100 (p5 instances)

---

## Next Steps

1. **Review the training script:** See `../shared-scripts/train.py` for how these configurations are used
2. **Experiment with parameters:**
   - Increase `train_batch_size` if memory permits
   - Adjust `tensor_parallel_degree` for different model sizes
   - Try `fp8=1` on p5 instances (see annotated FP8 example)
3. **Scale to 70B:**
   - Change `model_config = "70b"`
   - Increase `instance_count=8`
   - Use FSx for checkpointing

---

**End of Annotated Example 1**

*This annotation explained every SDK call and distributed training configuration. See other annotated examples for fine-tuning variants, serving patterns, and production deployments.*
