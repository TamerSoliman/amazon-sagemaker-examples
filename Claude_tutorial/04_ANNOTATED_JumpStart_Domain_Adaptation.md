# ANNOTATED EXAMPLE 4: JumpStart Domain Adaptation Fine-tuning

**Source:** `generative_ai/sm-jumpstart_foundation_llama_3_8b_domain_adaption_finetuning.ipynb`

**Purpose:** This annotation explains the SageMaker JumpStart SDK pattern for fine-tuning foundation models, showing how it abstracts away distributed training complexity compared to the low-level SMP v2 approach in Examples 1-3.

---

## Overview: JumpStart vs Low-Level Training

**What is SageMaker JumpStart?**
- **Pre-configured foundation models** with optimal training settings
- **High-level SDK** (`JumpStartEstimator`) that hides complexity
- **No distributed training config** - automatically determined by model
- **Quick start** - 10 lines of code vs 100+ for manual setup

**Comparison:**

| Aspect | JumpStart (This Example) | SMP v2 (Examples 1-3) |
|--------|--------------------------|----------------------|
| **SDK** | `JumpStartEstimator` | `PyTorch` estimator |
| **Distribution Config** | Automatic (hidden) | Manual `distribution={}` |
| **Model Selection** | `model_id="meta-textgeneration-llama-3-8b"` | Manual model loading |
| **Hyperparameters** | Simple: `epoch`, `batch_size` | Complex: `tensor_parallel_degree`, `hybrid_shard_degree` |
| **Instance Selection** | Auto-recommended | Manual |
| **Use Case** | Quick fine-tuning, standard workflows | Custom distributed training, research |
| **Flexibility** | Lower (pre-configured) | Higher (full control) |
| **Code Complexity** | 10-20 lines | 100+ lines |

---

## SECTION 1: Model Selection with JumpStart

```python
(model_id, model_version) = (
    "meta-textgeneration-llama-3-8b",
    "2.*",  # Latest version 2.x
)
```

### ANNOTATION: model_id System

**What is model_id:**
- Unique identifier for pre-configured models in JumpStart
- Format: `{vendor}-{task}-{model-name}-{variant}`
- Examples:
  - `meta-textgeneration-llama-3-8b`
  - `meta-textgeneration-llama-3-70b`
  - `mistralai-mistral-7b-instruct`
  - `huggingface-llm-falcon-40b`

**What happens under the hood:**
```python
# When you specify model_id, JumpStart:
1. Looks up model config from JumpStart registry
2. Determines optimal:
   - Docker image (with correct PyTorch/CUDA versions)
   - Instance type (based on model size)
   - Distributed training strategy (FSDP, TP, etc.)
   - Default hyperparameters (learning rate, batch size, etc.)
   - Data format requirements
3. Configures training job automatically
```

**Model Registry Lookup:**
```python
# Behind the scenes (you don't write this):
from sagemaker.jumpstart import metadata

model_metadata = metadata.get_model_specs(model_id, model_version)

# Returns:
{
    "default_inference_instance_type": "ml.g5.2xlarge",
    "default_training_instance_type": "ml.g5.12xlarge",
    "supported_training_instance_types": ["ml.g5.12xlarge", "ml.g5.24xlarge", "ml.g5.48xlarge"],
    "training_script": "s3://jumpstart-cache-.../llama3-training.tar.gz",
    "default_hyperparameters": {
        "epoch": "5",
        "per_device_train_batch_size": "4",
        "learning_rate": "1e-4",
        "lora_r": "8",
        "lora_alpha": "32",
    },
    "distribution_config": {
        # Pre-configured FSDP settings for Llama 3 8B
        "smdistributed": {
            "dataparallel": {"enabled": True}
        }
    }
}
```

**Key insight:** You don't configure distributed training - JumpStart does it for you!

---

## SECTION 2: Data Preparation for Domain Adaptation

### Domain Adaptation vs Instruction Tuning

**Domain Adaptation (This Example):**
```
Input Format: Raw text file (.txt)
Content: Domain-specific corpus (e.g., SEC filings, medical texts, legal documents)
Goal: Model learns domain-specific vocabulary and patterns
Method: Continues pre-training on domain data

Example:
"Note About Forward-Looking Statements
This Annual Report on Form 10-K includes forward-looking statements...
Actual results could differ materially for a variety of reasons..."
```

**Instruction Tuning (Alternative):**
```
Input Format: JSONL file with instruction-response pairs
Content: Question-answer or task-completion examples
Goal: Model learns to follow instructions
Method: Fine-tunes on supervised pairs

Example:
{"instruction": "Summarize this SEC filing", "context": "...", "response": "..."}
```

### Data Preparation Code

```python
# Download SEC filings
from sec_edgar_downloader import Downloader

dl = Downloader("Amazon", "companyinfo@amazon.com")
dl.get("10-K", "AMZN", limit=2)  # Get 2 most recent 10-K reports
```

**ANNOTATION:**
- Downloads SEC Form 10-K (annual reports) from EDGAR database
- Public data from sec.gov
- License: CC BY-SA 4.0 (Creative Commons)

**Why SEC filings for domain adaptation:**
- Dense, technical financial/business language
- Different from general web text LLMs are pre-trained on
- Tests model's ability to adapt to specialized domains

### Parsing SEC Filings

```python
from smjsindustry.finance.processor import SECXMLFilingParser

parser = SECXMLFilingParser(
    role=sagemaker.get_execution_role(),
    instance_count=1,
    instance_type="ml.c5.2xlarge",  # Processing instance
    sagemaker_session=sagemaker.Session(),
)

parser.parse(
    "rawfiles",  # Input directory with raw SEC XML files
    "s3://{}/{}/{}/output".format(bucket, sec_processed_folder)
)
```

**ANNOTATION - SECXMLFilingParser:**

**What it does:**
- Launches a SageMaker Processing job (separate from training)
- Extracts text from SEC XML filings
- Cleans and formats text (removes HTML, tables, boilerplate)
- Outputs clean .txt files to S3

**Why a separate processing job:**
- SEC XML is complex (XBRL format)
- Parsing is CPU-intensive
- Separates data prep from training

**Container execution:**
```
1. SageMaker launches ml.c5.2xlarge instance
2. Downloads smjsindustry parser container
3. Mounts rawfiles/ from local → /opt/ml/processing/input/
4. Runs parser: XML → clean text
5. Writes to /opt/ml/processing/output/ → S3
6. Terminates instance
```

**Output format:**
```
combined.txt (concatenated clean text):

Note About Forward-Looking Statements
This Annual Report on Form 10-K includes forward-looking statements...

GENERAL
Embracing Our Future
...

RISK FACTORS
...
```

---

## SECTION 3: JumpStartEstimator - The High-Level API

```python
from sagemaker.jumpstart.estimator import JumpStartEstimator

estimator = JumpStartEstimator(
    model_id=model_id,  # "meta-textgeneration-llama-3-8b"
    environment={"accept_eula": "true"},  # Llama license agreement
    instance_type="ml.g5.12xlarge",       # 4x A10G GPUs
    hyperparameters={
        "epoch": "5",
        "per_device_train_batch_size": "4",
    },
)

estimator.fit({"training": domain_training_data_location})
```

### DEEP ANNOTATION: JumpStartEstimator vs PyTorch Estimator

**Differences from Examples 1-3:**

**Low-Level (SMP v2):**
```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point="train.py",  # YOUR training script
    source_dir="./scripts",   # YOUR code
    hyperparameters={
        # YOU configure distributed training
        "tensor_parallel_degree": 2,
        "hybrid_shard_degree": 4,
        "bf16": 1,
        "max_steps": 100,
        # ... 20+ hyperparameters
    },
    distribution={
        # YOU configure distribution strategy
        "torch_distributed": {"enabled": True},
        "smdistributed": {
            "modelparallel": {
                "enabled": True,
                "parameters": {...}
            }
        }
    },
    instance_type="ml.p4d.24xlarge",  # YOU choose
    framework_version="2.4.1",          # YOU specify
    py_version="py311",
)
```

**High-Level (JumpStart):**
```python
from sagemaker.jumpstart.estimator import JumpStartEstimator

estimator = JumpStartEstimator(
    model_id="meta-textgeneration-llama-3-8b",  # JumpStart picks everything
    instance_type="ml.g5.12xlarge",              # Optional override
    hyperparameters={
        "epoch": "5",                            # Simple hyperparameters only
        "per_device_train_batch_size": "4",
    },
)
```

**What JumpStart does automatically:**

1. **Selects training script:**
   ```python
   # JumpStart downloads pre-built script from:
   s3://jumpstart-cache-prod-{region}/meta-llama/meta-textgeneration-llama-3-8b/scripts/training/v2.0.0/train.py

   # This script includes:
   - Model loading (HuggingFace Transformers)
   - LoRA/QLoRA implementation
   - Data loading and tokenization
   - Distributed training setup (FSDP)
   - Checkpoint saving
   - Metrics logging
   ```

2. **Configures distributed training:**
   ```python
   # For Llama 3 8B on ml.g5.12xlarge (4 GPUs), JumpStart sets:
   distribution = {
       "torch_distributed": {"enabled": True},
       "smdistributed": {
           "dataparallel": {  # Uses SMDDP (Data Parallel)
               "enabled": True,
               "custom_mpi_options": "-verbose -x NCCL_DEBUG=VERSION"
           }
       }
   }

   # NO tensor parallelism or FSDP config needed - it's pre-optimized!
   ```

3. **Sets default hyperparameters:**
   ```python
   # JumpStart defaults (you can override):
   default_hyperparameters = {
       "epoch": "5",
       "per_device_train_batch_size": "4",
       "learning_rate": "1e-4",
       "lora_r": "8",           # LoRA rank
       "lora_alpha": "32",      # LoRA scaling
       "lora_dropout": "0.05",
       "gradient_accumulation_steps": "1",
       "warmup_steps": "0",
       "max_seq_length": "2048",
       "bf16": "true",          # Automatic BF16
       "gradient_checkpointing": "true",  # Memory optimization
   }
   ```

4. **Picks Docker image:**
   ```python
   # JumpStart uses pre-built image:
   image_uri = "763104351884.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-training:2.3.0-transformers4.44-gpu-py311-cu121-ubuntu20.04"

   # Includes:
   - PyTorch 2.3
   - Transformers 4.44
   - PEFT library (for LoRA)
   - SageMaker Distributed libraries
   - All dependencies pre-installed
   ```

### Hyperparameters Annotation

```python
hyperparameters={
    "epoch": "5",
    "per_device_train_batch_size": "4",
}
```

**epoch=5:**
- Train for 5 full passes over the dataset
- For domain adaptation, 3-10 epochs typical
- More epochs = better domain specialization (but risk overfitting)

**per_device_train_batch_size=4:**
- Batch size **per GPU** (not total)
- ml.g5.12xlarge has 4 GPUs
- Effective batch size = 4 GPUs × 4 samples = 16
- With gradient accumulation=2: Effective = 32

**Other available hyperparameters:**
```python
hyperparameters={
    # Training config
    "epoch": "5",
    "learning_rate": "1e-4",
    "per_device_train_batch_size": "4",
    "gradient_accumulation_steps": "1",
    "warmup_steps": "0",

    # LoRA config (Parameter-Efficient Fine-Tuning)
    "lora_r": "8",           # Rank of LoRA matrices (higher = more params)
    "lora_alpha": "32",      # Scaling factor
    "lora_dropout": "0.05",  # Regularization

    # Optimization
    "bf16": "true",          # BFloat16 precision
    "gradient_checkpointing": "true",  # Save memory

    # Data
    "max_seq_length": "2048",  # Max tokens per sample
    "validation_split_ratio": "0.1",  # Hold out 10% for validation
}
```

**Default LoRA configuration:**
- JumpStart uses **LoRA (Low-Rank Adaptation)** by default
- Only trains ~0.5% of model parameters (small adapter layers)
- Much faster and cheaper than full fine-tuning
- 8B model: Full fine-tuning = 8B params, LoRA = ~40M params

---

## SECTION 4: Training Execution

```python
estimator.fit({"training": domain_training_data_location})
```

**ANNOTATION - What happens during fit():**

### 1. Data Download
```
S3: s3://bucket/amazon_sec_filing_data/output/combined.txt
  ↓
Container: /opt/ml/input/data/training/combined.txt
```

### 2. Training Script Execution
```python
# JumpStart's train.py (simplified):

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Add LoRA adapters
lora_config = LoraConfig(
    r=8,                      # From hyperparameters
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# Only ~0.5% of params are trainable
model.print_trainable_parameters()
# Output: trainable params: 40,894,464 || all params: 8,030,261,248 || trainable%: 0.509

# Load training data
dataset = load_dataset("text", data_files="/opt/ml/input/data/training/combined.txt")

# Tokenize
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")
tokenized = dataset.map(lambda x: tokenizer(x["text"], max_length=2048, truncation=True))

# Train with Hugging Face Trainer
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="/opt/ml/model",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    learning_rate=1e-4,
    bf16=True,
    gradient_checkpointing=True,
    dataloader_num_workers=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
)

trainer.train()

# Save LoRA adapters only (not full model)
model.save_pretrained("/opt/ml/model/")
```

### 3. Distributed Training Setup
```
ml.g5.12xlarge: 4× NVIDIA A10G GPUs

Process layout:
- Rank 0 (GPU 0): Loads full model + LoRA adapters
- Rank 1 (GPU 1): Loads full model + LoRA adapters
- Rank 2 (GPU 2): Loads full model + LoRA adapters
- Rank 3 (GPU 3): Loads full model + LoRA adapters

Data Parallelism:
- Each GPU processes different batch
- Gradients synchronized across GPUs (All-Reduce)
- Only LoRA params updated (40M params, not 8B)
```

**Memory per GPU:**
```
Base model (frozen): 8B × 2 bytes (BF16) = 16 GB
LoRA adapters: 40M × 2 bytes = 80 MB
Optimizer (AdamW): 40M × 8 bytes (momentum + variance) = 320 MB
Activations: ~10 GB (with gradient checkpointing)
Total per GPU: ~26 GB (fits in A10G 24GB with some optimization)
```

### 4. Output Artifacts
```
/opt/ml/model/
├── adapter_config.json    # LoRA configuration
├── adapter_model.bin      # Trained LoRA weights (40MB)
├── tokenizer_config.json
├── tokenizer.json
└── special_tokens_map.json

↓ Packaged to ↓

S3: s3://sagemaker-{region}-{account}/meta-textgeneration-llama-3-8b-{timestamp}/output/model.tar.gz
```

**Key difference from Examples 1-3:**
- Only LoRA adapters saved (40MB)
- Base model NOT saved (assumed available from HuggingFace)
- Much smaller checkpoint size

---

## SECTION 5: Deployment with JumpStart

```python
domain_fine_tuned_predictor = estimator.deploy()
```

**ANNOTATION - One-Line Deployment:**

**What this does:**
```python
# Behind the scenes:
1. Creates SageMaker Model:
   - Model data: s3://.../model.tar.gz (LoRA adapters)
   - Base model: meta-llama/Llama-3-8B (downloaded from HuggingFace)
   - Inference image: jumpstart-inference-pytorch-...

2. Creates Endpoint Configuration:
   - Instance type: ml.g5.2xlarge (auto-selected for 8B)
   - Instance count: 1
   - Container: Optimized inference container with vLLM

3. Creates Endpoint:
   - Downloads LoRA adapters
   - Downloads base Llama 3 8B
   - Merges adapters into base model (in memory)
   - Loads model with vLLM for fast inference
   - Ready for requests
```

**vs Manual Deployment (Examples 1-3):**
```python
# Manual approach requires:
from sagemaker.model import Model
from sagemaker.predictor import Predictor

model = Model(
    image_uri="...",  # YOU specify inference image
    model_data="...", # YOU specify model artifacts
    role=role,
    predictor_cls=Predictor,
)

predictor = model.deploy(
    instance_type="ml.g5.2xlarge",  # YOU choose
    initial_instance_count=1,
    serializer=...,  # YOU configure
    deserializer=...,
)
```

**JumpStart approach:**
```python
# One line!
predictor = estimator.deploy()

# JumpStart automatically:
# - Picks instance type (ml.g5.2xlarge for 8B)
# - Uses optimized inference container
# - Configures serializers/deserializers
# - Sets up vLLM/TGI for fast inference
```

---

## SECTION 6: Inference

```python
parameters = {
    "max_new_tokens": 300,
    "top_k": 50,
    "top_p": 0.8,
    "do_sample": True,
    "temperature": 0,
}

payload = {
    "inputs": "Risk factors highlighted in this 10-K report",
    "parameters": parameters,
}

response = domain_fine_tuned_predictor.predict(payload)
```

**ANNOTATION - Inference Parameters:**

**max_new_tokens=300:**
- Generate up to 300 new tokens
- Stops earlier if EOS token generated
- Longer = more comprehensive but slower

**top_k=50:**
- Sample from top 50 most likely next tokens
- Prevents sampling very unlikely tokens
- Lower = more focused, higher = more diverse

**top_p=0.8 (nucleus sampling):**
- Sample from smallest set of tokens with cumulative probability ≥ 0.8
- Dynamic cutoff (vs fixed top_k)
- Standard for modern LLMs

**temperature=0:**
- Greedy decoding (pick most likely token)
- Deterministic output
- temperature > 0: More random/creative

**do_sample=True:**
- Use sampling (vs greedy)
- Required when temperature > 0 or using top_k/top_p

**Expected output:**
```
Risk factors highlighted in this 10-K report include:
- Fluctuations in foreign exchange rates
- Changes in global economic conditions and customer demand
- Inflation and interest rates
- Regional labor market constraints
- Competition in cloud services and online commerce
... (model generates ~300 tokens)
```

**Why domain adaptation helps:**
- Model now uses SEC filing language/structure
- Better at financial terminology
- Understands 10-K report format
- Can generate domain-specific content

---

## SECTION 7: Cost Analysis

### Training Cost

**ml.g5.12xlarge:** $7.09/hour (4× A10G GPUs, 24GB each)

**Training time estimation:**
```
Dataset: ~100,000 tokens (combined SEC filings)
Batch size: 4 per GPU × 4 GPUs = 16
Sequence length: 2048 tokens
Epochs: 5

Samples per epoch: 100,000 / 2048 ≈ 49 samples
Steps per epoch: 49 / 16 ≈ 4 steps
Total steps: 4 × 5 epochs = 20 steps

Training time: ~30 minutes
Cost: $7.09 × 0.5 hours = $3.55
```

**vs SMP v2 (Example 1) for same dataset:**
```
ml.p4d.24xlarge: $32.77/hour (8× A100 40GB)
Training time: ~20 minutes (faster GPUs)
Cost: $32.77 × 0.33 hours = $10.81

JumpStart is 67% cheaper for small-scale fine-tuning!
```

### Inference Cost

**ml.g5.2xlarge:** $1.41/hour (1× A10G GPU, 24GB)

**Throughput:**
```
Tokens/sec: ~50 (with vLLM optimization)
Requests/hour: ~3,600 (assuming 50 tokens/request)
Cost per 1M tokens: $1.41 / (50 × 3600) = $0.008
```

**When to use JumpStart inference:**
- Low-moderate traffic (< 10 req/sec)
- Prototyping and development
- Cost-sensitive applications

**When to use optimized serving (Examples 6-7):**
- High traffic (> 50 req/sec)
- Need low latency (< 100ms)
- Batch processing

---

## SECTION 8: JumpStart vs Custom Training Decision Matrix

| Use Case | JumpStart | Custom (SMP v2) |
|----------|-----------|-----------------|
| **Quick fine-tuning on standard dataset** | ✓ Perfect | ✗ Overkill |
| **Small models (< 13B params)** | ✓ Cost-effective | ⚠️ More expensive |
| **Domain adaptation** | ✓ LoRA built-in | ⚠️ Need to implement LoRA |
| **Standard hyperparameters** | ✓ Pre-optimized | ⚠️ Must tune manually |
| **Large models (70B+ params)** | ⚠️ Limited control | ✓ Full control |
| **Custom distributed strategy** | ✗ Cannot customize | ✓ Configure everything |
| **Research/experiments** | ⚠️ Black box | ✓ Full transparency |
| **Multi-node training (8+ instances)** | ⚠️ Limited | ✓ Optimized |
| **FP8 training** | ✗ Not supported | ✓ Full support |
| **Custom model architecture** | ✗ Only pre-registered models | ✓ Any model |
| **Production deployment** | ✓ One-line deploy | ⚠️ Manual setup |

**Recommendation:**
- **Start with JumpStart** for prototyping
- **Move to custom** when hitting limitations (model size, custom logic, research needs)

---

## SECTION 9: Key Takeaways

### 1. **JumpStart Abstracts Complexity**
```python
# 3 lines vs 100+ lines
estimator = JumpStartEstimator(model_id=model_id)
estimator.fit({"training": data_location})
predictor = estimator.deploy()
```

### 2. **Automatic Distributed Training**
- No `distribution={}` config needed
- JumpStart picks optimal strategy based on model_id
- Uses LoRA by default (0.5% of params trained)

### 3. **Domain Adaptation Pattern**
- Input: Raw text file (.txt)
- Method: Continued pre-training on domain corpus
- Output: Model specialized for domain vocabulary/patterns

### 4. **Cost-Effective for Small-Scale**
- $3-10 for typical fine-tuning job
- Single GPU inference sufficient
- Much cheaper than manual setup for standard cases

### 5. **When NOT to Use JumpStart**
- Custom model architectures
- FP8 training (H100 GPUs)
- Multi-node training (> 8 instances)
- Research requiring full control

---

## Comparison: This Example vs Examples 1-3

| Aspect | Example 4 (JumpStart) | Examples 1-3 (SMP v2) |
|--------|----------------------|----------------------|
| **Estimator** | `JumpStartEstimator` | `PyTorch` |
| **Lines of Code** | ~20 | ~200 |
| **Distribution Config** | Automatic | Manual (50+ lines) |
| **Model Loading** | `model_id` | Manual HF download |
| **Fine-Tuning Method** | LoRA (automatic) | Full model or manual LoRA |
| **Instance Type** | ml.g5.12xlarge (4 GPUs) | ml.p4d.24xlarge (8 GPUs) |
| **Distributed Strategy** | Data Parallel (SMDDP) | FSDP + TP/EP |
| **Training Cost** | $3-5 | $10-30 |
| **Deployment** | 1 line | 20+ lines |
| **Flexibility** | Low (pre-configured) | High (full control) |
| **Best For** | Standard fine-tuning | Custom distributed training |

---

## Next Steps

1. **Try different JumpStart models:**
   ```python
   model_ids = [
       "meta-textgeneration-llama-3-70b",  # Larger model
       "mistralai-mistral-7b-instruct",     # Instruction-tuned
       "cohere-gpt-medium",                 # Alternative vendor
   ]
   ```

2. **Instruction tuning instead of domain adaptation:**
   - Use JSONL format with instruction-response pairs
   - See JumpStart instruction fine-tuning examples

3. **Scale to production:**
   - Add autoscaling: `estimator.deploy(min_capacity=1, max_capacity=5)`
   - Monitor with CloudWatch
   - See Examples 6-7 for optimized serving

4. **When you need more control:**
   - Graduate to Examples 1-3 (SMP v2)
   - Implement custom distributed strategies
   - Use FP8 training on P5 instances

---

**End of Annotated Example 4**

*This annotation explained the SageMaker JumpStart high-level SDK for quick fine-tuning, showing how it abstracts away distributed training complexity. See Examples 1-3 for low-level control and Examples 6-7 for production serving patterns.*
