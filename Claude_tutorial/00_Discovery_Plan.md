# SageMaker Generative AI Deep Dive - Discovery Plan

**Project Goal:** Create deep-dive tutorials and reference guides for Generative AI (Gen AI), Foundation Models (FM), and Agentic AI examples in the amazon-sagemaker-examples repository.

**Total Files Identified:** 136 Gen AI/LLM/FM examples
**Curated Selection:** Top 40 most relevant files for deep analysis

---

## Curation Criteria

Files were selected based on:
1. **Latest Technology:** SageMaker Model Parallelism v2, FP8 training, modern LLMs
2. **Comprehensive Coverage:** Distributed training, fine-tuning, serving, monitoring
3. **Production Readiness:** Real-world patterns, optimization, autoscaling
4. **Educational Value:** Clear SDK usage, well-documented configurations
5. **Relevance:** Focus on LLMs, excluding classical ML and non-Gen AI content

---

## Category 1: Distributed Training with FSDP & Advanced Parallelism (10 files)

### 1. Llama 3.1 FSDP+TP Fine-tuning
**Path:** `build_and_train_models/sm-distributed_model_parallel_v2/llama_v3d1/sm-fsdp-tp_finetuning_llama_v3d1.ipynb`
**Why Selected:** Demonstrates latest SageMaker Model Parallel v2 with Fully Sharded Data Parallel (FSDP) + Tensor Parallelism (TP) for Llama 3.1. Critical for understanding how to scale training across multiple GPUs/nodes.

### 2. Llama 3.1 FSDP+TP with FP8 Training
**Path:** `build_and_train_models/sm-distributed_model_parallel_v2/llama_v3d1/sm-fsdp-tp-fp8_train_llama_v3d1.ipynb`
**Why Selected:** Shows cutting-edge FP8 (8-bit floating point) training configuration, reducing memory footprint and increasing training speed. Essential for large model training on limited GPU memory.

### 3. Llama 2/3 FSDP+TP+Context Parallelism
**Path:** `build_and_train_models/sm-distributed_model_parallel_v2/llama_v2_v3/sm-fsdp-tp-cp_train_llama_v2_v3.ipynb`
**Why Selected:** Introduces Context Parallelism (CP) for handling extremely long sequences. Unique three-way parallelism strategy (FSDP+TP+CP) not commonly documented.

### 4. Llama 2/3 FSDP+TP Fine-tuning
**Path:** `build_and_train_models/sm-distributed_model_parallel_v2/llama_v2_v3/sm-fsdp-tp_finetuning_llama_v2_v3.ipynb`
**Why Selected:** Canonical example of FSDP+TP configuration. Shows core distributed training setup that applies to multiple Llama versions.

### 5. Llama 2/3 FSDP+TP with FP8
**Path:** `build_and_train_models/sm-distributed_model_parallel_v2/llama_v2_v3/sm-fsdp-tp-fp8_train_llama_v2_v3.ipynb`
**Why Selected:** FP8 training variant for Llama 2/3, complementing the Llama 3.1 FP8 example for comparison.

### 6. Mixtral FSDP+Expert Parallelism
**Path:** `build_and_train_models/sm-distributed_model_parallel_v2/mixtral/sm-fsdp-ep_train_mixtral.ipynb`
**Why Selected:** Demonstrates Expert Parallelism (EP) for Mixture-of-Experts (MoE) models like Mixtral. Shows how SMP v2 handles different model architectures beyond dense transformers.

### 7. Mixtral FSDP+EP with FP8
**Path:** `build_and_train_models/sm-distributed_model_parallel_v2/mixtral/sm-fsdp-ep-fp8_train_mixtral.ipynb`
**Why Selected:** Combines EP with FP8 training for MoE models. Unique configuration pattern for memory-efficient MoE training.

### 8. GPT-NeoX FSDP+TP Training
**Path:** `build_and_train_models/sm-distributed_model_parallel_v2/gpt-neox/sm-fsdp-tp_train_gpt-neox.ipynb`
**Why Selected:** Shows FSDP+TP applied to GPT-NeoX architecture, demonstrating portability of distributed training patterns across different model families.

### 9. Llama 2 FSDP+FP8 on P5 Instances
**Path:** `generative_ai/sm-fsdp_training_of_llama_v2_with_fp8_on_p5.ipynb`
**Why Selected:** P5 instance-specific optimizations with NVIDIA H100 GPUs. Shows instance type considerations for distributed training.

### 10. Mixtral Training on P4 Instances
**Path:** `build_and_train_models/sm-distributed_training_model_parallel_v2_mixtral_on_p4.ipynb`
**Why Selected:** P4 instance configuration for Mixtral, showing how to optimize for NVIDIA A100 GPUs vs H100.

---

## Category 2: Fine-Tuning (LoRA, QLoRA, PEFT, Domain Adaptation) (10 files)

### 11. Llama 3.1 405B Fine-tuning
**Path:** `generative_ai/sm-jumpstart_foundation_llama_3_1_405b_finetuning.ipynb`
**Why Selected:** Largest Llama model fine-tuning example. Shows how JumpStart handles extremely large models with distributed fine-tuning under the hood.

### 12. Llama 3 8B Domain Adaptation Fine-tuning
**Path:** `generative_ai/sm-jumpstart_foundation_llama_3_8b_domain_adaption_finetuning.ipynb`
**Why Selected:** Domain adaptation pattern - fine-tuning on domain-specific data. Shows complete workflow from data preparation to model evaluation.

### 13. Llama 3.2 Vision-Language Fine-tuning
**Path:** `generative_ai/sm-jumpstart_foundation_llama_3_2_vision_language_finetuning.ipynb`
**Why Selected:** Multimodal fine-tuning for vision-language models. Emerging capability with Llama 3.2.

### 14. Llama 3.2 3B Fine-tuning
**Path:** `generative_ai/sm-jumpstart_foundation_llama_3_2_3b_finetuning.ipynb`
**Why Selected:** Small model fine-tuning, more accessible for learners. Shows complete fine-tuning lifecycle.

### 15. Llama 3 Fine-tuning (General)
**Path:** `generative_ai/sm-jumpstart_foundation_llama_3_finetuning.ipynb`
**Why Selected:** General Llama 3 fine-tuning guide, serves as baseline reference.

### 16. Mixtral 8x7B Fine-tune and Deploy with QLoRA
**Path:** `generative_ai/sm-mixtral_8x7b_fine_tune_and_deploy/sm-mixtral_8x7b_fine_tune_and_deploy.ipynb`
**Why Selected:** Custom QLoRA implementation for MoE model. Shows full end-to-end workflow: fine-tune → deploy → inference.

### 17. Custom HuggingFace Fine-tuning with Your Own Scripts
**Path:** `generative_ai/sm-finetuning_huggingface_with_your_own_scripts_and_data/sm-finetuning_huggingface_with_your_own_scripts_and_data.ipynb`
**Why Selected:** BYOC (Bring Your Own Code) pattern. Essential for custom training logic beyond JumpStart defaults.

### 18. Mistral 7B Domain Adaptation
**Path:** `generative_ai/sm-jumpstart_foundation_mistral_7b_domain_adaption_finetuning.ipynb`
**Why Selected:** Mistral model fine-tuning, alternative to Llama. Shows domain adaptation on different architecture.

### 19. Gemma Fine-tuning
**Path:** `generative_ai/sm-jumpstart_foundation_gemma_fine_tuning.ipynb`
**Why Selected:** Google's Gemma model fine-tuning. Shows SageMaker's multi-vendor FM support.

### 20. Code Llama Fine-tuning with HumanEval
**Path:** `generative_ai/sm-jumpstart_foundation_code_llama_fine_tuning_human_eval.ipynb`
**Why Selected:** Code generation model fine-tuning with evaluation benchmarks (HumanEval). Shows task-specific fine-tuning and evaluation.

---

## Category 3: LLM Serving & Inference Optimization (10 files)

### 21. DJL DeepSpeed BLOOM 176B Deployment
**Path:** `generative_ai/sm-djl_deepspeed_bloom_176b_deploy.ipynb`
**Why Selected:** Largest model deployment example (176B parameters). Shows DJL Serving with DeepSpeed for multi-GPU inference.

### 22. Text Generation Inference (TGI) Overview
**Path:** `generative_ai/sm-jumpstart_foundation_text_generation_inference.ipynb`
**Why Selected:** Comprehensive TGI introduction. HuggingFace's optimized inference server with features like continuous batching, token streaming.

### 23. Llama 2 70B with LMI
**Path:** `archived/workshops/lab11-llama2/meta-llama-2-70b-lmi.ipynb`
**Why Selected:** Large Model Inference (LMI) container for 70B model. Shows SageMaker's managed inference optimizations.

### 24. Llama 2 70B with TensorRT-LLM
**Path:** `archived/workshops/deploy-V7-lmi/llama2_70b-lmi-trtllm.ipynb`
**Why Selected:** TensorRT-LLM integration for maximum inference performance on NVIDIA GPUs. Shows advanced optimization techniques.

### 25. GPT-NeoX with DJL Accelerate
**Path:** `archived/workshops/lab3-optimize-llm/djl_accelerate_deploy_g5_12x_GPT_NeoX.ipynb`
**Why Selected:** HuggingFace Accelerate for inference. Alternative to DeepSpeed, shows multiple acceleration frameworks.

### 26. Async Inference Walkthrough
**Path:** `deploy_and_monitor/sm-async_inference_walkthrough/sm-async_inference_walkthrough.ipynb`
**Why Selected:** Asynchronous inference pattern for long-running LLM tasks. Essential for batch processing and cost optimization.

### 27. Async Inference with Python SDK
**Path:** `deploy_and_monitor/sm-async_inference_with_python_sdk/sm-async_inference_with_python_sdk.ipynb`
**Why Selected:** SDK-focused async inference guide. Shows programmatic async endpoint management.

### 28. Multi-Model Endpoint BYOC
**Path:** `deploy_and_monitor/sm-multi_model_endpoint_bring_your_own_container/sm-multi_model_endpoint_bring_your_own_container.ipynb`
**Why Selected:** Host multiple models on single endpoint for cost efficiency. Shows custom container integration.

### 29. Llama 2 7B LMI with Autoscaling
**Path:** `archived/workshops/lab-inference-components-with-scaling/2c_meta-llama2-7b-lmi-autoscaling.ipynb`
**Why Selected:** Inference Components with autoscaling. Latest SageMaker feature for flexible capacity management.

### 30. GPT-J 6B with LMI Token Streaming
**Path:** `archived/workshops/lab6-token-streaming-eleutherai-gpt-j-6b-lmi.ipynb`
**Why Selected:** Token streaming implementation. Critical for real-time chat applications.

---

## Category 4: Production Patterns, Monitoring & JumpStart (10 files)

### 31. RAG with LangChain Question Answering
**Path:** `generative_ai/sm-jumpstart_foundation_rag_langchain_question_answering.ipynb`
**Why Selected:** Retrieval-Augmented Generation (RAG) pattern with LangChain integration. Most common production pattern for LLM applications.

### 32. RAG with Cohere and LangChain
**Path:** `generative_ai/sm-jumpstart_rag_question_answering_with_cohere_and_langchain.ipynb`
**Why Selected:** Alternative RAG implementation with Cohere embeddings. Shows multi-vendor integration.

### 33. Text Embedding
**Path:** `generative_ai/sm-jumpstart_text_embedding.ipynb`
**Why Selected:** Embedding models for RAG, semantic search. Foundation for vector databases.

### 34. Custom Text Embedding Processing
**Path:** `generative_ai/sm-text_embedding_custom_processing.ipynb`
**Why Selected:** Custom embedding preprocessing and batching. Shows production-grade embedding workflows.

### 35. LLM Monitoring BYOC
**Path:** `deploy_and_monitor/sm-model_monitor_byoc_llm_monitor/sm-model_monitor_byoc_llm_monitor.ipynb`
**Why Selected:** LLM-specific monitoring for hallucinations, toxicity, bias. Critical for production safety.

### 36. Application Autoscaling for Real-time Endpoints
**Path:** `deploy_and_monitor/sm-app_autoscaling_realtime_endpoints/sm-app_autoscaling_realtime_endpoints.ipynb`
**Why Selected:** Dynamic scaling configuration for cost optimization and performance. Shows target tracking and step scaling.

### 37. Inference Components Autoscaling
**Path:** `deploy_and_monitor/sm-app_autoscaling_realtime_endpoints_inference_components/sm-app_autoscaling_realtime_endpoints_inference_components.ipynb`
**Why Selected:** Latest Inference Components feature with autoscaling. Enables sharing GPU capacity across multiple models.

### 38. Llama Guard Text Moderation
**Path:** `generative_ai/sm-jumpstart_foundation_llama_guard_text_moderation.ipynb`
**Why Selected:** Safety and moderation using Llama Guard. Essential for responsible AI deployment.

### 39. Llama 2 Text Completion
**Path:** `generative_ai/sm-jumpstart_llama_2_text_completion.ipynb`
**Why Selected:** Basic JumpStart deployment and inference. Entry point for beginners.

### 40. Llama 3 Vision-Language Deployment
**Path:** `generative_ai/sm-jumpstart-llama_3_vision_language_model_deployment.ipynb`
**Why Selected:** Multimodal model deployment. Shows emerging vision-language capabilities.

---

## Summary Statistics

| Category | Count | Focus |
|----------|-------|-------|
| **Distributed Training (FSDP, TP, EP, CP)** | 10 | Latest SMP v2, FP8, P5/P4 instances |
| **Fine-tuning (LoRA, QLoRA, Domain Adapt)** | 10 | Llama 3.x, Mixtral, Gemma, custom scripts |
| **Serving & Inference (TGI, LMI, DJL)** | 10 | Large models, async, streaming, autoscaling |
| **Production (RAG, Monitoring, Safety)** | 10 | Real-world patterns, embeddings, moderation |
| **Total** | **40** | - |

## Model Coverage

- **Llama Family:** 15 files (2, 3, 3.1, 3.2, Code Llama, Llama Guard)
- **Mixtral (MoE):** 4 files
- **GPT-NeoX:** 2 files
- **BLOOM:** 1 file (176B)
- **Mistral:** 1 file
- **Gemma:** 1 file
- **Other (GPT-J, etc.):** 2 files
- **Multi-model/General:** 14 files

## Technology Coverage

### Distributed Training Frameworks
- ✅ SageMaker Model Parallel v2 (FSDP + Tensor Parallelism)
- ✅ FP8 Training (Mixed Precision)
- ✅ Context Parallelism (Long Sequences)
- ✅ Expert Parallelism (MoE Models)
- ✅ P5/P4 Instance Optimization

### Fine-tuning Techniques
- ✅ LoRA (Low-Rank Adaptation)
- ✅ QLoRA (Quantized LoRA)
- ✅ Domain Adaptation
- ✅ Instruction Tuning
- ✅ Multimodal Fine-tuning
- ✅ Custom Training Scripts (BYOC)

### Serving Technologies
- ✅ DJL Serving (DeepSpeed, Accelerate)
- ✅ Text Generation Inference (TGI)
- ✅ Large Model Inference (LMI)
- ✅ TensorRT-LLM
- ✅ Async Inference
- ✅ Multi-Model Endpoints
- ✅ Inference Components
- ✅ Token Streaming

### Production Features
- ✅ RAG (Retrieval-Augmented Generation)
- ✅ Vector Embeddings
- ✅ LLM Monitoring & Safety
- ✅ Autoscaling
- ✅ Content Moderation
- ✅ LangChain Integration

---

## Next Steps

### Phase 2: Code-Centric Deep Dive
For each of the 40 files above:
1. Create annotated copies explaining **every SDK call**
2. Trace distributed training configurations (JSON/Python dicts)
3. Map SDK parameters to infrastructure behavior (containers, S3, etc.)

### Phase 3: Synthesis & Reference Guides
1. **Distributed Training Configuration Guide** - Template with all config keys
2. **SageMaker Object Reference Table** - 40 critical SDK objects with usage examples
3. **Fine-tuning Data Flow Guide** - Complete S3 → Container → S3 lifecycle

---

**Document Version:** 1.0
**Created:** 2025-11-23
**Repository:** amazon-sagemaker-examples
**Branch:** claude/sagemaker-genai-tutorials-01HRKfzeesRuYmDLUTugLoyW
