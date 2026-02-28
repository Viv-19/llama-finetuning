# Llama-3.2-3B Fine-tuning, Merging & Quantization Pipeline

This project provides a complete, end-to-end pipeline for fine-tuning the **Llama-3.2-3B** model (Instruction-tuned) using **QLoRA**, followed by model merging and **AWQ (Activation-aware Weight Quantization)**.

## 🚀 Project Overview

The goal of this project is to create a lightweight, instruction-following model capable of multiple tasks:
1.  **Conversational AI:** General chat capabilities.
2.  **Summarization:** Condensing long documents (scientific and general).
3.  **Persona-based Response:** Role-playing (e.g., Chandler Bing persona).
4.  **Efficiency:** 4-bit quantization for deployment on consumer hardware.

## 🛠️ Technical Stack

-   **Model:** `meta-llama/Llama-3.2-3B-Instruct`
-   **Methodology:** QLoRA (4-bit Quantized LoRA) for memory efficiency.
-   **Optimization:** AWQ 4-bit Quantization for faster inference.
-   **Libraries:** `transformers`, `peft`, `trl`, `bitsandbytes`, `autoawq`.

## 📂 Project Structure

-   `colab_simple.ipynb`: The main fine-tuning notebook. Handles dataset loading, QLoRA setup, and training.
-   `colab_complete_pipeline.ipynb`: Post-processing notebook. Merges LoRA adapters back into the base model and applies AWQ quantization.
-   `data/`: Contains the raw multi-task dataset.
    -   `chat/`: General chat data (OpenOrca, ShareGPT).
    -   `summarization/`: Summarization data (ArXiv, SamSum).
    -   `persona/`: Persona-specific data (Chandler Bing).

## 🧬 Fine-tuning Pipeline

### 1. Training (QLoRA)
Training is performed on a Google Colab T4 GPU. We use 4-bit quantization for the base model and train LoRA adapters on the target modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).

### 2. Merging
The LoRA adapters are merged with the base model to create a standalone weight file. This allows for easier deployment and further quantization steps.

### 3. Quantization (AWQ)
The merged model is quantized to 4-bit using **AutoAWQ**. This reduces the model size significantly (to ~2GB) while maintaining high performance and drastically reducing VRAM usage.

## 📋 How to Use

1.  **Fine-tune:** Open `colab_simple.ipynb` in Google Colab, set runtime to **T4 GPU**, and run all cells.
2.  **Process Model:** Run `colab_complete_pipeline.ipynb` to merge and quantize the model.
3.  **Inference:**
    ```python
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    model = AutoAWQForCausalLM.from_quantized("./quantized-llama-3.2-3b-awq", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("./quantized-llama-3.2-3b-awq")
    ```

## 📊 Dataset Details

The model is trained on a mixture of:
- **Summarization**: 1,500 ArXiv samples + 500 SamSum samples.
- **Chat**: 600 OpenOrca samples + 400 ShareGPT samples.
- **Persona**: 240+ samples of Chandler Bing's witty and sarcastic responses.
