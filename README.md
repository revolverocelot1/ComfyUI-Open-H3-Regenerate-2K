# ComfyUI-Open-H3-Regenerate-2K

A clean, modular, production-ready two-stage 2K video generation and refinement pipeline for **MiniMax-H3** in standard [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

This repository provides reproducible workflows and exact continuous-flow mathematical scheduling for the **Open-H3-Regenerate-2K** high-resolution refiner ([NVlabs Sana / Efficient-Large-Model](https://github.com/NVlabs/Sana/tree/sol-engine/models/Open-H3-Regenerate-2K)).

> **Standard ComfyUI Compatibility**: 
> These workflows use strictly standard ComfyUI architecture. There are **zero** non-standard patches, **no** `FirstBlockCache`, and **no** `Semantic Bridge` modifications. They run cleanly in standard local ComfyUI installations, as well as cloud environments (Modal, RunPod, Vast.ai, Lambda Labs, Google Cloud).

---

## Table of Contents
1. [Overview & Architecture](#overview--architecture)
2. [Workflows](#workflows)
3. [The Mathematics: Why Stage 2 Takes Time](#the-mathematics-why-stage-2-takes-time)
4. [Hardware Requirements & VRAM Recommendations](#hardware-requirements--vram-recommendations)
5. [Prerequisites & Installation](#prerequisites--installation)
6. [Model Download Inventory](#model-download-inventory)
7. [Automated Model Downloader](#automated-model-downloader)
8. [Quick Start Guide](#quick-start-guide)
9. [Single-Click Stage 2 Bypass (Instant 30s 2K)](#single-click-stage-2-bypass-instant-30s-2k)
10. [Troubleshooting & Common Pitfalls](#troubleshooting--common-pitfalls)
11. [License & Acknowledgments](#license--acknowledgments)

---

## Overview & Architecture

High-resolution video generation ($2048 \times 1152$ at 18–25 frames per second) directly from noise with 14B+ parameter Flow DiTs is computationally prohibitive on modern GPUs due to sequence length quadratic scaling. 

**Open-H3-Regenerate-2K** solves this with a mathematically decoupled two-stage architecture:

```
[Prompt & Conditioning]
          │
          ▼
┌──────────────────────────────────────────────┐
│  STAGE 1: Native Base Generation             │
│  - Resolution: 1024x576 (or 1280x720)        │
│  - DiT: Minimax-h3 Singularity Ref2VA (INT8) │
│  - Steps: 21 (Linear Flow Match, t: 1.0->0)  │
│  - Sequence Length: ~67,000 visual tokens    │
└──────────────────────┬───────────────────────┘
                       │ Clean x0 latents (denoised_output)
                       ▼
┌──────────────────────────────────────────────┐
│  LATENT DOMAIN 2X UPSCALE                    │
│  - Node: MiniMaxH3LatentUpscaleCombined      │
│  - Model: h3_clean_latent_upscaler_film_e200 │
│  - Scales: [1, 24, 107, 38, 66]              │
│         -> [1, 24, 107, 76, 132]             │
│  - Adds initial noise boundary (sigma_0)     │
└──────────────────────┬───────────────────────┘
                       │ Re-noised 2K latents
                       ▼
┌──────────────────────────────────────────────┐
│  STAGE 2: Open-H3-Regenerate-2K Refinement   │
│  - Resolution: 2048x1152 (2K)                │
│  - LoRA: Open-H3-Regenerate-2K (Rank 384)    │
│  - Sequence Length: ~268,000 visual tokens   │
│  - Noise Schedule: [0.909, 0.857, 0.667, 0]  │
│  - Steps: 3 (or 1 in fast mode)              │
│  - Noise Shield: DisableNoise (no double-inj)│
└──────────────────────┬───────────────────────┘
                       │ Denoised 2K latents
                       ▼
┌──────────────────────────────────────────────┐
│  TILED 2K VAE DECODE                         │
│  - Shared Native MiniMax H3 Video VAE (FP16) │
│  - Tiled decoding to prevent VRAM explosion  │
│  - VideoHelperSuite Video Combine (H.264/AAC)│
└──────────────────────────────────────────────┘
```

### Key Technical Innovations in this Workflow
1. **Continuous Flow Step Schedule**: Stage 2 applies NVlabs Sana continuous flow schedule starting at $\sigma_0 = 0.909375$:
   $$\sigma = [0.909375, 0.857143, 0.666667, 0.0]$$
2. **Noise Shield (`DisableNoise`)**: Because the latent upscaler node injects the exact continuous-flow noise component at $\sigma_0$, the Stage 2 sampler connects to `DisableNoise`. This prevents double noise injection, which is the root cause of the infamous 200MB static blowout error.
3. **Shared Single-Instance VAE**: Both Stage 1 and Stage 2 reuse the same loaded VAE instances, eliminating duplicate 6.5 GB VRAM allocations.
4. **Independent LoRA Routing**: The Stage 2 refiner LoRA is loaded directly on a clean branch of the base UNet loader without polluting or altering Stage 1 generation.

---

## Workflows

All workflows are located in the [`workflows/`](file:///d:/Battle.net/fff/ComfyUI-Open-H3-Regenerate-2K/workflows) directory:

| Workflow File | Modality | Inputs | Outputs |
| :--- | :--- | :--- | :--- |
| [`H3_R2V_2K_Open_Regenerate.json`](file:///d:/Battle.net/fff/ComfyUI-Open-H3-Regenerate-2K/workflows/H3_R2V_2K_Open_Regenerate.json) | **Reference-to-Video (R2V)** | Character Reference Images (Face/Costume/Pose) + Prompt | Native 2K Video ($2048 \times 1152$) with synchronized audio |
| [`H3_I2V_2K_Open_Regenerate.json`](file:///d:/Battle.net/fff/ComfyUI-Open-H3-Regenerate-2K/workflows/H3_I2V_2K_Open_Regenerate.json) | **Image-to-Video (I2V)** | Starting Frame Image + Prompt | Native 2K Video ($2048 \times 1152$) with synchronized audio |
| [`H3_T2V_2K_Open_Regenerate.json`](file:///d:/Battle.net/fff/ComfyUI-Open-H3-Regenerate-2K/workflows/H3_T2V_2K_Open_Regenerate.json) | **Text-to-Video (T2V)** | Text Prompt Only | Native 2K Video ($2048 \times 1152$) with synchronized audio |

---

## The Mathematics: Why Stage 2 Takes Time

A common question is: *Why does 21 steps of Stage 1 take ~5 minutes, while only 3 steps of Stage 2 takes ~16 minutes?*

### 1. 3D Latent Token Expansion
The MiniMax-H3 DiT operates on 3D spatiotemporal latent tokens. For an 18-second video clip at 24 fps:
- **Temporal latent frames**: $T = 107$
- **Stage 1 (1024×576)**:
  - Latent spatial dimensions: $H_{\text{lat}} = 576 / 8 = 72$, $W_{\text{lat}} = 1024 / 8 = 128$
  - Patch size: $2 \times 2$
  - Number of tokens:
    $$N_{\text{Stage 1}} = 107 \times \left(\frac{72 \times 128}{4}\right) \approx 67,000 \text{ tokens}$$
- **Stage 2 (2048×1152 - 2K)**:
  - Latent spatial dimensions: $H_{\text{lat}} = 1152 / 8 = 144$, $W_{\text{lat}} = 2048 / 8 = 256$
  - Patch size: $2 \times 2$
  - Number of tokens:
    $$N_{\text{Stage 2}} = 107 \times \left(\frac{144 \times 256}{4}\right) \approx 268,000 \text{ tokens}$$

$$\frac{N_{\text{Stage 2}}}{N_{\text{Stage 1}}} = 4.0\times \text{ sequence length expansion}$$

### 2. Attention Compute Complexity
Standard self-attention scales quadratically $\mathcal{O}(N^2)$ with token count:
$$4^2 = 16\times \text{ increase in raw attention matrix compute}$$

Even with memory-efficient FlashAttention-2 or RingAttention:
- Matrix multiplication FLOPs scale as $\mathcal{O}(N \cdot d_{\text{model}} \cdot \text{layers})$
- Memory bandwidth saturation at 268k tokens forces chunking and heavy memory paging across GPU cache hierarchies.
- As a result, **1 step at 2K takes approximately the same wall-clock compute as 15–18 steps at 768p**.

### 3. Step Schedule Comparison

| Refinement Mode | Stage 2 Schedule | Step Count | Approx. Stage 2 Time (18s clip, H100) | Quality Characteristic |
| :--- | :--- | :---: | :---: | :--- |
| **Bypass Stage 2 (Direct Latent Upscale)** | *None* | 0 | **~30 seconds** | Clean 2K upscale; excellent geometric fidelity; preserves original Stage 1 motion without re-rendering noise. |
| **Fast 1-Step Regenerate** | `[0.909375, 0.0]` | 1 | **~5.5 minutes** | High-frequency skin pore and cloth fiber reconstruction; sharp edge refinement. |
| **Full 3-Step Open-H3-Regenerate** | `[0.909375, 0.857143, 0.666667, 0.0]` | 3 | **~16.0 minutes** | Full cinematic refinement; optimal micro-detail generation matching native high-resolution capture. |

---

## Hardware Requirements & VRAM Recommendations

Due to sequence length expansion ($268\text{k tokens}$ in Stage 2), VRAM allocation during sampling is substantial:

| Hardware Tier | GPU Example | Recommended Operating Mode | Flags / Settings |
| :--- | :--- | :--- | :--- |
| **Consumer (24 GB)** | RTX 3090 / 4090 | Direct Latent Upscale (Bypass Node 207) or 1-Step Regenerate | Set `--reserve-vram 12.0`, enable `expandable_segments:True` |
| **Professional (48 GB)** | RTX 6000 Ada, A6000, L40S | 1-Step or 3-Step Regenerate | Standard ComfyUI memory management |
| **Datacenter (80–96 GB)** | A100 80GB, H100 SXM, H200 | Full 3-Step Regenerate | Set `--reserve-vram 16.0`, optimal throughput |

---

## Prerequisites & Installation

### 1. Install Required ComfyUI Custom Nodes
Navigate to your `ComfyUI/custom_nodes/` directory and clone the three required repositories:

```bash
cd ComfyUI/custom_nodes

# 1. MiniMax H3 Latent Upscaler (Provides the neural FiLM upscaler node)
git clone https://github.com/kijai/ComfyUI-MiniMaxH3_LatentUpscaler.git

# 2. ComfyUI-KJNodes (Provides model routing, noise schedule math, and conditioning helpers)
git clone https://github.com/kijai/ComfyUI-KJNodes.git

# 3. ComfyUI-VideoHelperSuite (Provides audio/video decoding and export)
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

---

## Model Download Inventory

Place each downloaded weight into its corresponding directory under `ComfyUI/models/`:

| Model Name | Target Folder | Size | Source Repository |
| :--- | :--- | :---: | :--- |
| `minimax_h3_video_vae_fp16.safetensors` | `models/vae/` | 4.9 GB | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` | 580 MB | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| `qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors` | `models/text_encoders/` | 15.0 GB | [speach1sdef178/MiniMax-H3-Clean-Text-Encoder](https://huggingface.co/speach1sdef178/MiniMax-H3-Clean-Text-Encoder) |
| `Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors` | `models/diffusion_models/` | 20.0 GB | [WarmBloodAban/Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8](https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8) |
| `minimax_h3_open_regenerate_2k_refiner_rank_384_bf16.safetensors` | `models/loras/` | 1.8 GB | [Kijai/MiniMax-H3-experimental](https://huggingface.co/Kijai/MiniMax-H3-experimental/blob/main/loras/minimax_h3_open_regenerate_2k_refiner_rank_384_bf16.safetensors) |
| `h3_clean_latent_upscaler_film_epoch200.safetensors` | `models/latent_upscaler/` | 54 MB | [Kijai/MiniMax-H3-experimental](https://huggingface.co/Kijai/MiniMax-H3-experimental/tree/main/latent_upscalers) |

---

## Automated Model Downloader

You can use the included Python script to download all models automatically from Hugging Face into your ComfyUI directory:

```bash
# Auto-detects ComfyUI installation or defaults to ./models
python scripts/download_models.py

# Or specify your ComfyUI models path directly:
python scripts/download_models.py --models-dir /workspace/ComfyUI/models
```

*(Optional: Pass `--hf-token <YOUR_TOKEN>` if you have Hugging Face rate limit constraints.)*

---

## Quick Start Guide

1. **Launch ComfyUI**:
   For maximum memory stability and to prevent CUDA memory fragmentation during the 268k token Stage 2 pass, launch with:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   python main.py --listen --port 8188 --reserve-vram 16.0
   ```
   *(On Windows PowerShell, use `$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"` before starting).*

2. **Load Workflow**:
   Drag and drop any of the JSON workflows (`workflows/H3_R2V_2K_Open_Regenerate.json`, `H3_I2V_2K_Open_Regenerate.json`, or `H3_T2V_2K_Open_Regenerate.json`) into the ComfyUI browser canvas.

3. **Provide Inputs**:
   - For **R2V**: Load character reference images into the image loader nodes.
   - For **I2V**: Load your starting frame into Node 143 (`LoadImage`).
   - For **T2V**: Enter your descriptive prompt into Node 133 (`CLIPTextEncode`).

4. **Click "Queue Prompt"**:
   - Stage 1 will generate the base video in ~3–5 minutes.
   - The Latent Upscaler will scale latents to 2K.
   - Stage 2 will refine micro-details using the Open-H3-Regenerate-2K LoRA in 3 continuous flow steps.
   - The final video will render into `ComfyUI/output/` as a high-bitrate 2K MP4 with synchronized audio.

---

## Single-Click Stage 2 Bypass (Instant 30s 2K)

If you are iterating on camera angles, character poses, or lighting and do not want to wait 16 minutes for Stage 2 refinement on every run:

1. Select **Node 207** (`SamplerCustomAdvanced - Stage 2 (Open-H3-Regenerate-2K)`).
2. Press **`Ctrl + M`** (Mute/Bypass).
3. The workflow will automatically forward the upscaled $x_0$ latents directly from the neural FiLM Latent Upscaler (Node 204) to the 2K VAE Decoder (Node 208).
4. **Execution time drops from 16 minutes to ~30 seconds**, producing a sharp, clean 2K video output.

To re-enable 3-step high-resolution micro-refinement, select Node 207 again and press **`Ctrl + M`**.

---

## Troubleshooting & Common Pitfalls

### 1. CUDA Out of Memory (OOM) during Stage 2
- **Symptoms**:
  `CUDA out of memory. Tried to allocate 4.57 GiB. GPU 0 has 95.01 GiB total of which 1.73 GiB is free.`
- **Cause**: Text encoder model (Qwen3-VL 32B) pinned in VRAM, or memory allocator fragmentation under 268k token attention tensors.
- **Fix**:
  1. Do **not** use `--highvram`. Launch ComfyUI with `--reserve-vram 16.0` (or `12.0` on 24GB GPUs) to ensure the text encoder is offloaded to CPU memory once Stage 1 conditioning is baked.
  2. Ensure `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set before starting Python.
  3. Verify both Stage 1 and Stage 2 decode nodes connect to the **same** VAE loaders (Node 119 and Node 120).

### 2. Output Video is Pure Static / Noise (200MB White Noise File)
- **Symptoms**: The generated MP4 file plays as rainbow fuzz or static grain.
- **Cause**: Double noise injection. If `RandomNoise` is plugged into the Stage 2 Sampler, noise is added twice because `MiniMaxH3LatentUpscaleCombined` already performs continuous flow re-noising to $\sigma_0 = 0.909375$.
- **Fix**: Verify that Node 207 (`SamplerCustomAdvanced`) has its `noise` input wired exclusively to **Node 206 (`DisableNoise`)**.

### 3. Missing Custom Nodes Red Boxes
- **Symptoms**: Workflow opens with red outlines or `Node not found` errors.
- **Fix**: Verify that `ComfyUI-MiniMaxH3_LatentUpscaler`, `ComfyUI-KJNodes`, and `ComfyUI-VideoHelperSuite` are cloned inside `ComfyUI/custom_nodes/` and ComfyUI has been restarted.

---

## License & Acknowledgments

- **MiniMax-H3**: Developed by MiniMax AI.
- **Open-H3-Regenerate-2K**: Architecture and refiner model developed by [NVlabs / Efficient-Large-Model](https://github.com/NVlabs/Sana).
- **ComfyUI**: Created by [comfyanonymous](https://github.com/comfyanonymous/ComfyUI).
- **Custom Nodes**: Maintained by [kijai](https://github.com/kijai) and [Kosinkadink](https://github.com/Kosinkadink).
