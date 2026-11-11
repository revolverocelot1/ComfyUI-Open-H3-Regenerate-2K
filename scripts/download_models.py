"""
Open-H3-Regenerate-2K-IC: Model Downloader for ComfyUI
Downloads all required diffusion models, text encoders, VAEs, 3D conv latent upscalers,
and In-Context (IC) refiner LoRAs directly into your ComfyUI models directory.

Usage:
    python download_models.py
    python download_models.py --models-dir /path/to/ComfyUI/models
"""

import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Error: huggingface_hub is required. Install it via:")
    print("    pip install huggingface_hub")
    sys.exit(1)

MODELS = [
    # 1. Video VAE & Audio VAE
    {
        "category": "vae",
        "repo": "Comfy-Org/MiniMax-H3",
        "remote_file": "vae/minimax_h3_video_vae_fp16.safetensors",
        "local_file": "minimax_h3_video_vae_fp16.safetensors",
        "desc": "MiniMax H3 Native Video VAE (FP16)",
    },
    {
        "category": "vae",
        "repo": "Comfy-Org/MiniMax-H3",
        "remote_file": "vae/minimax_h3_audio_vae_fp32.safetensors",
        "local_file": "minimax_h3_audio_vae_fp32.safetensors",
        "desc": "MiniMax H3 Native Audio VAE (FP32)",
    },
    # 2. Text Encoder (Qwen3-VL 32B NVFP4)
    {
        "category": "text_encoders",
        "repo": "speach1sdef178/MiniMax-H3-Clean-Text-Encoder",
        "remote_file": "qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors",
        "local_file": "qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors",
        "desc": "Qwen3-VL 32B Multimodal Text Encoder (NVFP4, 15 GB)",
    },
    # 3. Base Diffusion Models (FL2VA & Ref2VA INT8)
    {
        "category": "diffusion_models",
        "repo": "Comfy-Org/MiniMax-H3",
        "remote_file": "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "local_file": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "desc": "MiniMax H3 FL2VA (T2V / I2V) Pruned INT8 ConvRot Model",
    },
    {
        "category": "diffusion_models",
        "repo": "Comfy-Org/MiniMax-H3",
        "remote_file": "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "local_file": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "desc": "MiniMax H3 Ref2VA Pruned INT8 ConvRot Model (Stage 1 R2V & Stage 2 IC)",
    },
    # 4. Open-H3-Regenerate-2K-IC 3-LoRA Stack
    {
        "category": "loras",
        "repo": "slmonker/Open-H3-Regenerate-2K-IC-ComfyUI",
        "remote_file": "minimax_h3_open_regenerate_2k_ic_comfy_bf16.safetensors",
        "local_file": "minimax_h3_open_regenerate_2k_ic_comfy_bf16.safetensors",
        "desc": "Open-H3-Regenerate-2K-IC Converted Refiner LoRA (BF16)",
    },
    {
        "category": "loras",
        "repo": "RunningHubAI/rh-minimax-h3-hyperflow-8step-v1.0-comfyui-bf16.safetensors-lora",
        "remote_file": "minimax_h3_hyperflow_8step_v1.0_comfyui_bf16.safetensors",
        "local_file": "minimax_h3_hyperflow_8step_v1.0_comfyui_bf16.safetensors",
        "desc": "MiniMax H3 HyperFlow 8-Step LoRA (BF16)",
    },
    {
        "category": "loras",
        "repo": "RunningHubAI/rh-minimax-h3-lms-v1.0-r64.safetensors-lora",
        "remote_file": "minimax_h3_lms_v1.0_r64.safetensors",
        "local_file": "minimax_h3_lms_v1.0_r64.safetensors",
        "desc": "MiniMax H3 LMS Sharp r64 LoRA",
    },
    # 5. Official 3D-Conv Latent Upscaler
    {
        "category": "latent_upscale_models",
        "repo": "LBH-123-AI/Minimax_h3_latent_Upscaler",
        "remote_file": "minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors",
        "local_file": "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors",
        "desc": "Official MiniMax H3 3D-Conv Latent Upscaler v1 (BF16)",
    },
]


def resolve_models_dir(user_input_dir=None):
    if user_input_dir:
        p = Path(user_input_dir).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent.parent / "models",
        script_dir.parent.parent / "ComfyUI" / "models",
        Path.cwd() / "models",
        Path.cwd() / "ComfyUI" / "models",
        Path("/root/models"),
    ]
    for c in candidates:
        if c.exists() and (c / "checkpoints" in c.parent.glob("*") or c.name == "models"):
            return c

    default_path = Path.cwd() / "models"
    default_path.mkdir(parents=True, exist_ok=True)
    return default_path


def main():
    parser = argparse.ArgumentParser(description="Download all models for Open-H3-Regenerate-2K-IC ComfyUI Pipeline")
    parser.add_argument("--models-dir", type=str, default=None, help="Path to ComfyUI models directory")
    parser.add_argument("--hf-token", type=str, default=None, help="Optional HuggingFace access token")
    args = parser.parse_args()

    models_dir = resolve_models_dir(args.models_dir)
    print("=== Open-H3-Regenerate-2K-IC Model Downloader ===")
    print(f"Target Directory: {models_dir}\n")

    for item in MODELS:
        cat_dir = models_dir / item["category"]
        cat_dir.mkdir(parents=True, exist_ok=True)
        target_path = cat_dir / item["local_file"]

        if target_path.exists() and target_path.stat().st_size > 1024 * 1024:
            size_mb = target_path.stat().st_size / (1024 * 1024)
            print(f"[FOUND] {item['desc']}")
            print(f"        -> {target_path} ({size_mb:.1f} MB)\n")
            continue

        print(f"[DOWNLOADING] {item['desc']}...")
        print(f"              Source: https://huggingface.co/{item['repo']}")
        try:
            downloaded = hf_hub_download(
                repo_id=item["repo"],
                filename=item["remote_file"],
                local_dir=str(cat_dir),
                token=args.hf_token,
            )
            downloaded_path = Path(downloaded)
            if downloaded_path != target_path:
                target_path.write_bytes(downloaded_path.read_bytes())
                downloaded_path.unlink(missing_ok=True)
            print(f"[OK] Downloaded to {target_path}\n")
        except Exception as e:
            print(f"[ERROR] Failed to download {item['desc']}: {e}\n")

    print("=====================================================")
    print("  All models ready for Open-H3-Regenerate-2K-IC!")
    print("=====================================================")


if __name__ == "__main__":
    main()
