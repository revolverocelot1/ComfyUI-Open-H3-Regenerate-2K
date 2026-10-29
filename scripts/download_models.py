"""
Open-H3-Regenerate-2K: Model Downloader for ComfyUI
Downloads all required diffusion models, text encoders, VAEs, latent upscalers, and refiner LoRAs
directly into your ComfyUI models directory.

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

# Default model inventory
MODELS = [
    # 1. Video VAE & Audio VAE
    {
        "category": "vae",
        "repo": "Comfy-Org/MiniMax-H3",
        "remote_file": "split_files/vae/minimax_h3_video_vae_fp16.safetensors",
        "local_file": "minimax_h3_video_vae_fp16.safetensors",
        "desc": "MiniMax H3 Native Video VAE (FP16)",
    },
    {
        "category": "vae",
        "repo": "Comfy-Org/MiniMax-H3",
        "remote_file": "split_files/vae/minimax_h3_audio_vae_fp32.safetensors",
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
    # 3. Base Diffusion Model (Ref2VA INT8)
    {
        "category": "diffusion_models",
        "repo": "WarmBloodAban/Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8",
        "remote_file": "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors",
        "local_file": "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors",
        "desc": "MiniMax H3 Singularity Ref2VA INT8 Base Model (20 GB)",
    },
    # 4. Open-H3-Regenerate-2K Refiner LoRA (NVlabs Sana / Kijai)
    {
        "category": "loras",
        "repo": "Kijai/MiniMax-H3-experimental",
        "remote_file": "loras/minimax_h3_open_regenerate_2k_refiner_rank_384_bf16.safetensors",
        "local_file": "minimax_h3_open_regenerate_2k_refiner_rank_384_bf16.safetensors",
        "desc": "Open-H3-Regenerate-2K Refiner LoRA (Rank 384 BF16, 5.87 GB)",
    },
    # 5. 3D-Convolutional Latent Upscaler
    {
        "category": "h3_latent_upscalers",
        "repo": "Tridae/H3LatentUpscaler",
        "remote_file": "h3_clean_latent_upscaler_film_epoch200.safetensors",
        "local_file": "h3_clean_latent_upscaler_film_epoch200.safetensors",
        "desc": "H3 3D-Conv Clean Latent Upscaler FiLM (2X Spatial Latent Super-Resolution)",
    },
    # 6. StreamRender-H3 Fast Non-Causal VAE Decoder (Optional 1.6s 2K Decode)
    {
        "category": "vae",
        "repo": "Efficient-Large-Model/Open-H3-Regenerate-2K",
        "remote_file": "vae_fast/fast_h3_vae_noncausal_24l.safetensors",
        "local_file": "fast_h3_vae_noncausal_24l.safetensors",
        "desc": "StreamRender-H3 Fast Non-Causal 24L VAE Decoder (Optional fast decode)",
        "optional": True,
    },
]


def resolve_models_dir(user_input_dir=None):
    if user_input_dir:
        p = Path(user_input_dir).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    # Auto-detect ComfyUI/models from common relative locations
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
    parser = argparse.ArgumentParser(description="Download all models for Open-H3-Regenerate-2K ComfyUI Pipeline")
    parser.add_argument("--models-dir", type=str, default=None, help="Path to ComfyUI models directory")
    parser.add_argument("--skip-optional", action="store_true", help="Skip optional models (e.g. fast 24L VAE)")
    parser.add_argument("--hf-token", type=str, default=None, help="Optional HuggingFace access token")
    args = parser.parse_args()

    models_dir = resolve_models_dir(args.models_dir)
    print(f"=== Open-H3-Regenerate-2K Model Downloader ===")
    print(f"Target Directory: {models_dir}\n")

    for item in MODELS:
        is_optional = item.get("optional", False)
        if is_optional and args.skip_optional:
            continue

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
            if downloaded_path.name != item["local_file"]:
                target_path.write_bytes(downloaded_path.read_bytes())
                downloaded_path.unlink(missing_ok=True)
            print(f"[OK] Downloaded to {target_path}\n")
        except Exception as e:
            if is_optional:
                print(f"[SKIPPED] Optional model could not be downloaded: {e}\n")
            else:
                print(f"[ERROR] Failed to download {item['desc']}: {e}\n")

    # Symlink alias for latent upscaler if required
    alt_dir = models_dir / "latent_upscale_models"
    orig_dir = models_dir / "h3_latent_upscalers"
    if orig_dir.exists() and not alt_dir.exists():
        try:
            alt_dir.symlink_to(orig_dir, target_is_directory=True)
        except Exception:
            pass

    print("=====================================================")
    print("  All models ready for Open-H3-Regenerate-2K!")
    print("=====================================================")


if __name__ == "__main__":
    main()
