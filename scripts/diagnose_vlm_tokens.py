#!/usr/bin/env python3
"""Diagnose Qwen3-VL visual token compression behavior.

Systematically tests how prompt_tokens scale with:
- Frame count (1, 2, 4, 8, 16, 32, 64, 128)
- Resolution (240p, 360p, 480p, 720p, 1080p)
- detail parameter (none, low, high)
- Provider (OpenRouter Qwen vs GPT-4o baseline)
- Truncation vs compression (numbered frames test)

Usage:
    export OPENAI_API_KEY=sk-...        # OpenRouter key
    export OPENAI_BASE_URL=https://openrouter.ai/api/v1

    python scripts/diagnose_vlm_tokens.py [--frame-cache /path/to/cache] [--video-id 2RLOmpMC1gw]
"""

import argparse
import base64
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from openai import OpenAI


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FRAME_CACHE = "/data/yutao/csh/FakeVideoPipeline/.frame_cache"
DEFAULT_VIDEO_ID = "2RLOmpMC1gw"

MODELS = {
    "openrouter_qwen": {
        "provider": "OpenRouter",
        "model": "qwen/qwen3-vl-235b-a22b-instruct",
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "default_base_url": "https://openrouter.ai/api/v1",
    },
    "openrouter_gpt4o": {
        "provider": "OpenRouter",
        "model": "openai/gpt-4o",
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "default_base_url": "https://openrouter.ai/api/v1",
    },
}

PROMPT = "Describe what you see in these images briefly. Reply in JSON: {\"description\": \"...\"}"

TRUNCATION_PROMPT = (
    "Each image has a large number drawn on it. "
    "List ALL the numbers you can see, in order. "
    "Reply in JSON: {\"numbers_seen\": [1, 2, 3, ...], \"total_images_seen\": N}"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_client(api_key_env: str, base_url_env: str, default_base_url: str) -> OpenAI:
    api_key = os.environ.get(api_key_env, "")
    base_url = os.environ.get(base_url_env, default_base_url)
    if not api_key:
        raise ValueError(f"Missing env var: {api_key_env}")
    return OpenAI(api_key=api_key, base_url=base_url)


def load_frames(cache_dir: str, video_id: str, resolution: str, count: int) -> list[str]:
    """Load `count` frame paths from cache directory.

    Tries multiple naming conventions:
    - {video_id}_h{resolution}_n{count}
    - {video_id}_h{resolution}_n{any}  (use larger cache, take first N)
    - {video_id}_h{resolution}
    """
    base = Path(cache_dir)

    # Exact match
    exact = base / f"{video_id}_h{resolution}_n{count}"
    if exact.is_dir():
        frames = _glob_frames(exact)
        if frames:
            return frames[:count]

    # Any n* match for this resolution (e.g., n128 cache when we need 32)
    for d in sorted(base.glob(f"{video_id}_h{resolution}_n*")):
        frames = _glob_frames(d)
        if len(frames) >= count:
            return frames[:count]

    # Generic resolution dir
    generic = base / f"{video_id}_h{resolution}"
    if generic.is_dir():
        frames = _glob_frames(generic)
        if frames:
            return frames[:count]

    return []


def _glob_frames(d: Path) -> list[str]:
    """Get sorted frame paths from a directory."""
    frames = sorted(str(p) for p in d.glob("*.jpg"))
    if not frames:
        frames = sorted(str(p) for p in d.glob("*.png"))
    return frames


def create_numbered_images(count: int, tmp_dir: str, width: int = 854, height: int = 480) -> list[str]:
    """Create images with large numbers drawn on them for truncation testing."""
    out_dir = Path(tmp_dir) / "numbered"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for i in range(count):
        img = np.ones((height, width, 3), dtype=np.uint8) * 200  # light gray bg
        num_text = str(i + 1)
        font_scale = min(width, height) / 80.0
        thickness = max(2, int(font_scale * 2))
        text_size = cv2.getTextSize(num_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        cv2.putText(img, num_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        # Add a colored border so frames are visually distinct
        border_color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                        (255, 0, 255), (0, 255, 255), (128, 0, 128), (0, 128, 128)]
        color = border_color[i % len(border_color)]
        cv2.rectangle(img, (0, 0), (width - 1, height - 1), color, 8)
        out_path = str(out_dir / f"num_{i+1:03d}.jpg")
        cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        out_paths.append(out_path)
    return out_paths


def resize_frames(frame_paths: list[str], target_height: int, tmp_dir: str) -> list[str]:
    """Resize frames to target height, save to tmp_dir, return new paths."""
    out_dir = Path(tmp_dir) / f"resized_h{target_height}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for i, p in enumerate(frame_paths):
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        new_w = int(w * target_height / h)
        resized = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA)
        out_path = str(out_dir / f"frame_{i:04d}.jpg")
        cv2.imwrite(out_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
        out_paths.append(out_path)
    return out_paths


def build_content(text: str, image_paths: list[str], detail: str | None = None) -> list[dict]:
    """Build OpenAI multimodal content with optional detail parameter."""
    parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for p in image_paths:
        with open(p, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        img_url: dict[str, Any] = {"url": f"data:image/jpeg;base64,{img_b64}"}
        if detail:
            img_url["detail"] = detail
        parts.append({"type": "image_url", "image_url": img_url})
    return parts


def call_api(client: OpenAI, model: str, content: list[dict], temperature: float = 0.0,
             json_mode: bool = True) -> dict:
    """Call API and return token usage dict + response text."""
    try:
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)
        usage = response.usage
        text = response.choices[0].message.content if response.choices else ""
        return {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
            "response_text": text or "",
            "success": True,
            "error": None,
        }
    except Exception as e:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "response_text": "",
            "success": False,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def run_experiment(
    name: str,
    model_key: str,
    model_cfg: dict,
    image_paths: list[str],
    detail: str | None = None,
) -> dict:
    """Run a single API call and return results."""
    client = get_client(
        model_cfg["api_key_env"],
        model_cfg["base_url_env"],
        model_cfg["default_base_url"],
    )
    content = build_content(PROMPT, image_paths, detail=detail)
    result = call_api(client, model_cfg["model"], content)
    result["experiment"] = name
    result["model_key"] = model_key
    result["provider"] = model_cfg["provider"]
    result["model"] = model_cfg["model"]
    result["num_images"] = len(image_paths)
    result["detail"] = detail or "none"
    return result


def experiment_1_frame_scan(cache_dir: str, video_id: str) -> list[dict]:
    """Experiment 1: Frame count scan at 480p."""
    print("\n=== 实验 1: 帧数扫描 (480p) ===")
    frame_counts = [1, 2, 4, 8, 16, 32, 64, 128]
    results = []

    for count in frame_counts:
        frames = load_frames(cache_dir, video_id, "480", count)
        if not frames:
            print(f"  {count}帧: 缓存未找到，跳过")
            continue
        frames = frames[:count]
        print(f"  {count}帧: 找到 {len(frames)} 张图", end=" → ")

        for model_key in ["openrouter_qwen", "openrouter_gpt4o"]:
            cfg = MODELS[model_key]
            if not os.environ.get(cfg["api_key_env"]):
                continue
            r = run_experiment("帧数扫描", model_key, cfg, frames)
            results.append(r)
            tokens_per_img = r["prompt_tokens"] / len(frames) if frames else 0
            print(f"{model_key}: {r['prompt_tokens']}p ({tokens_per_img:.0f}/img)  ", end="")
        print()
        time.sleep(1)

    return results


def experiment_2_resolution_scan(cache_dir: str, video_id: str, tmp_dir: str) -> list[dict]:
    """Experiment 2: Resolution scan at 32 frames.

    If target resolution cache doesn't exist, resizes from 480p frames.
    """
    print("\n=== 实验 2: 分辨率扫描 (32帧) ===")
    resolutions = [240, 360, 480, 720, 1080]
    results = []

    # Pre-load 480p frames as fallback for resizing
    fallback_frames = load_frames(cache_dir, video_id, "480", 32)

    for res in resolutions:
        frames = load_frames(cache_dir, video_id, str(res), 32)
        if not frames:
            # Resize from 480p
            if fallback_frames:
                print(f"  {res}p: 缓存未找到，从 480p resize...", end=" ")
                frames = resize_frames(fallback_frames[:32], res, tmp_dir)
            else:
                print(f"  {res}p: 无可用帧，跳过")
                continue
        else:
            frames = frames[:32]
        print(f"  {res}p: {len(frames)} 张图", end=" → ")

        for model_key in ["openrouter_qwen", "openrouter_gpt4o"]:
            cfg = MODELS[model_key]
            if not os.environ.get(cfg["api_key_env"]):
                continue
            r = run_experiment("分辨率扫描", model_key, cfg, frames)
            results.append(r)
            tokens_per_img = r["prompt_tokens"] / len(frames) if frames else 0
            print(f"{model_key}: {r['prompt_tokens']}p ({tokens_per_img:.0f}/img)  ", end="")
        print()
        time.sleep(1)

    return results


def experiment_3_detail_param(cache_dir: str, video_id: str) -> list[dict]:
    """Experiment 3: detail parameter test."""
    print("\n=== 实验 3: detail 参数测试 (480p 32帧) ===")
    frames = load_frames(cache_dir, video_id, "480", 32)
    if not frames:
        print("  缓存未找到，跳过")
        return []
    frames = frames[:32]
    results = []

    for detail in [None, "low", "high"]:
        label = detail or "none"
        print(f"  detail={label}:", end=" ")

        for model_key in ["openrouter_qwen", "openrouter_gpt4o"]:
            cfg = MODELS[model_key]
            if not os.environ.get(cfg["api_key_env"]):
                continue
            r = run_experiment("detail参数", model_key, cfg, frames, detail=detail)
            results.append(r)
            print(f"{model_key}: {r['prompt_tokens']}p  ", end="")
        print()
        time.sleep(1)

    return results


def experiment_4_provider_comparison(cache_dir: str, video_id: str) -> list[dict]:
    """Experiment 4: Cross-provider comparison."""
    print("\n=== 实验 4: 多服务商对比 (480p 32帧) ===")
    frames = load_frames(cache_dir, video_id, "480", 32)
    if not frames:
        print("  缓存未找到，跳过")
        return []
    frames = frames[:32]
    results = []

    for model_key, cfg in MODELS.items():
        if not os.environ.get(cfg["api_key_env"]):
            print(f"  {model_key}: 无 API key，跳过")
            continue
        print(f"  {model_key} ({cfg['provider']}):", end=" ")
        r = run_experiment("服务商对比", model_key, cfg, frames)
        results.append(r)
        tokens_per_img = r["prompt_tokens"] / len(frames) if frames else 0
        print(f"{r['prompt_tokens']}p ({tokens_per_img:.0f}/img)")
        time.sleep(1)

    return results


def experiment_5_single_image(cache_dir: str, video_id: str) -> list[dict]:
    """Experiment 5: Single image baseline."""
    print("\n=== 实验 5: 单图基线 (480p) ===")
    frames = load_frames(cache_dir, video_id, "480", 1)
    if not frames:
        print("  缓存未找到，跳过")
        return []
    frames = frames[:1]
    results = []

    for model_key, cfg in MODELS.items():
        if not os.environ.get(cfg["api_key_env"]):
            continue
        print(f"  {model_key} ({cfg['provider']}):", end=" ")
        r = run_experiment("单图基线", model_key, cfg, frames)
        results.append(r)
        print(f"{r['prompt_tokens']}p")
        time.sleep(1)

    return results


def experiment_6_truncation_test(tmp_dir: str) -> list[dict]:
    """Experiment 6: Truncation vs compression test.

    Creates images with large numbers (1, 2, 3, ..., N), sends them to the model,
    and asks it to list all numbers seen. If numbers are missing, frames were truncated.
    """
    print("\n=== 实验 6: 截断 vs 压缩测试 ===")
    frame_counts = [8, 16, 32, 64, 128]
    results = []

    for count in frame_counts:
        frames = create_numbered_images(count, tmp_dir)
        print(f"  {count}帧 (编号 1-{count}):", end=" ")

        for model_key in ["openrouter_qwen", "openrouter_gpt4o"]:
            cfg = MODELS[model_key]
            if not os.environ.get(cfg["api_key_env"]):
                continue

            client = get_client(cfg["api_key_env"], cfg["base_url_env"], cfg["default_base_url"])
            content = build_content(TRUNCATION_PROMPT, frames)
            r = call_api(client, cfg["model"], content)
            r["experiment"] = "截断测试"
            r["model_key"] = model_key
            r["provider"] = cfg["provider"]
            r["model"] = cfg["model"]
            r["num_images"] = count
            r["detail"] = "none"
            results.append(r)

            # Parse response to check which numbers were seen
            expected = set(range(1, count + 1))
            seen = set()
            try:
                resp = json.loads(r["response_text"])
                seen = set(resp.get("numbers_seen", []))
            except (json.JSONDecodeError, TypeError):
                pass
            missing = expected - seen
            tokens_per_img = r["prompt_tokens"] / count if count else 0

            if missing:
                status = f"截断! 缺少 {sorted(missing)}"
            else:
                status = "完整"
            print(f"{model_key}: {r['prompt_tokens']}p ({tokens_per_img:.0f}/img) [{status}]  ", end="")
        print()
        time.sleep(2)

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(all_results: list[dict]):
    """Print formatted summary tables."""
    print("\n" + "=" * 80)
    print("汇总结果")
    print("=" * 80)

    # Group by experiment
    experiments: dict[str, list[dict]] = {}
    for r in all_results:
        exp = r["experiment"]
        if exp not in experiments:
            experiments[exp] = []
        experiments[exp].append(r)

    for exp_name, exp_results in experiments.items():
        print(f"\n--- {exp_name} ---")
        print(f"{'provider':<15} {'model':<35} {'images':>6} {'detail':>6} {'prompt_t':>10} {'comp_t':>10} {'per_img':>8} {'status':>8}")
        print("-" * 110)
        for r in exp_results:
            per_img = r["prompt_tokens"] / r["num_images"] if r["num_images"] else 0
            if not r["success"]:
                status = "ERR"
            elif exp_name == "截断测试":
                # Parse truncation result
                expected = set(range(1, r["num_images"] + 1))
                seen = set()
                try:
                    resp = json.loads(r["response_text"])
                    seen = set(resp.get("numbers_seen", []))
                except (json.JSONDecodeError, TypeError):
                    pass
                missing = expected - seen
                status = f"缺{len(missing)}" if missing else "完整"
            else:
                status = "OK"
            print(
                f"{r['provider']:<15} {r['model']:<35} {r['num_images']:>6} {r['detail']:>6} "
                f"{r['prompt_tokens']:>10} {r['completion_tokens']:>10} {per_img:>8.1f} {status:>8}"
            )


def save_results(all_results: list[dict], output_dir: str):
    """Save results to JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "diagnosis_results.json"
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Diagnose VLM token compression")
    parser.add_argument("--frame-cache", default=FRAME_CACHE, help="Frame cache directory")
    parser.add_argument("--video-id", default=DEFAULT_VIDEO_ID, help="Video ID to use")
    parser.add_argument("--output-dir", default="result/token_diagnosis", help="Output directory")
    parser.add_argument("--experiment", type=int, default=0, help="Run specific experiment (1-6, 0=all)")
    args = parser.parse_args()

    print(f"Frame cache: {args.frame_cache}")
    print(f"Video ID: {args.video_id}")
    print(f"Available providers:")
    for key, cfg in MODELS.items():
        has_key = "YES" if os.environ.get(cfg["api_key_env"]) else "NO"
        print(f"  {key}: {cfg['provider']} / {cfg['model']} (key: {has_key})")

    all_results: list[dict] = []

    # Create temp dir for resized frames
    tmp_dir = tempfile.mkdtemp(prefix="vlm_diagnosis_")
    print(f"Temp dir for resized frames: {tmp_dir}")

    exp_num = args.experiment
    if exp_num == 0 or exp_num == 1:
        all_results.extend(experiment_1_frame_scan(args.frame_cache, args.video_id))
    if exp_num == 0 or exp_num == 2:
        all_results.extend(experiment_2_resolution_scan(args.frame_cache, args.video_id, tmp_dir))
    if exp_num == 0 or exp_num == 3:
        all_results.extend(experiment_3_detail_param(args.frame_cache, args.video_id))
    if exp_num == 0 or exp_num == 4:
        all_results.extend(experiment_4_provider_comparison(args.frame_cache, args.video_id))
    if exp_num == 0 or exp_num == 5:
        all_results.extend(experiment_5_single_image(args.frame_cache, args.video_id))
    if exp_num == 0 or exp_num == 6:
        all_results.extend(experiment_6_truncation_test(tmp_dir))

    print_summary(all_results)
    save_results(all_results, args.output_dir)


if __name__ == "__main__":
    main()
