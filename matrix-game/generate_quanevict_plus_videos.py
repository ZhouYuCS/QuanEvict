#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 demo_images 的每个直接子目录各随机取一张图，用 QuanEvict+ 生成并保存视频。

用法（在 mola 目录下运行）：
    CUDA_VISIBLE_DEVICES=1 python generate_quanevict_plus_videos.py
"""
import os
import sys
import gc
import random

# 确保 process_video 生效（evaluate 脚本在无 --save_video 时会用 noop 替换）
if "--save_video" not in sys.argv:
    sys.argv.append("--save_video")

import torch

import evaluate_qe_fastar_full as eqf

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
DEMO_ROOT = "demo_images"
OUT_DIR = "outputs_quanevict_plus_videos"
MAX_FRAMES = 51
SEED = 0


def collect_images_recursive(d):
    out = []
    for root, _dirs, files in os.walk(d):
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                out.append(os.path.join(root, f))
    return sorted(out)


def make_args():
    saved = sys.argv
    sys.argv = [saved[0]]  # 无命令行参数 -> 全默认
    try:
        args = eqf.parse_args()
    finally:
        sys.argv = saved
    args.max_num_output_frames = MAX_FRAMES
    args.seed = SEED
    args.output_folder = OUT_DIR
    args.save_video = True
    args.methods = "quanevict_plus"
    return args


def main():
    if not os.path.isdir(DEMO_ROOT):
        raise SystemExit(f"找不到目录 {DEMO_ROOT}（请在 mola 目录下运行）")
    os.makedirs(OUT_DIR, exist_ok=True)

    subdirs = sorted(
        d for d in os.listdir(DEMO_ROOT)
        if os.path.isdir(os.path.join(DEMO_ROOT, d))
    )
    print(f"直接子目录：{subdirs}")

    picks = []
    for sub in subdirs:
        imgs = collect_images_recursive(os.path.join(DEMO_ROOT, sub))
        if not imgs:
            print(f"[跳过] {sub}：无图片")
            continue
        pick = random.choice(imgs)
        picks.append((sub, pick))
        print(f"[选图] {sub} -> {pick}")

    args = make_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_blocks = args.max_num_output_frames // 3
    actions, revisit_pairs, shape = eqf.generate_trajectory(
        num_blocks, device, seed=args.seed, shape=None)
    print(f"轨迹 shape={shape} blocks={num_blocks} revisit_pairs={revisit_pairs}")

    eqf.set_seed(args.seed)
    game = eqf.InteractiveGameInference(args)
    mode = game.config.pop("mode")
    assert mode == "universal"

    method = eqf.METHOD_BY_KEY["quanevict_plus"]

    for sub, img in picks:
        rel = os.path.relpath(img, DEMO_ROOT)
        name = os.path.splitext(rel)[0].replace(os.sep, "__")
        print(f"\n=== 生成 {name}（{img}）===", flush=True)
        cond, noise = eqf.build_inputs(game, args, mode, img_path=img, keep_vae=True)
        video, bank, peak, wan, vae = eqf.run_one(game, args, cond, noise, actions, method, name)
        print(f"[完成] {name}.mp4  bank={bank / 1e6:.1f}MB peak={peak / 1e9:.2f}GiB "
              f"wan={wan:.2f}s vae={vae:.2f}s", flush=True)
        del video
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n全部完成，视频已保存到 {OUT_DIR}/")


if __name__ == "__main__":
    main()
