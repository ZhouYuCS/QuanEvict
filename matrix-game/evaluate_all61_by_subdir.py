#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""demo_images 全部 61 张图 × 6 方法评测：按 4 个子文件夹报告相对 full_kv 的逐帧质量。

- 方法：full_kv(基线) + fastar_full/quanevict/quanevict_plus/worldkv/flowcache
- 指标：每方法相对 full_kv 的逐帧 LPIPS/PSNR/SSIM + 子文件夹内全帧 FID
- 不保存视频(mp4)与视频 npy（process_video 被 noop，video 不落盘）
- 报告按 4 个子文件夹（gta_drive/init_image/temple_run/universal）分组

用法（在 mola 目录下运行）：
    CUDA_VISIBLE_DEVICES=1 python evaluate_all61_by_subdir.py
"""
import os
import sys
import gc
import json
import time
import traceback

import numpy as np
import torch

import evaluate_qe_fastar_full as eqf  # noqa: E402

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
DEMO_ROOT = "demo_images"
SUB_DIRS = ["gta_drive", "init_image", "temple_run", "universal"]
OUT_DIR = "outputs_all61_quality"
MAX_FRAMES = 51
SEED = 0


def collect_images():
    groups = {}
    for sub in SUB_DIRS:
        d = os.path.join(DEMO_ROOT, sub)
        imgs = []
        for root, _dirs, files in os.walk(d):
            for f in files:
                if f.lower().endswith(IMAGE_EXTS):
                    imgs.append(os.path.join(root, f))
        groups[sub] = sorted(imgs)
    return groups


def make_args():
    saved = sys.argv
    sys.argv = [saved[0]]
    try:
        args = eqf.parse_args()
    finally:
        sys.argv = saved
    args.max_num_output_frames = MAX_FRAMES
    args.seed = SEED
    args.output_folder = OUT_DIR
    args.methods = "full_kv,fastar_full,quanevict,quanevict_plus,worldkv,flowcache"
    return args


def main():
    groups = collect_images()
    total = sum(len(v) for v in groups.values())
    counts = {k: len(v) for k, v in groups.items()}
    print(f"共 {total} 张图：{counts}", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    args = make_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_blocks = args.max_num_output_frames // 3
    actions, revisit_pairs, shape = eqf.generate_trajectory(num_blocks, device, seed=args.seed, shape=None)
    print(f"轨迹 shape={shape} blocks={num_blocks}", flush=True)

    eqf.set_seed(args.seed)
    game = eqf.InteractiveGameInference(args)
    mode = game.config.pop("mode")
    assert mode == "universal"

    import lpips
    lpips_fn = lpips.LPIPS(net="alex").to(game.device).eval()
    fid_computer = eqf.FIDComputer(game.device)

    METHODS = list(eqf.METHODS)
    non_full = [m for m in METHODS if m["mode"] != "full_kv"]

    final = {}
    for g in SUB_DIRS:
        imgs = groups.get(g, [])
        if not imgs:
            continue
        print(f"\n===== 子文件夹 {g}（{len(imgs)} 图）=====", flush=True)
        acc = {m["key"]: {"label": m["label"], "lpips": [], "psnr": [], "ssim": []} for m in non_full}
        full_feats = []
        method_feats = {m["key"]: [] for m in non_full}

        for i, img in enumerate(imgs):
            print(f"--- [{g}] 图 {i + 1}/{len(imgs)}: {img} ---", flush=True)
            try:
                cond, noise = eqf.build_inputs(game, args, mode, img_path=img, keep_vae=True)
            except Exception as e:  # noqa: BLE001
                print(f"  build_inputs 失败：{e}", flush=True)
                traceback.print_exc()
                continue

            video_full = None
            try:
                video_full, *_ = eqf.run_one(
                    game, args, cond, noise, actions, eqf.METHOD_BY_KEY["full_kv"],
                    f"{g}_{os.path.splitext(os.path.basename(img))[0]}_fullkv")
                full_feats.append(fid_computer.features(video_full, args.fid_batch_size))
            except Exception as e:  # noqa: BLE001
                print(f"  full_kv 失败：{e}", flush=True)
                traceback.print_exc()
                continue

            for m in non_full:
                try:
                    video_m, *_ = eqf.run_one(
                        game, args, cond, noise, actions, m,
                        f"{g}_{os.path.splitext(os.path.basename(img))[0]}_{m['key']}")
                    af = eqf.compute_all_frame_metrics(lpips_fn, video_m, video_full, game.device)
                    acc[m["key"]]["lpips"].extend(af["lpips"])
                    acc[m["key"]]["psnr"].extend(af["psnr"])
                    acc[m["key"]]["ssim"].extend(af["ssim"])
                    method_feats[m["key"]].append(fid_computer.features(video_m, args.fid_batch_size))
                    del video_m
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as e:  # noqa: BLE001
                    print(f"  {m['key']} 失败：{e}", flush=True)
                    traceback.print_exc()

            if video_full is not None:
                del video_full
            gc.collect()
            torch.cuda.empty_cache()

        # 聚合该组
        group_result = {"n_imgs": len(imgs), "methods": {}}
        if full_feats:
            ff = np.concatenate(full_feats, axis=0)
        else:
            ff = None
        for k, a in acc.items():
            lp_mean, lp_std = eqf._mean_std(a["lpips"])
            ps_mean, ps_std = eqf._mean_std(a["psnr"])
            ss_mean, ss_std = eqf._mean_std(a["ssim"])
            fid = None
            if ff is not None and method_feats[k]:
                fm = np.concatenate(method_feats[k], axis=0)
                fid = eqf.FIDComputer.fid(fm, ff)
            group_result["methods"][k] = {
                "label": a["label"],
                "lpips_mean": lp_mean, "lpips_std": lp_std,
                "psnr_mean": ps_mean, "psnr_std": ps_std,
                "ssim_mean": ss_mean, "ssim_std": ss_std,
                "fid": fid, "n_frames": len(a["lpips"]),
            }
        final[g] = group_result

        out_json = os.path.join(OUT_DIR, "results.json")
        with open(out_json, "w") as f:
            json.dump(final, f, indent=2, ensure_ascii=False)
        print(f"[{g}] 完成并已写入 {out_json}", flush=True)

        del acc, full_feats, method_feats, ff
        gc.collect()
        torch.cuda.empty_cache()

    report = build_report(args, final, shape)
    out_md = os.path.join(OUT_DIR, "评价报告.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n报告已保存：{out_md}", flush=True)


def build_report(args, final, shape):
    L = []
    L.append("# demo_images 全样本 6 方法质量评测（按子文件夹分组）")
    L.append("")
    L.append(f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append("")
    L.append(f"- 轨迹：{shape}（{args.max_num_output_frames} latent 帧）")
    L.append("- 指标：各方法相对 full-kv 的逐帧 LPIPS/PSNR/SSIM + 子文件夹内全帧 FID")
    L.append("")
    for g in SUB_DIRS:
        if g not in final:
            continue
        r = final[g]
        L.append(f"## {g}（{r['n_imgs']} 张图）")
        L.append("")
        L.append("| 方法 | 平均 LPIPS↓ | 平均 PSNR↑ | 平均 SSIM↑ | 全帧 FID↓ | 帧数 |")
        L.append("|---|---|---|---|---|---|")
        for k, m in r["methods"].items():
            lp = f"{m['lpips_mean']:.4f}±{m['lpips_std']:.4f}" if m["lpips_mean"] is not None else "N/A"
            ps = f"{m['psnr_mean']:.2f}±{m['psnr_std']:.2f}" if m["psnr_mean"] is not None else "N/A"
            ss = f"{m['ssim_mean']:.4f}±{m['ssim_std']:.4f}" if m["ssim_mean"] is not None else "N/A"
            fid = f"{m['fid']:.2f}" if m["fid"] is not None else "N/A"
            L.append(f"| {m['label']} | {lp} | {ps} | {ss} | {fid} | {m['n_frames']} |")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
