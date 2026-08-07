#!/usr/bin/env python3
"""EMA checkpoint 와 .live checkpoint 를 파라미터별로 대조한다.

pretrain.py:250-252 는 같은 step 에서 두 벌을 저장한다:
    step_N        <- EMA 가중치 (ema_copy 로 만든 사본)
    step_N.live   <- 실제 SGD 가중치

EMA 가 어떤 텐서를 아예 갱신하지 못했다면 그 텐서는 두 파일에서 **비트 동일**이 되고,
상대차가 정확히 0 으로 떨어진다. 이것이 판정 신호다.

    상대차 = mean( |ema - live| / (|live| + 1e-8) )

용법:
    python3 scripts/diag_ema_vs_live.py <ema_ckpt> <live_ckpt> [--top 5]
"""
import argparse
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ema")
    ap.add_argument("live")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--must", nargs="*",
                    default=["puzzle_emb.weights", "L_level.layers.0.mlp.gate_up"],
                    help="반드시 표에 포함할 파라미터 이름 부분문자열")
    args = ap.parse_args()

    print(f"[ema ] {args.ema}")
    print(f"[live] {args.live}\n")
    sd_e = torch.load(args.ema, map_location="cpu", weights_only=True)
    sd_l = torch.load(args.live, map_location="cpu", weights_only=True)

    only_e = sorted(set(sd_e) - set(sd_l))
    only_l = sorted(set(sd_l) - set(sd_e))
    if only_e or only_l:
        print(f"! 키 불일치 ema-only={only_e} live-only={only_l}\n")

    rows = []
    for k in sorted(set(sd_e) & set(sd_l)):
        a, b = sd_e[k], sd_l[k]
        if not torch.is_floating_point(a) or a.shape != b.shape:
            continue
        a = a.double()
        b = b.double()
        rel = ((a - b).abs() / (b.abs() + 1e-8)).mean().item()
        # 보조 지표: 원소별 상대차는 |live| 이 0 근처인 원소에서 발산한다.
        # 노름비는 그 영향을 받지 않으므로 "실제로 얼마나 벌어졌나" 는 이쪽을 본다.
        nr = ((a - b).norm() / (b.norm() + 1e-12)).item()
        bit_identical = bool(torch.equal(sd_e[k], sd_l[k]))
        rows.append((rel, k, tuple(a.shape), a.numel(), bit_identical, nr))

    rows.sort(key=lambda r: r[0])
    n = len(rows)
    print(f"비교 가능한 부동소수 텐서 {n} 개\n")

    def show(title, sel):
        print(f"### {title}\n")
        print("| 상대차(지정식) | 노름비 ‖Δ‖/‖live‖ | bit-identical | shape | 파라미터 |")
        print("|---:|---:|:---:|---|---|")
        for rel, k, sh, ne, bi, nr in sel:
            print(f"| {rel:.6f} | {nr:.6f} | {'예' if bi else '아니오'} | {sh} | `{k}` |")
        print()

    show("★ 지정 항목", [r for r in rows if any(m in r[1] for m in args.must)])
    show(f"하위 {args.top} (EMA 가 안 움직인 쪽)", rows[: args.top])
    show(f"상위 {args.top} (EMA 가 가장 많이 벌어진 쪽)", rows[-args.top:][::-1])

    zeros = [r for r in rows if r[0] == 0.0]
    print(f"### 요약\n")
    print(f"- 상대차 정확히 0 인 텐서: **{len(zeros)} / {n}**")
    for rel, k, sh, ne, bi, nr in zeros:
        print(f"  - `{k}`  shape={sh}  bit-identical={'예' if bi else '아니오'}")
    nz = [r[0] for r in rows if r[0] > 0.0]
    if nz:
        print(f"- 0 이 아닌 텐서 상대차 범위: {min(nz):.6f} ~ {max(nz):.6f} "
              f"(중앙값 {sorted(nz)[len(nz)//2]:.6f})")


if __name__ == "__main__":
    main()
