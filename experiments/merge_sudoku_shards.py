"""연속블록 샤드 JSON 을 퍼즐수 n 가중평균으로 합친다.

모든 지표가 퍼즐 단위 평균이라 샤드별 n 가중평균이 전수 측정과 정확히 같다.
샤드 범위가 겹치거나 비면 assert 로 막는다.

    python experiments/merge_sudoku_shards.py OUT.json IN1.json IN2.json ...
"""
import json
import sys

RATE_BLOCKS = ["prior_all_samples", "pass_at_N", "majority_at_N", "verifier_selected_at_N"]
SCALARS = ["N", "seed", "sigma_scale", "mu_scale", "halt_max_steps", "checkpoint",
           "data", "limit", "unconditional", "shard_count", "n_total_in_split"]


def merge(paths: list) -> dict:
    shards = [json.load(open(p)) for p in paths]
    for k in SCALARS:
        vals = {json.dumps(s[k]) for s in shards}
        assert len(vals) == 1, f"샤드 간 {k} 불일치: {vals}"

    rng = sorted(tuple(s["shard_range"]) for s in shards)
    assert rng[0][0] == 0, f"첫 샤드가 0 에서 시작하지 않는다: {rng[0]}"
    for a, b in zip(rng, rng[1:]):
        assert a[1] == b[0], f"샤드 범위 불연속/중복: {a} -> {b}"

    total = sum(s["n"] for s in shards)
    assert total == rng[-1][1], f"n 합 {total} != 범위 끝 {rng[-1][1]}"

    out = {k: shards[0][k] for k in SCALARS}
    out.update({"n": total, "shard_range": [rng[0][0], rng[-1][1]],
                "merged_from": list(paths), "shard_ns": [s["n"] for s in shards]})
    for blk in RATE_BLOCKS:
        keys = shards[0][blk].keys()
        out[blk] = {k: sum(s[blk][k] * s["n"] for s in shards) / total for k in keys}
    return out


def main() -> None:
    out_path, in_paths = sys.argv[1], sys.argv[2:]
    assert in_paths, "입력 샤드 JSON 을 하나 이상 넘겨라"
    res = merge(in_paths)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    p, m, s = res["prior_all_samples"], res["majority_at_N"], res["verifier_selected_at_N"]
    print(f"{res['checkpoint']}  hms={res['halt_max_steps']} N={res['N']} "
          f"seed={res['seed']} n={res['n']}\n"
          f"  prior  exact={p['exact']:.4f} solved={p['solved']:.4f}\n"
          f"  pass@N exact={res['pass_at_N']['exact']:.4f}\n"
          f"  maj@N  exact={m['exact']:.4f}\n"
          f"  vsel@N exact={s['exact']:.4f}\n"
          f"  [out] {out_path}")


if __name__ == "__main__":
    main()
