"""wandb 클라우드 UI 정리 -- run 에 태그를 붙인다. 로컬 wandb/ 는 건드리지 않는다.

삭제가 아니라 분류다. probe/failed 를 태그로 갈라서 기본 뷰에서 필터로 빼고,
성능 상위는 headline 로 묶어 "보고 싶은 것만" 보는 뷰를 만들 수 있게 한다.

  probe     1~102 step 검증용 run (run_configs/hinton/README.md 가 명시)
  failed    state 가 crashed / killed 인 run
  headline  논문 대비 표에 실제로 인용하는 run

기본이 dry-run 이다. 실제 반영은 --commit.

    python experiments/wandb_tag_runs.py                  # dry-run
    python experiments/wandb_tag_runs.py --commit
    python experiments/wandb_tag_runs.py --remove-tag probe --commit   # 되돌리기
"""

import argparse
import os
from collections import defaultdict

PROBE_RUNS = {"nd_off_cat_m4", "nd_off_cat_m4_b", "nd_yproj_on", "nd_both_on"}
# 지시받은 이름 중 nq8_v5_add_b0_s{0,1,2} 는 실제로 _b500 이 들어간다
# (wandb: nq8_v5_add_b0_b500_s0). 두 표기 다 받는다.
HEADLINE_RUNS = {"nq8_v2_base_b500_s0", "nq8_v2_sum007_catnd_s0",
                 "nq10_on_b520_s1", "gc8_v5_catnd_s1",
                 "nq8_v5_add_b0_b500_s0", "nq8_v5_add_b0_b500_s1", "nq8_v5_add_b0_b500_s2",
                 "nq8_v5_add_b0_s0", "nq8_v5_add_b0_s1", "nq8_v5_add_b0_s2"}
FAILED_STATES = {"crashed", "killed", "failed"}
PROJECTS = ["Nqueens-8-ACT-torch", "Nqueens-10-ACT-torch", "Graph-coloring-8-ACT-torch"]


def classify(run):
    """이 run 에 붙일 태그 집합. 이름은 config.run_name 우선, 없으면 run.name."""
    name = (run.config or {}).get("run_name") or run.name
    tags = set()
    if name in PROBE_RUNS or name.startswith("p100_") or name.startswith("probe_"):
        tags.add("probe")
    if (run.state or "").lower() in FAILED_STATES:
        tags.add("failed")
    if name in HEADLINE_RUNS:
        tags.add("headline")
    # harness 업로드용 run (wandb_harness_upload.py 의 harness_run_name):
    # `<train_run>__live_scan` / `__step_N_scan`. 학습 곡선과 성격이 달라 섞이면
    # 안 되므로 태그로 갈라 학습 뷰에서 필터 제외한다.
    if "__live_scan" in name or "_scan" in name or (run.job_type or "") == "harness":
        tags.add("harness")
    return name, tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true", help="실제 반영 (기본 dry-run)")
    ap.add_argument("--projects", nargs="*", default=PROJECTS)
    ap.add_argument("--remove-tag", default=None,
                    help="이 스크립트가 붙인 태그를 되돌린다 (probe|failed|headline)")
    args = ap.parse_args()

    os.environ.setdefault("WANDB_SILENT", "true")
    import wandb
    api = wandb.Api(timeout=60)
    ent = api.default_entity
    print(f"entity={ent}  {'COMMIT' if args.commit else 'DRY-RUN (반영 안 함)'}\n")

    plan, states = [], defaultdict(int)
    for proj in args.projects:
        try:
            runs = list(api.runs(f"{ent}/{proj}", per_page=500))
        except Exception as e:
            print(f"  [warn] {proj} 조회 실패: {type(e).__name__}: {e}")
            continue
        for r in runs:
            name, want = classify(r)
            states[(proj, r.state)] += 1
            cur = set(r.tags or [])
            if args.remove_tag:
                if args.remove_tag in cur:
                    plan.append((proj, r, name, cur, cur - {args.remove_tag}))
                continue
            new = cur | want
            if new != cur:
                plan.append((proj, r, name, cur, new))

    print("state 분포:")
    for (proj, st), n in sorted(states.items()):
        print(f"  {proj:<30} {str(st):<10} {n:>4}")

    if not plan:
        print("\n변경할 run 없음 (이미 태그가 붙어 있거나 대상이 없다)")
        return

    print(f"\n변경 대상 {len(plan)}건:")
    byt = defaultdict(int)
    for proj, r, name, cur, new in plan:
        added = new - cur
        removed = cur - new
        for t in added:
            byt[t] += 1
        mark = ("+" + ",".join(sorted(added))) if added else ("-" + ",".join(sorted(removed)))
        print(f"  [{'set  ' if args.commit else 'plan '}] {proj.split('-')[0]:<12} "
              f"{name:<28} {r.id}  state={str(r.state):<9} {mark}")
        if args.commit:
            r.tags = sorted(new)
            r.update()
    print("\n태그별 건수: " + ", ".join(f"{k}={v}" for k, v in sorted(byt.items())))
    if not args.commit:
        print("반영하려면 --commit")


if __name__ == "__main__":
    main()
