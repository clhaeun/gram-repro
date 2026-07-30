"""wandb Report "GRAM repro — 논문 대비" 를 만든다. workspace 는 건드리지 않는다.

Report 가 정본 뷰다 -- 120개 run 이 자동으로 들어오면 못 읽으므로 **패널마다 run 을
명시적으로 고른다**.

  Panel C  v5 4-run 의 eval/nqueens/accuracy(EMA) vs accuracy_raw(live).
           34k / 40k 가 §3-3 의 EMA 바닥 구간이다.
  Panel D  train/diag/sigma_prior_std_mean -- β·y_onehot·seed·예산이 다른 11 run.
  Panel A/F/G  GRAM-repro-summary 에 올린 표·이미지 패널 (source ours/paper 대비).

주의: metric 이름은 실측으로 확인한 것이다.
  eval/nqueens/accuracy, eval/nqueens/accuracy_raw   (있음)
  train/diag/sigma_prior_std_mean                     (`diag/...` 는 0건 -- train/ 접두사 필요)

    python experiments/wandb_make_report.py            # dry-run (구성만 출력)
    python experiments/wandb_make_report.py --commit
"""

import argparse
import os
import sys

ENTITY = None            # None -> api.default_entity
NQ8 = "Nqueens-8-ACT-torch"
SUMMARY_PROJECT = "GRAM-repro-summary"
REPORT_TITLE = "GRAM repro — 논문 대비"

# Panel C -- 명시적으로 4개만
PANEL_C_RUNS = ["nq8_v5_add_b0_b500_s0", "nq8_v5_add_b0_b500_s1",
                "nq8_v5_add_b0_b500_s2", "nq8_v5_add_bm4_b500_s1"]

# Panel D -- β(6점) x y_onehot(2) x seed(3) x 예산(3000/6000) 을 가르는 11개.
# 전부 train/diag/sigma_prior_std_mean 보유를 API 로 확인했다.
PANEL_D_RUNS = [
    "nq8_on_b420_ema6k",      # β4.2 on  seed0 6000
    "nq8_on_b450_ema6k",      # β4.5 on  seed0 6000
    "nq8_on_b480_ema6k",      # β4.8 on  seed0 6000
    "nq8_on_b500_ema6k",      # β5.0 on  seed0 6000
    "nq8_on_b520_ema6k",      # β5.2 on  seed0 6000
    "nq8_on_b560_ema6k",      # β5.6 on  seed0 6000
    "nq8_off_b500_ema6k",     # β5.0 OFF seed0 6000   <- y_onehot 축
    "nq8_on_b500_s1",         # β5.0 on  seed1 6000   <- seed 축
    "nq8_on_b500_s2",         # β5.0 on  seed2 6000   <- seed 축
    "nq8_on_b450_ema",        # β4.5 on  seed0 3000   <- 예산 축
    "nq8_on_b560_ema",        # β5.6 on  seed0 3000   <- 예산 축
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--pylibs", default=os.environ.get("GRAM_PYLIBS"),
                    help="wandb_workspaces 를 별도 설치한 경로")
    args = ap.parse_args()

    if args.pylibs and os.path.isdir(args.pylibs) and args.pylibs not in sys.path:
        sys.path.insert(0, args.pylibs)
    try:
        import wandb_workspaces.reports.v2 as wr
    except ImportError:
        print("wandb_workspaces 없음. pip install --target <dir> wandb-workspaces 후 "
              "--pylibs <dir> (또는 GRAM_PYLIBS)")
        return 1
    import wandb
    api = wandb.Api(timeout=60)
    ent = ENTITY or api.default_entity

    def only(runs):
        """run_name 로 필터 -- 자동으로 전 run 이 들어오는 것을 막는다.

        wandb_workspaces 의 filters 는 문자열 표현식을 받는다
        (Runset docstring: "Config('lr') = 0.001 and State = 'finished'").
        """
        lst = ", ".join(repr(n) for n in runs)
        return f"Config('run_name') in [{lst}]"

    blocks = [
        wr.MarkdownBlock(text=(
            "**축 규약** — 가중치 `EMA`(=`eval/*`) / `live`(=`*_raw`), 집계 **단일시도 = pass@1**, "
            "`step_kind` **peak**(test set 으로 고른 step, **선택 효과 포함**) / **final**(마지막 step).\n\n"
            "8×8 헤드라인은 `nq8_v2_base_b500_s0` **정점 0.9977 / 최종 0.9753** 이다 — "
            "격차 **2.24%p** 가 선택 효과의 크기이고, 논문 0.997±0.003 은 그 사이에 있다. "
            "정점만 인용하면 논문을 넘고 최종만 인용하면 못 넘는다.\n\n"
            "Runs 뷰에서 `probe` / `failed` / `harness` 태그는 제외하고 볼 것 "
            "(`harness` 는 harness 업로드용 run 이라 학습 곡선과 섞이면 안 된다)."
        )),
        wr.H1(text="Panel C — EMA 는 붕괴가 아니라 늦게 수렴한다 (v5 4-run)"),
        wr.MarkdownBlock(text=(
            "`eval/nqueens/accuracy` = **EMA 사본**, `eval/nqueens/accuracy_raw` = **live(raw SGD)**. "
            "**34k~40k 구간이 EMA 바닥이다** — 논문 예산(3,000 epochs ≈ 17k step)에서 끊으면 "
            "이 곡선은 바닥 이전이거나 바닥 구간이고 '붕괴'로 오독된다. "
            "raw 는 같은 구간에서 이미 살아 있다. (Report 패널에 수직 기준선은 넣을 수 없어 "
            "x 범위를 34k~40k 가 보이게 잡았다.)"
        )),
        wr.PanelGrid(
            runsets=[wr.Runset(entity=ent, project=NQ8, name="v5 4-run (명시 선택)",
                               filters=only(PANEL_C_RUNS))],
            panels=[
                wr.LinePlot(title="EMA — eval/nqueens/accuracy",
                            x="_step", y=["eval/nqueens/accuracy"],
                            layout=wr.Layout(w=12, h=8)),
                wr.LinePlot(title="live(raw) — eval/nqueens/accuracy_raw",
                            x="_step", y=["eval/nqueens/accuracy_raw"],
                            layout=wr.Layout(w=12, h=8)),
            ],
        ),
        wr.H1(text="Panel D — σ 궤적은 학습 길이의 함수다"),
        wr.MarkdownBlock(text=(
            "`train/diag/sigma_prior_std_mean`. β(4.2~5.6) · y_onehot(on/off) · seed(0/1/2) · "
            "예산(3,000 vs 6,000 epochs)이 다른 **11 run 만** 골랐다.\n\n"
            "**3,000 epochs run 두 개(`nq8_on_b450_ema`, `nq8_on_b560_ema`)의 σ 가 0.68~0.70 으로 "
            "낮고, 같은 β 의 6,000 epochs run 은 0.93 이다.** β 나 seed 가 아니라 **예산**이 σ 를 "
            "가른다 — `logvar_bias_init` 이 최종 σ 를 정하는 손잡이가 아니라 도달 시각을 늦추는 "
            "손잡이라는 §3-2 와 같은 이야기다."
        )),
        wr.PanelGrid(
            runsets=[wr.Runset(entity=ent, project=NQ8, name="σ 대조 11-run (명시 선택)",
                               filters=only(PANEL_D_RUNS))],
            panels=[wr.LinePlot(title="prior σ (train/diag/sigma_prior_std_mean)",
                                x="_step",
                                y=["train/diag/sigma_prior_std_mean"],
                                layout=wr.Layout(w=24, h=10))],
        ),
        wr.H1(text="Panel A / E / H / F / G — 논문 대비 · coverage 축 · 다양성 · 실패 3종"),
        wr.MarkdownBlock(text=(
            "`GRAM-repro-summary` 의 통합 표(646행 × 27열, `server` 열로 29/50 구분)와 그 표에서 "
            "그린 막대 패널이다.\n\n"
            "- **Panel A** ours(peak) / ours(final) / paper. **coverage 막대는 넣지 않았다** — "
            "coverage 는 N 에 강하게 의존해서(§1.4) 단일 막대로 그리면 축이 안 맞는 비교가 된다. "
            "coverage 는 Panel E 로 본다. A 에는 Accuracy(nq8/nq10)와 GC conflict 만 남겼고, "
            "conflict 는 스케일이 달라(0.02 vs 2.7) 별도 축 + log scale 이다 — 이중 y축은 쓰지 않았다.\n"
            "- **Panel E** coverage vs 해 개수(논문 Fig 15 버킷). 실선=실측, 점선=그 N 의 천장 "
            "`min(N,k)/k`, 점선(회색)=논문값. **N=1 에서 실측이 천장에 붙어 있다** — 모델이 아니라 "
            "test 셋 해 개수 분포가 coverage 를 정한다.\n"
            "- **Panel H** 해 개수 분포. 8×8 은 **64%가 해 1개**라 N=1 천장이 **0.7909** 로 막히고, "
            "논문 구간(0.884~0.922)이 원리적으로 도달 불가다. 10×10 천장 0.5930, GC-8 0.4836.\n"
            "- **Panel F** distinct@20(1~20)과 valid(0~1)은 다른 축이라 분리했다. nq8 은 distinct "
            "1.65~2.79(거의 결정론)인데 valid 0.98, gc8 은 distinct 20.00 인데 valid 0.98 — "
            "**다양성과 정확도는 독립이다.**\n"
            "- **Panel G** prior vs posterior. lr decay 0.033/0.859, forcehalt 0.008/0.951, "
            "저β 0.000/0.997 — **세 경우 모두 posterior 는 살아 있고 prior 만 죽는다.**\n\n"
            "**GC conflict 의 ± 는 run 간 std(논문 축)** 이고 예측 5,100개 간 std(±0.169)가 아니다. "
            "우리 GC 3-run 은 arm 이 섞인 근사라 논문의 *over runs* 와 완전히 같은 축은 아니다."
        )),
        wr.PanelGrid(
            runsets=[wr.Runset(entity=ent, project=SUMMARY_PROJECT, name="unified summary")],
            panels=[
                wr.MediaBrowser(media_keys=["panel/A_paper_vs_ours"],
                                layout=wr.Layout(w=24, h=9)),
                wr.MediaBrowser(media_keys=["panel/E_coverage_by_solutions"],
                                layout=wr.Layout(w=24, h=9)),
                wr.MediaBrowser(media_keys=["panel/H_solution_count_dist"],
                                layout=wr.Layout(w=24, h=8)),
                wr.MediaBrowser(media_keys=["panel/F_distinct_vs_valid"],
                                layout=wr.Layout(w=12, h=8)),
                wr.MediaBrowser(media_keys=["panel/G_prior_vs_posterior"],
                                layout=wr.Layout(w=12, h=8)),
            ],
        ),
        wr.MarkdownBlock(text=(
            "### 이 뷰의 한계\n"
            "- **29번 run 은 가중치가 이 박스에 없어 재측정 불가**다. 결과 JSON 의 값이 전부다.\n"
            "- `metric_type=inloop` 행은 harness 와 섞어 읽으면 안 된다(단일 배치 accuracy, "
            "coverage 없음). harness 미측정 run(v4)이 표에서 빠지지 않게 넣은 것이다.\n"
            "- `server=?` 8행은 `all_config.yaml` 이 두 박스 어디에도 없어 판정 못 했다.\n"
            "- probe run(`nd_*`, `p100_*`, `probe_*`)은 **wandb 에 애초에 없다** — "
            "태그로 뺄 대상이 아니었다."
        )),
    ]

    print(f"entity={ent}  title={REPORT_TITLE!r}  {'COMMIT' if args.commit else 'DRY-RUN'}")
    print(f"  블록 {len(blocks)}개 / Panel C {len(PANEL_C_RUNS)} run / "
          f"Panel D {len(PANEL_D_RUNS)} run")
    if not args.commit:
        print("  (workspace 는 건드리지 않는다. 실제 생성은 --commit)")
        return 0

    report = wr.Report(project=SUMMARY_PROJECT, entity=ent, title=REPORT_TITLE,
                       description="세 task x 두 서버. 패널마다 run 을 명시 선택했다.",
                       blocks=blocks)
    report.save()
    print(f"  생성 완료: {report.url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
