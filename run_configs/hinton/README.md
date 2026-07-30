# 29번(hinton) run 별 all_config.yaml 사본

`checkpoints/` 가 `.gitignore` 대상이라 `checkpoints/<project>/<run>/all_config.yaml` 이
git 에 올라가지 않는다.  50번에서 29번 run 의 손잡이 조합(β / kl_reduction / y_inject /
logvar_bias_init / nodecay_* / y_onehot / seed / epochs)을 대조할 수 없어 REPORT_FINAL
정본 관리에 구멍이 생겼다.  그래서 원본을 손대지 않고 사본만 여기에 둔다.

- 경로: `run_configs/hinton/<project>/<run>.yaml`
- 원본: `checkpoints/<project>/<run>/all_config.yaml` (내용 동일, 파일명만 run 이름으로)
- 64 개 / 268 KB (Nqueens-8 60, Nqueens-10 4)
- host: hinton (= 29번).  wandb entity/project 는 원본과 같다 --
  `jennylee0358-sungkyunkwan-university/{Nqueens-8,Nqueens-10}-ACT-torch`

갱신 방법 (run 이 추가되면 29번에서 재실행):

    find checkpoints -name all_config.yaml | while read f; do
      rel=${f#checkpoints/}; rel=${rel%/all_config.yaml}
      mkdir -p "run_configs/hinton/$(dirname "$rel")"
      cp "$f" "run_configs/hinton/$(dirname "$rel")/$(basename "$rel").yaml"
    done

## 주의 -- 검증용 probe 가 섞여 있다

아래 4 개는 학습 실험이 아니라 코드 검증용 짧은 run(1~102 step)이다.  집계에서 빼야 한다.

    nd_off_cat_m4, nd_off_cat_m4_b, nd_yproj_on, nd_both_on

`nq8_off_b500` 은 지시에 따라 step 1147 에서 kill 한 run 이다(step_1139 만 있음).
`nq8_off_b500_ema6k` 가 같은 β=5.0 OFF 를 EMA/dual_eval 로 다시 돌린 완주 run 이다.
