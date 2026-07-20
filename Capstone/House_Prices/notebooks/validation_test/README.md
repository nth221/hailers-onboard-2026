# Validation tests

이 폴더는 가격 모델과 입력 feature를 고정한 채 validation 설계만 비교한다.

- [`basemlp_validation_strategy_comparison.ipynb`](basemlp_validation_strategy_comparison.ipynb): 7개 validation 관점과 3개 training-row 정책 비교
- [`basemlp_repeated_cv_submission.ipynb`](basemlp_repeated_cv_submission.ipynb): keep-all BaseMLP를 5-fold × 3 seeds로 실행하고 15-model 평균 제출 CSV 생성
- [`basemlp_repeated_cv_baseline.ipynb`](basemlp_repeated_cv_baseline.ipynb): test/public 입력 없이 동일 recipe의 CV·OOF를 재현한 향후 feature 실험용 frozen validation baseline
- [`basemlp_feature_blend_rank_validation.ipynb`](basemlp_feature_blend_rank_validation.ipynb): baseline anchor와 raw-year/세 feature blend의 반복-CV·OOF 방향을 실제 public 순위와 비교
- 주 검증 후보: `KFold(5, shuffle=True)`를 split seed `42`, `2026`, `3407`에서 반복
- 보조 stress test: `GrLivArea > 4,000` 네 행을 fold마다 하나씩 배치한 tail-balanced 4-fold

기존 Kaggle public 점수는 정책 순위 진단에만 사용하며 split 생성이나 모델 학습에는 사용하지 않는다.

향후 feature 후보의 주 비교 기준은 frozen baseline의 동일 15개 seed/fold paired delta다. 절대 CV delta 0.0005 미만은 동률로 취급하고 반복 OOF·3-seed ensemble OOF·개선 fold 수를 함께 확인한다.

## W&B tracking

`basemlp_repeated_cv_baseline.ipynb`은 split seed와 fold마다 W&B run 하나를 만들고,
15개 run을 experiment ID group으로 묶는다. 각 run은 epoch별 train loss,
validation loss, validation RMSLE, learning rate와 best/stopping epoch를 기록한다.

```bash
.venv/bin/wandb login
.venv/bin/wandb sync artifacts/validation_test/wandb/wandb/offline-run-*
```

새 실행을 바로 업로드하려면 `WANDB_MODE=online`, 로컬 기록만 만들려면
`WANDB_MODE=offline`, W&B를 완전히 끄려면 `WANDB_MODE=disabled`로 notebook을
실행한다. 기본 project는 `house-prices`이며 `WANDB_PROJECT`와 `WANDB_ENTITY`로
변경할 수 있다.
