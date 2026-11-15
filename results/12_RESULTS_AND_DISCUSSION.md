# Final Results and Discussion

## 1. Benchmark progression

The weekly seasonal-naive benchmark produced a locked final-test MSE of 7.1658, RMSE of 2.6769, MAE of 1.2440, and WAPE of 0.8622. Zero-shot TTM reduced MSE to 4.6379, equivalent to a 35.28% reduction in squared error. This demonstrates substantial transfer from the pre-trained TTM before any retail-domain parameter update.

Head probing further reduced final-test MSE from 4.6379 to 4.5630, an additional 1.61% improvement. The gain was smaller than the initial zero-shot improvement but established a stronger frozen retail-adapted base model.

## 2. Primary EA-TTM performance

The locked primary EA-TTM achieved MSE 4.5286 compared with 4.5630 for frozen HP-TTM, a 0.75% MSE improvement and a 0.38% RMSE improvement.

The improvement was not universal across metrics: MAE worsened by 0.88% and WAPE worsened by 0.88%. This divergence indicates that EA-TTM primarily reduced comparatively large errors rather than uniformly reducing absolute error.

## 3. Normal and extreme regimes

EA-TTM reduced all-extreme MSE by 1.86% and positive-extreme MSE by 2.13%. By contrast, normal-regime MSE worsened by 0.73% and negative-extreme MSE worsened by 1.92%.

The final-test extreme rate was 5.93%. Positive extremes represented 5.55% of all observations, while negative extremes represented only 0.38%. The strongest empirical support is therefore for positive demand surges, not symmetric improvement across upward and downward shocks.

## 4. Gate mechanism

The gate achieved AUROC 0.768 and AUPRC 0.201. Its AUPRC was 3.39 times the extreme prevalence. Mean gate probability was 0.121 on extreme observations and 0.052 on normal observations.

The mean absolute effective correction |g×Delta| was 2.29 times larger for extreme observations, and correction direction agreed with the required residual direction on 73.89% of extreme observations. This demonstrates meaningful routing, but mechanism quality is not equivalent to forecasting optimality.

## 5. Ablation evidence

The retail-only gated adapter achieved MSE 4.5328, while the full primary EA-TTM achieved 4.5286. Together with the validation ablation, this supports a small incremental contribution from the decoder cross-feed.

The ungated cross-feed adapter achieved MSE 4.5132 and extreme MSE 42.3192, both lower than the primary EA-TTM values of 4.5286 and 43.3512. Relative to the primary model, the ungated ablation reduced overall MSE by 0.34% and extreme MSE by 2.38%.

This limits the strongest architectural claim: decoder cross-feed and residual correction are useful, but the multiplicative gate is not the forecast-optimal component under these experiments. The primary model was not replaced because it had been locked before final-test access.

## 6. Bootstrap robustness

The paired series-level MSE difference (HP-TTM minus EA-TTM) was 0.03444. Across 2,000 bootstrap replicates, the 95% percentile interval was [0.02967, 0.03982], entirely above zero under the chosen store-item resampling scheme.

## 7. Performance across the 28-day horizon

EA-TTM reduced overall MSE relative to HP-TTM on 27 of 28 forecast steps and reduced extreme MSE on 27 of 28. The strongest overall horizon improvement occurred at horizon 18 (1.92%), while the weakest occurred at horizon 6 (-0.16%).

The strongest extreme-MSE improvement occurred at horizon 13 (3.87%). The largest mean overall MSE improvement occurred during Days 15-21 (1.19%), while the largest mean extreme-MSE improvement occurred during Days 15-21 (2.50%).

Positive-extreme MSE improved on 26 horizons, whereas negative-extreme MSE improved on only 8. MAE improved on 0 horizons and WAPE on 0. This reinforces the aggregate finding that EA-TTM mainly reduces squared error in the positive tail rather than uniformly improving absolute error.

The descriptive correlation between horizon-specific extreme prevalence and extreme-MSE improvement was 0.40. This is treated as association only, not causation.

## 8. Overall interpretation

The evidence supports a qualified conclusion. Pre-trained TTM transferred strongly to retail forecasting, head probing added further retail adaptation, and the locked EA-TTM produced a modest but broadly distributed reduction in squared error with stronger gains on positive extreme-demand errors. The gate learned meaningful extreme-regime behavior, but the ungated cross-feed ablation achieved lower overall and extreme MSE. The contribution is therefore best framed as evidence for lightweight decoder cross-feed and retail-aware residual adaptation under tight compute constraints, rather than universal metric superiority or proof that multiplicative gating is uniquely optimal.