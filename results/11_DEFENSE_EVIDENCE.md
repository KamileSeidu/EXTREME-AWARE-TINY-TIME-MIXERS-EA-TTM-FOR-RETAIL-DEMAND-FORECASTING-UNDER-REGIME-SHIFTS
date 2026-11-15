# EA-TTM Defense Evidence

## Experimental progression

The project progressed from a weekly seasonal-naive benchmark to zero-shot TTM, then to a retail-adapted head-probed TTM, and finally to the supervisor-approved EA-TTM residual adapter. Model development was locked before the final test was opened.

## Final primary result

The locked primary EA-TTM achieved final-test MSE 4.528610 compared with 4.563047 for frozen HP-TTM.

This corresponds to a 0.755% MSE reduction.

Extreme-regime MSE improved by 1.859%, while positive-extreme MSE improved by 2.129%.

The result is not uniformly better across all metrics: MAE and WAPE each worsen by about 0.876%, normal-regime MSE worsens by 0.732%, and negative-extreme MSE worsens by 1.925%.

## Gate mechanism

The final-test extreme prevalence is 0.0593. The gate achieves AUROC 0.7677 and AUPRC 0.2012, equal to 3.39 times the prevalence.

The mean absolute effective correction is 2.29 times larger on extreme than normal observations. The correction direction agrees with the required residual direction on 73.89% of extreme observations.

## Ablation interpretation

The ungated cross-feed adapter achieved final-test MSE 4.513243, lower than the locked primary EA-TTM. This indicates that the decoder cross-feed and residual pathway are useful, but the multiplicative gate is not automatically beneficial for every forecasting objective.

The primary model was not changed after observing this result because model development and checkpoint selection had already been locked.

## Bootstrap evidence

The observed paired series-level MSE difference (HP-TTM minus EA-TTM) is 0.034437. The 95% paired bootstrap interval is [0.029668, 0.039819].

All 2000 bootstrap replicates produced a positive mean difference under this series-level resampling procedure.

## Core defense conclusion

EA-TTM should be presented as a lightweight extreme-aware residual adaptation that provides a modest but robust squared-error improvement over the frozen head-probed TTM, with stronger gains on positive extreme-demand observations.

It should not be presented as universally superior across every metric or as proof that the gate is the only beneficial architectural component.