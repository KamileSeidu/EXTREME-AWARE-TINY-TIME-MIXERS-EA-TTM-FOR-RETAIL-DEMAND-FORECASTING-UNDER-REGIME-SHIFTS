# EA-TTM Limitations

## L01

Evaluation uses one retail dataset and one locked 28-day final-test interval.

**Implication:** Generalization to other retailers, countries, hierarchies, and horizons is not established.

## L02

Primary EA-TTM improves MSE but worsens MAE and WAPE relative to HP-TTM.

**Implication:** The gain is objective-specific rather than universal across error metrics.

## L03

Negative extremes are rare and primary EA-TTM worsens negative-extreme MSE.

**Implication:** Evidence is stronger for positive demand spikes than negative shocks.

## L04

The ungated cross-feed ablation outperforms the gated primary model on overall and extreme MSE.

**Implication:** The gate detects extremes but its multiplicative routing can suppress useful corrections.

## L05

The extreme label is a training-derived robust statistical rule.

**Implication:** Alternative domain definitions of retail shocks could produce different labels and outcomes.

## L06

The future-known covariate set is deliberately compact.

**Implication:** Richer promotions, inventory, weather, macroeconomic, or event information may add signal.

## L07

The paired bootstrap resamples store-item series.

**Implication:** The interval is a robustness diagnostic under that resampling scheme, not a model of every dependence structure.
