# Paper narrative — scientific decisions

_Fill this in as decisions are made. One section per topic.
Dates help track when a decision was settled._

---

## Framing and objective

<!-- What is the paper about, in one paragraph.
     What is NOT the paper about (scope limits). -->

---

## Key results

<!-- The 3-5 results that carry the paper.
     One bullet per result, with the number and its source experiment. -->

---

## Onset skill

<!-- Why onset skill matters, not overall r.
     Table of onset r per model / phase.
     Interpretation: what does TbotAtm beating persistence at onset imply? -->

---

## Partition: local vs remote drivers

<!-- Partition table (MSE, kfold, 5 folds).
     Interpretation of local_only > full.
     What remote_only onset skill implies for the narrative. -->

---

## XAI methodology

<!-- Which IG flavor (signed vs unsigned, per-year vs pooled).
     What Hobday decomposition adds.
     What GradCAM adds (or why it's not in the paper). -->

---

## Granger methodology

<!-- Disclaimer on F-stat comparability across variables.
     What Granger is used for (causal direction only, not ranking).
     How it complements IG (short-lag coupling vs multi-week preconditioning).
     Whether ΔR² is included. -->

---

## Window size and lead time

<!-- Why 60-day window (link to ACF / tau_ns).
     Why lead=7d (operational relevance + skill curve).
     Lead sweep results if included. -->

---

## Architecture choices

<!-- Why LSTM-only (vs attention, TCN, ConvLSTM).
     Why MSE (vs GNLL — include GNLL as supplementary or drop).
     K-fold CV: why 5 folds, how splits were made. -->

---

## Open scientific questions

<!-- Things not settled yet. Remove entries when resolved. -->
