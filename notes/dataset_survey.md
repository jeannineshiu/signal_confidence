# Dataset survey (2026-09-11)

Requirement: **only real data**, with nothing synthetic, for the headlines and for validating the
evaluation code.

## A. Headline datasets

| Dataset | Period | Licence | Verdict |
|---|---|---|---|
| **Kaggle `frankossai/apple-stock-aapl-historical-financial-news-data`** (mostly Yahoo Finance) | 2016-02 → 2024-11-27, UTC timestamps | CC0 (as listed) | ✅ **Chosen.** 2,501 eligible Apple headlines from 2023-11 onward, all after gpt-4o-mini's 2023-10-01 training cutoff |
| Kaggle `miguelaenlle/...nlpbacktests` (Benzinga) | 2009–2020, minute ET timestamps | CC0 (as listed) | Real, but entirely inside the LLM's training window. Used in the first Phase 1 build, then replaced |
| Kaggle `rdolphin/financial-news-with-ticker-level-sentiment` (Polygon) | 2023 only | MIT | ❌ About 100 articles per ticker, mostly before the cutoff |
| Kaggle `pratyushpuri/financial-news-market-events-dataset-2025` | 2025 | Apache 2.0 | ❌ **Synthetic**: its description says so |
| Kaggle `ibktommy/aggregated-financial-news-dataset` | 2009–2020 | ODC-By | ❌ A reprocessing of the Benzinga dataset |
| HF `KrossKinetic/SP500-Financial-News-Articles-Time-Series` (CNBC) | 2023 | MIT | ❌ Inside the training window, no exact times |
| HF `m-ric/financial-news-2024` | 2024 | none stated | ❌ No ticker, no licence |
| HF `Zihan1004/FNSPID` | 1999–2023 | CC BY-NC 4.0 | ❌ Non-commercial, a 5.7 GB single CSV, inside the training window |
| GitHub `yumoxu/stocknet-dataset` | 2014–2016 tweets | MIT | ❌ Tweets rather than news, inside the training window |

## B. Real forecasts for validating the evaluation code (replaces synthetic test data)

| Dataset | Content | Licence | Use |
|---|---|---|---|
| **GitHub `fivethirtyeight/checking-our-work-data`** | Published pre-game win probabilities and results (NBA 8.9k games, NFL, MLB …) | CC BY 4.0 | ✅ Phase 2: a hand-computed fixture from real rows, plus a cross-check against scikit-learn on the full real file |
| GitHub `forecastingresearch/forecastbench-datasets` | Real LLM probability forecasts and resolutions | CC BY-SA 4.0 | Optional |
| Harvard Dataverse `gjp` (Good Judgment Project) | Millions of human geopolitical forecasts | Dataverse terms | Not needed |

538's own calibration pages were rendered client-side, and 538 was shut down in 2025. Their method also
differs from ours: 0%/100% forecasts were removed and updates weighted. So the reference for validation
is an independent implementation (scikit-learn) run on the same real rows, not 538's published figures.

## Sources
- https://platform.openai.com/docs/models/gpt-4o-mini (knowledge cutoff 2023-10-01)
- https://www.kaggle.com/datasets/frankossai/apple-stock-aapl-historical-financial-news-data
- https://github.com/fivethirtyeight/checking-our-work-data
- https://github.com/forecastingresearch/forecastbench-datasets
- https://huggingface.co/datasets/Zihan1004/FNSPID
- https://github.com/yumoxu/stocknet-dataset
