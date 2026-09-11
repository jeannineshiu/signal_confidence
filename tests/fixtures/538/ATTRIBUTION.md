# FiveThirtyEight forecast data (test fixtures)

`nba_games.csv` and `nfl_games.csv` are copied **unmodified** from
[fivethirtyeight/checking-our-work-data](https://github.com/fivethirtyeight/checking-our-work-data)
at commit `f6d5b2e1d6da` (2023-02-02). They are the real pre-game win probabilities FiveThirtyEight
published, together with the actual results.

Licence: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/),
© FiveThirtyEight / ABC News.

These files are used only to validate the evaluation harness on real probabilistic forecasts. They are
not part of the project's results.
- Hand-computed Brier and ECE on 5 NFL games
- Cross-checks against scikit-learn on all NBA games
- The Brier identity between the two-sided and favourite forms
