# Credit Card Customer Behavior — Predicting `Credit_Score`

End-to-end analysis of a [synthetic credit-card customer dataset](https://www.kaggle.com/datasets/dharmendrapandit12/credit-card-customer-behavior-segmentation-dataset/) (50,000 customers, 30 columns). The notebook recovers the data-generating process for `Credit_Score`, ranks every feature on a single comparable scale, and shows that a **7-term linear regression** matches — and slightly beats — gradient boosting on 28 features.

## Dataset

| | |
| --- | --- |
| Source | [Kaggle — Credit Card Customer Behavior & Segmentation](https://www.kaggle.com/datasets/dharmendrapandit12/credit-card-customer-behavior-segmentation-dataset/) |
| File | `synthetic_credit_card_customer_behavior_dataset.csv` (**not tracked** — download from Kaggle) |
| Size | 50,000 rows × 30 columns |
| Target | `Credit_Score` (integer, FICO-like scale 367–792) |
| Quality | No missing values, no duplicates, one row per customer |

Download the CSV from the [dataset page](https://www.kaggle.com/datasets/dharmendrapandit12/credit-card-customer-behavior-segmentation-dataset/) and place it next to `notebook.ipynb`, or run the notebook on Kaggle — the loader resolves both paths automatically.

## What the notebook does

1. **Data overview** — integrity checks, target distribution, scale of effects.
2. **Algebraic redundancy audit** — finds exact identities (`Monthly_Spending` = sum of categories, `Payment_Ratio` = payment / statement, …) before looking at correlations.
3. **Numerical features vs target** — Pearson / Spearman, decile plots; only two features dominate marginally.
4. **Categorical features vs target** — boxplots and correlation ratio $\eta^2$, ranked on the same scale as $r^2$.
5. **Unified feature ranking** — all 29 features ordered by explained variance (+ mutual information).
6. **Reverse-engineering the generator** — recovers the formula, including five secondary effects that marginal correlation understates by a factor of $\sim 4.6$.
7. **Modeling** — Dummy, Ridge, and `HistGradientBoostingRegressor` across five feature sets, with permutation importance and partial dependence.
8. **Conclusion** — methodological takeaways.

## Main result

$$
\begin{aligned}
\texttt{Credit\_Score} = \;& 537.9 + 220.0\,\texttt{Payment\_Ratio} - 200.3\,\texttt{Credit\_Utilization} \\
& + 0.124\,\texttt{Card\_Age\_Months} + 1.54\,\texttt{International\_Transactions} \\
& - 10.03\,\mathbb{1}[\texttt{Cash\_Advance} > 0] + 0.287\,\texttt{Mobile\_App\_Login} - 3.90\,\texttt{EMI\_Count}
\end{aligned}
$$

| Model | Features | $R^2$ | RMSE | MAE |
| --- | ---: | ---: | ---: | ---: |
| Dummy (mean) | — | 0.000 | 68.77 | 55.61 |
| **Ridge [formula]** | **7** | **0.968** | **12.35** | **9.88** |
| HistGBR [full] | 28 | 0.967 | 12.48 | 9.98 |
| Ridge [key] (2 ratios only) | 2 | 0.952 | 15.10 | 11.99 |

Irreducible noise floor: $\sigma_\varepsilon \approx 12.3$ score points. The 7-term Ridge sits 0.07 RMSE points above it.

**What drives the score:** how much of the bill the customer pays and how much of the limit they use (~95% of variance). Five behavioral features add a further ~2%. Income, card tier, occupation, gender, and the spending mix add essentially nothing once the above are known.

## Requirements

```text
python >= 3.10
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn >= 1.4
nbformat / jupyter   # only if you re-execute the notebook from the CLI
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

## How to run

Open `notebook.ipynb` in Jupyter, VS Code, or Cursor and run all cells top to bottom (~30 s on a laptop).

From the CLI (optional):

```bash
python -c "import nbformat; from nbclient import NotebookClient; \
nb = nbformat.read('notebook.ipynb', as_version=4); \
NotebookClient(nb, timeout=1200, kernel_name='python3').execute(); \
nbformat.write(nb, 'notebook.ipynb')"
```

## Project layout

```text
.
├── README.md
├── .gitignore
├── notebook.ipynb
└── synthetic_credit_card_customer_behavior_dataset.csv   # local only (gitignored)
```

## Methodological highlights

- **$\eta^2$ for categoricals, $r^2$ for numericals** — one ranking table for all 29 features.
- **Effect size over p-values** — at $n = 50{,}000$, ANOVA flags card type as “highly significant” while it explains 0.2% of variance.
- **Mine residuals, don’t stop at normality** — the 2-variable model’s residuals passed Shapiro–Wilk ($p = 0.63$) and still hid five recoverable effects.
- **Do not prune by correlation threshold** — remove algebraic identities; keep collinear but informative features and use Ridge / permutation importance.
- **Trees cannot do arithmetic** — hiding the two ratios but keeping their numerators and denominators costs ~38% RMSE; engineer ratios explicitly.
- **`partial_dependence(..., method="brute")`** — with `categorical_features="from_dtype"`, the default `"recursion"` silently attaches curves to the wrong features.

## License

This project is for educational / analytical purposes. The underlying dataset is published on Kaggle under the terms set by its author; check the [dataset page](https://www.kaggle.com/datasets/dharmendrapandit12/credit-card-customer-behavior-segmentation-dataset/) for usage rights.
