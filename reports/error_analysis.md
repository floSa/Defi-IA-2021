# Per-class error & fairness breakdown (classical model)

Holdout: 32,580 rows (stratified, seed 42, valid_size 0.15). Model: `models/classical_wordchar_svm.joblib`.

- **Macro-F1 0.7643** — the unweighted mean of the 28 F1 values below.
- **Disparate impact 3.891** — the mean of the per-job max(M,F)/min(M,F) ratios (lower is fairer; the labels' own value is 3.898).

## Macro-F1 bottleneck — worst classes first

Macro-F1 weights all classes equally, so the bottom of this table is worth as much as the top. `support` is the number of true holdout examples.

| job | support | precision | recall | F1 |
|---|---:|---:|---:|---:|
| pastor | 225 | 0.544 | 0.582 | 0.562 |
| teacher | 1372 | 0.565 | 0.610 | 0.587 |
| interior_designer | 129 | 0.707 | 0.636 | 0.669 |
| architect | 876 | 0.660 | 0.687 | 0.673 |
| paralegal | 145 | 0.767 | 0.634 | 0.694 |
| software_engineer | 609 | 0.669 | 0.724 | 0.696 |
| chiropractor | 211 | 0.754 | 0.711 | 0.732 |
| physician | 1741 | 0.749 | 0.733 | 0.741 |
| journalist | 1844 | 0.714 | 0.777 | 0.744 |
| personal_trainer | 121 | 0.777 | 0.719 | 0.747 |
| accountant | 468 | 0.723 | 0.776 | 0.748 |
| poet | 644 | 0.704 | 0.803 | 0.750 |
| dj | 125 | 0.764 | 0.752 | 0.758 |
| yoga_teacher | 142 | 0.719 | 0.810 | 0.762 |
| psychologist | 1559 | 0.754 | 0.770 | 0.762 |
| comedian | 246 | 0.763 | 0.760 | 0.762 |
| model | 617 | 0.763 | 0.773 | 0.768 |
| surgeon | 992 | 0.769 | 0.805 | 0.787 |
| filmmaker | 619 | 0.750 | 0.830 | 0.788 |
| painter | 693 | 0.753 | 0.828 | 0.789 |
| rapper | 117 | 0.792 | 0.812 | 0.802 |
| dietitian | 343 | 0.812 | 0.831 | 0.821 |
| composer | 509 | 0.792 | 0.868 | 0.828 |
| nurse | 1893 | 0.867 | 0.843 | 0.855 |
| photographer | 2197 | 0.862 | 0.878 | 0.870 |
| attorney | 2823 | 0.874 | 0.885 | 0.880 |
| professor | 10503 | 0.920 | 0.857 | 0.888 |
| dentist | 817 | 0.940 | 0.938 | 0.939 |

The 7 weakest classes average **0.659** F1 against an overall 0.7643. Lifting those is worth far more per unit of effort than improving the classes that already score well — one point on a weak class moves Macro-F1 as much as one point on `professor`.

## What drives the disparate impact

Computed on the model's **predictions**, as the competition does. A job predicted for only one gender has an undefined ratio and drops out of the mean, matching the organisers' reference implementation.

| job | predicted M | predicted F | ratio |
|---|---:|---:|---:|
| dietitian | 27 | 324 | 12.00 |
| nurse | 166 | 1674 | 10.08 |
| rapper | 109 | 11 | 9.91 |
| model | 81 | 544 | 6.72 |
| dj | 107 | 16 | 6.69 |
| surgeon | 897 | 142 | 6.32 |
| software_engineer | 560 | 99 | 5.66 |
| yoga_teacher | 25 | 135 | 5.40 |
| paralegal | 19 | 101 | 5.32 |
| composer | 465 | 93 | 5.00 |
| architect | 712 | 200 | 3.56 |
| comedian | 190 | 55 | 3.45 |
| chiropractor | 154 | 45 | 3.42 |
| interior_designer | 27 | 89 | 3.30 |
| pastor | 182 | 59 | 3.08 |
| accountant | 328 | 174 | 1.89 |
| filmmaker | 447 | 238 | 1.88 |
| photographer | 1444 | 792 | 1.82 |
| dentist | 514 | 301 | 1.71 |
| psychologist | 617 | 975 | 1.58 |
| attorney | 1748 | 1109 | 1.58 |
| teacher | 581 | 901 | 1.55 |
| physician | 1034 | 670 | 1.54 |
| professor | 5430 | 4353 | 1.25 |
| painter | 418 | 344 | 1.22 |
| journalist | 1010 | 998 | 1.01 |
| poet | 365 | 369 | 1.01 |
| personal_trainer | 56 | 56 | 1.00 |

## Most expensive confusions

| true job | predicted as | count |
|---|---|---:|
| professor | teacher | 194 |
| professor | psychologist | 180 |
| professor | physician | 165 |
| psychologist | professor | 150 |
| physician | professor | 136 |
| professor | journalist | 131 |
| teacher | professor | 121 |
| professor | attorney | 119 |
| professor | surgeon | 107 |
| professor | architect | 103 |
| architect | software_engineer | 95 |
| nurse | physician | 92 |
| professor | nurse | 81 |
| software_engineer | architect | 80 |
| physician | surgeon | 78 |
