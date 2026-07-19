# Per-class: transformer vs classical

Shared holdout of 32,580 rows. Transformer run: `roberta_large_6ep`.

- classical Macro-F1   **0.7643**
- transformer Macro-F1 **0.8061**  (+0.0417)

## Testing the prediction from the error analysis

The claim was that the transformer's gain concentrates in the classes the bag-of-words confuses with their semantic neighbours, not in the rare ones.

- mean gain on the 7 predicted-weak classes : **+0.0530**
- mean gain on the other 21 classes         : **+0.0380**

**Prediction holds** (×1.4). The two models fail on different classes, so the ensemble has genuine diversity to exploit — blend it.

## Per class, biggest gain first

| job | support | classical | transformer | Δ |
|---|---:|---:|---:|---:|
| pastor ⭐ | 225 | 0.562 | 0.695 | +0.133 |
| dj | 125 | 0.758 | 0.861 | +0.103 |
| rapper | 117 | 0.802 | 0.889 | +0.087 |
| comedian | 246 | 0.762 | 0.847 | +0.085 |
| filmmaker | 619 | 0.788 | 0.865 | +0.077 |
| model | 617 | 0.768 | 0.845 | +0.077 |
| software_engineer ⭐ | 609 | 0.696 | 0.769 | +0.073 |
| composer | 509 | 0.828 | 0.899 | +0.071 |
| journalist | 1844 | 0.744 | 0.810 | +0.066 |
| poet | 644 | 0.750 | 0.803 | +0.052 |
| teacher ⭐ | 1372 | 0.587 | 0.636 | +0.049 |
| architect ⭐ | 876 | 0.673 | 0.719 | +0.046 |
| personal_trainer | 121 | 0.747 | 0.788 | +0.041 |
| yoga_teacher | 142 | 0.762 | 0.800 | +0.038 |
| photographer | 2197 | 0.870 | 0.907 | +0.037 |
| interior_designer ⭐ | 129 | 0.669 | 0.705 | +0.036 |
| paralegal ⭐ | 145 | 0.694 | 0.728 | +0.033 |
| painter | 693 | 0.789 | 0.821 | +0.032 |
| attorney | 2823 | 0.880 | 0.909 | +0.030 |
| psychologist | 1559 | 0.762 | 0.780 | +0.018 |
| accountant | 468 | 0.748 | 0.763 | +0.014 |
| dietitian | 343 | 0.821 | 0.833 | +0.012 |
| chiropractor ⭐ | 211 | 0.732 | 0.732 | +0.001 |
| nurse | 1893 | 0.855 | 0.853 | -0.002 |
| physician | 1741 | 0.741 | 0.738 | -0.004 |
| dentist | 817 | 0.939 | 0.934 | -0.004 |
| professor | 10503 | 0.888 | 0.873 | -0.015 |
| surgeon | 992 | 0.787 | 0.768 | -0.019 |

⭐ = flagged as a bottleneck by `reports/error_analysis.md`.

Holdout built with seed 42 (32,580 rows).
