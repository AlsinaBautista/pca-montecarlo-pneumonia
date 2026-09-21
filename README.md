# PCA & Monte Carlo on PneumoniaMNIST

PCA dimensionality reduction and Monte Carlo robustness analysis of a pneumonia X-ray classifier (PneumoniaMNIST).

> Project developed for **I206 – Inferencia y Estimación** (Ingeniería en Inteligencia Artificial, Universidad de San Andrés, 2026).

---

## Overview

Chest X-ray images live in a very high-dimensional space: each 128×128 image is a vector of 16,384 pixel values, many of them highly correlated. This project explores two questions:

1. **How much can the data be compressed?** Using Principal Component Analysis (PCA), each image is projected onto a small number *K* of principal components. A logistic regression classifier (healthy vs. pneumonia) is used as a benchmark to measure how much useful information survives the compression.
2. **How robust is the resulting system?** Some test images may arrive with the wrong orientation. A Monte Carlo simulation randomly rotates test images by 180° with probability *p* and estimates how the classifier's performance degrades.

Finally, the Monte Carlo estimators are justified theoretically through the **Law of Large Numbers**.

## Dataset

The images come from [PneumoniaMNIST](https://medmnist.com/) (MedMNIST+, 128×128 version): grayscale chest X-rays labeled as

| Label | Class |
|---|---|
| `0` | Normal |
| `1` | Pneumonia |

| Split | Images | Class balance |
|---|---|---|
| Train | 2,428 | 1,214 / 1,214 |
| Test | 468 | 234 / 234 |

The dataset is **not included** in this repository. To run the notebook, place it in the project root with this structure:

```
dataset_tp1/
├── train/              # PMINST_0001.png, ...
├── test/               # PMINST_2429.png, ...
├── train_labels.csv
└── test_labels.csv
```

## Methods

### Principal Component Analysis (SVD-based)

- Images are flattened into row vectors to build the data matrix **X** ∈ ℝ^(n×m).
- The mean is estimated and removed using the **training set only**.
- PCA is computed via the Singular Value Decomposition of the centered training matrix, which avoids building the 16,384 × 16,384 covariance matrix.
- Both training and test data are projected onto the first *K* principal directions learned from the training set.

### Classification

A logistic regression model (`scikit-learn`, default settings) is trained on the full images and on PCA projections for K=1,…,300. Performance is measured with accuracy on the test set. This sweep is descriptive; selecting K for future use would require validation within the training set.

The consigna's example uses `max_iter=2000`. With that value, 21 of the 300 fits in the sweep exhaust the iterations without reaching the tolerance and scikit-learn raises a `ConvergenceWarning` (K between 204 and 229, all but 205, 206, 212, 226 and 227). Raising the limit reduces that number but never to zero: seven fits (K = 218–224) keep failing to converge no matter how high the limit is set, including values far above the one used here. The cause was not identified; fixing it would mean using other tools, which is outside the scope of this work. The sweep therefore uses `max_iter=20000`, which leaves only those seven cases. Against equivalent fits that do converge, four of the seven give exactly the same accuracy and the other three differ by 2, 2 and 6 images out of 468 (K = 221, 222, 223). The full-image fit converges in 89 iterations and the exercise-2 model (K = 2) in 11, so neither depends on this limit. No K reported in the report's tables or conclusions falls between 218 and 224. Exercise 1 uses pixel values in [0,255]; exercise 2 uses [0,1]. The latter changes the scale relative to the regularization penalty, even though the observed unperturbed K=2 accuracy agrees.

### Monte Carlo robustness analysis

With the PCA (*K* = 2) + logistic regression pipeline trained once on unperturbed data:

- In each simulation, every test image is independently rotated 180° with probability *p*.
- The accuracy *A_p* and the loss *L(p) = A_0 − A_p* are recorded.
- Over *N_MC* independent simulations, the following are estimated as functions of *p*:
  - the expected accuracy **E[A_p]**,
  - the probability **P(L(p) > δ)** with δ = 0.1,
  - the empirical distribution of *A_p* (histograms).

## Repository structure

```
.
├── notebooks/
│   └── TP1_pca_montecarlo.ipynb   # Exercise 1 (PCA, accuracy vs. K) + Exercise 2 (Monte Carlo)
├── report/                  # TP1_G01.tex and TP1_G01.pdf
├── figures/                 # Plots used in the report and this README
├── requirements.txt
└── README.md
```

## Getting started

```bash
git clone https://github.com/AlsinaBautista/pca-montecarlo-pneumonia.git
cd pca-montecarlo-pneumonia
pip install -r requirements.txt
jupyter notebook notebooks/TP1_pca_montecarlo.ipynb
```

Use Python 3.14 and the pinned versions in `requirements.txt`. Run the notebook **from the first cell to the last** (Restart Kernel and Run All) from the `notebooks/` directory; it reads `../dataset_tp1/` and writes the plots to `../figures/`. With the supplied dataset, the complete execution may take a few minutes, depending on the computer.

The notebook regenerates all the figures. It also checks that the fast Monte Carlo predictions match rotating, projecting and predicting directly, and compares the estimated means and standard deviations against their analytical expressions. Zero observed events are reported as a simulation frequency, not as proof of zero probability.

To compile the report after verifying that its discussion matches the numerical results:

```bash
cd report
pdflatex -interaction=nonstopmode -halt-on-error TP1_G01.tex
pdflatex -interaction=nonstopmode -halt-on-error TP1_G01.tex
```

The submission archive is `TP1_G01.zip`; it contains the final PDF, the executed notebook, the requirements and this README. It excludes the dataset, Python environments and Git metadata. The notebook recreates the figures when run with the separately supplied dataset.

## Results

### Test set in the first two principal components

![PCA scatter](figures/pca_scatter.png)

### Accuracy vs. number of components

![Accuracy vs K](figures/accuracy_vs_k.png)

### Effect of random rotations on the PCA space

![Perturbed scatters](figures/perturbed_scatter.png)

### Expected accuracy vs. rotation probability

![E[A_p]](figures/expected_accuracy_vs_p.png)

### Probability of exceeding the loss tolerance

![P(L > δ)](figures/loss_probability_vs_p.png)

### Accuracy distributions

![Histograms, p = 0.1, 0.3, 0.5, 0.7, 0.9](figures/accuracy_histograms.png)

### Stability of the Monte Carlo estimates

![Running estimates for p = 0.35](figures/monte_carlo_convergence.png)

The curves show how the estimated mean accuracy and loss probability change as more simulations are included.

## Authors

- Bautista Alsina
- Santos Bunge
- Hans Dietrich

## References

- Yang, J. et al. (2024). *MedMNIST+*: PneumoniaMNIST (128×128).
- Leon-Garcia, A. (2008). *Probability, Statistics, and Random Processes for Electrical Engineering* (3rd ed.). Prentice Hall.
- Mahmood, B. (2016). *How to Predict Yes/No Outcomes Using Logistic Regression.* Medium.
