# The Builder-Worker Paradox: A Mixed-Methods Analysis of Affective Dissonance in Agentic AI Discourse

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18238765.svg)](https://doi.org/10.5281/zenodo.18238765)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

This repository contains the official source code and dataset for the research paper **"The builder-worker paradox: A mixed-methods analysis of affective dissonance in agentic AI discourse"**, submitted to *Royal Society Open Science*.

## 📄 Abstract
This study investigates the divergence in public discourse regarding "Agentic AI." By analyzing **19,250 Reddit posts** from late 2024 to late 2025, we identify a "Builder-Worker Paradox," where technical practitioners exhibit neutral frustration regarding operational friction, while the general workforce exhibits high levels of existential anxiety. The study employs a mixed-methods approach combining **BERTopic** modeling, **GoEmotions** affective analysis (RoBERTa), and qualitative coding.

## 👥 Authors
* **Umar Hasan** (North South University)
* **Hossain Md. Shahriar** (North South University)
* **S.M. Shah Nawaz Hossain** (North South University)
* **Snahashis Sarker Arnob** (North South University)
* **Sifat Momen** (North South University)

---

## 📂 Repository Structure

```text
├── data/
│   └── dataset.zip            # Raw scraped data (CC BY 4.0)
├── src/
│   ├── scraper.py             # PRAW-based script for data collection
│   ├── topic_modeling.py      # BERTopic implementation & visualization, after data preprocessing
│   ├── emotion_analysis.py    # GoEmotions (RoBERTa) classification
│   └── statistical_tests.py   # Mann-Whitney U & significance testing
├── ablation/
│   ├── random_noise.py        # Ablation study: Random noise validation
│   ├── sentiment_baseline.py  # Ablation study: Sentiment baseline comparison
│   └── lexical_masking.py     # Ablation study: Lexical masking test
├── requirements.txt           
├── LICENSE                    # GNU GPLv3 License text
└── README.md                  # Project documentation

```

---

## 💾 Data Availability

The complete dataset used in this study (N=19,250 posts) is archived and publicly available on Zenodo with a permanent Digital Object Identifier (DOI).

* **Dataset DOI:** [10.5281/zenodo.18238765](https://doi.org/10.5281/zenodo.18238765)
* **Format:** `.csv` (UTF-8 encoded)
* **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

A zipped copy is also included in `data/dataset.zip` for immediate replication convenience.

---

## 📊 Citation

If you use this code or dataset in your research, please cite the following:

**Dataset:**

```bibtex
@dataset{hasan_2026_18238765,
  author       = {Hasan, Umar and Shahriar, Hossain Md. and Hossain, S.M. Shah Nawaz and Arnob, Snahashis Sarker and Momen, Sifat},
  title        = {Dataset for: "The builder-worker paradox: A mixed-methods analysis of affective dissonance in agentic AI discourse"},
  month        = feb,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.18238765},
  url          = {[https://doi.org/10.5281/zenodo.18238765](https://doi.org/10.5281/zenodo.18238765)}
}

```

**Article (Preprint/Submitted):**

> Hasan, U. et al. (2026). *The builder-worker paradox: A mixed-methods analysis of affective dissonance in agentic AI discourse*. Royal Society Open Science (Under Review).

---

## ⚖️ License

This project is dual-licensed:

1. **Source Code:** All files in the `src/` and `ablation/` directories are licensed under the **GNU General Public License v3.0 (GPLv3)**. You may copy, distribute, and modify the software as long as you track changes and release modifications under GPLv3.
2. **Dataset:** The dataset located in `data/` is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)**. You are free to share and adapt the data as long as appropriate credit is given.

```
