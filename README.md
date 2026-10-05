# Auditing Geographic Domain Shift in Genomic AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.placeholder.svg)](https://zenodo.org)

Official code repository for the manuscript:  
**"Auditing Geographic Domain Shift in Genomic AI: A Memory-Efficient Pipeline for Tuberculosis Drug Resistance Prediction"** (Mohammad Kawsar).

---

## Overview

Applying machine learning to whole-genome sequencing (WGS) facilitates rapid drug susceptibility testing (DST) for *Mycobacterium tuberculosis* (MTB). However, models deployed across borders frequently suffer from dataset shift. 

This repository provides an end-to-end, memory-efficient data processing pipeline that:
1. Streams high-volume vertical genomic variant records in low-memory batches.
2. Formats a patient-level genomic feature matrix downcasted to 1-byte integers (`int8`).
3. Evaluates geographic generalizability using XGBoost across disparate international laboratory sites from the CRyPTIC dataset compendium.
4. Audits biological interpretability using TreeExplainer SHAP (SHapley Additive exPlanations) to verify that decisions depend on valid mutational coordinates rather than geographic artifacts.

### Key Finding: Reverse Domain Shift
The evaluated XGBoost model achieved **74.92% accuracy** on the source training domain, but generalized with **85.85% accuracy** on the unseen international target domain. This demonstrates that models can yield superior performance when transferring to populations with lower intrinsic genetic variance.

---

## Repository Structure

```text
├── data/                  # Directory for CRyPTIC data files (git-ignored)
├── figures/               # Output directory for SHAP audit plots
├── pipeline.py            # Complete end-to-end extraction, training, & audit script
├── CITATION.cff           # Citation metadata file for Zenodo DOI integration
├── requirements.txt       # Environment dependencies
├── LICENSE                # MIT Open Source License
└── README.md              # Project documentation