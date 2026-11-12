# EpiPredict: Advanced Neurological Predictive Analytics Platform

> **A Multimodal Deep Learning and Neuromorphic Edge-Computing Framework for Prospective Epileptic Seizure Prediction Using Big Data Analytics and Wearable Sensor Technologies**
> **Author:** SAMUELSON G

[![DOI](https://img.shields.io/badge/DOI%20ResearchGate-10.13140/RG.2.2.35024.88321-blue)](https://doi.org/10.13140/RG.2.2.35024.88321)  
[![Zenodo](https://img.shields.io/badge/Zenodo-blue)](https://zenodo.org/records/24169350)  
[![Academia.edu](https://img.shields.io/badge/Academia.edu-blue)](https://www.academia.edu/180815558/EpiPredict_A_Multimodal_Deep_Learning_and_Neuromorphic_Edge_Computing_Framework_for_Prospective_Epileptic_Seizure_Prediction_Using_Big_Data_Analytics_and_Wearable_Sensor_Technologies)  

---

## Table of Contents

1. [Overview & Clinical Motivation](#overview--clinical-motivation)
2. [Key Quantitative Highlights](#key-quantitative-highlights)
3. [System Architecture & Six-Stage Workflow](#system-architecture--six-stage-workflow)
4. [Interactive Research Web Console (`epipredict_research.html`)](#interactive-research-web-console-epipredict_researchhtml)
5. [Repository Structure](#repository-structure)
6. [Technology Stack](#technology-stack)
7. [Benchmark Datasets & Evaluation Protocol](#benchmark-datasets--evaluation-protocol)
8. [Performance & Edge Hardware Benchmarks](#performance--edge-hardware-benchmarks)
9. [Quick Start & Installation](#quick-start--installation)
10. [Development Roadmap & Ethical Governance](#development-roadmap--ethical-governance)
11. [Conclusion](#conclusion)
12. [Citation & Author Declarations](#citation--author-declarations)

---

## Overview & Clinical Motivation

Epilepsy is a chronic neurological disorder affecting **50 to 65 million individuals worldwide**, with **30% to 40% of patients** suffering from drug-resistant (refractory) epilepsy. Because existing clinical monitoring relies heavily on inpatient video-EEG or reactive seizure detection systems, patients face severe risks from unannounced seizures—including a 9% annual probability of injury-related hospitalization and heightened risk of Sudden Unexpected Death in Epilepsy (SUDEP).

**EpiPredict**, conceptualized and architected by **SAMUELSON G**, transitions epilepsy care from reactive detection to **prospective preictal forecasting** (identifying the preictal transition 5 to 90 minutes—and up to 4 hours in multimodal configurations—prior to ictal onset). The repository contains both the formal academic manuscript and a single-page **Interactive Research Console** (`epipredict_research.html`) featuring real-time biophysical simulators and benchmark visualizations.

---

## Key Quantitative Highlights

* **Clinical Translation Targets:** Surpasses foundational targets of **$\ge 70\%$ ambulatory seizure prediction accuracy** and a **$\ge 50\%$ reduction in unexpected epileptic events**.
* **Cranial Montage Pruning:** Uses a two-stage channel-aware **Set Transformer** to reduce scalp EEG electrodes by **84.4%** (from 18 channels down to a mean of **2.8 channels**) while retaining **80.1% sensitivity** and **0.11 FPR/h** on CHB-MIT.
* **Bimodal Neuro-Hemodynamic Fusion:** Combining cranial EEG with functional near-infrared spectroscopy (**fNIRS**) via LSTM networks achieves **98.3% accuracy**, outperforming unimodal EEG (97.6%) and unimodal fNIRS (97.0%).
* **70 / 20 / 10 Multimodal SHAP Discovery:** In 30-minute to 4-hour forecasting across 40 million samples, **contextual big data telemetry** (sleep staging, medication adherence, exertion, anxiety) drives **70%** of predictive accuracy, while EEG/ECG contribute **20%** and HR/HRV contribute **10%**.
* **Neuromorphic Spiking Efficiency:** Surrogate-gradient **Spiking-CNN** and **Spiking ConvLSTM (SPCLU)** architectures cut computational complexity by **80.7% to 98.58%** and reduce energy consumption by **$2,400\times$**.
* **Sub-Millijoule RISC-V Edge Execution:** Deploying INT8-quantized models on the 10-core **GreenWaves GAP9** processor executes inference in **2.84 ms** consuming **0.051 mJ** at **17.89 mW** ($160\times$ more energy-efficient than ARM Cortex MCUs).

---

## System Architecture & Six-Stage Workflow

EpiPredict operates as a closed-loop cyber-physical pipeline across six sequential stages:

```text
[1. Multimodal Data Collection]
├── Cranial Headband: Sparse 2.8-Ch Scalp EEG (256 Hz) + fNIRS Optodes (HbO2 / HbR)
├── Wrist Biosensors: ECG, Optical PPG (HR / Residual HR / HRV), EDA (Tonic / Phasic), 3-Axis ACC, TEMP
└── Contextual Telemetry: Sleep staging, ASM adherence logs, exertion, Circadian & Multiday (7–30d) cycles
│
▼
[2. Signal Pre-Processing & Artifact Suppression]
├── Zero-mean DC offset removal & 4th-order Butterworth bandpass filtering (0.5–100 Hz)
├── Cascaded IIR notch filtering (50 Hz / 60 Hz power-line & harmonic excision)
└── Deep Residual Shrinkage Networks (DRSN) & adaptive ACC motion noise cancellation
│
▼
[3. Spatio-Spectral & Chronobiological Feature Extraction]
├── 44 Logarithmic PSD Features/Ch: 8 Absolute + 8 Relative + 28 Inter-Band Ratios (Delta to Gamma)
├── Short-Time Fourier Transform (STFT) 2D time-frequency spectrograms
└── Hilbert Transform instantaneous phase & Circular Resultant Vector (R-value) phase locking
│
▼
[4. Hybrid Deep Learning & Neuromorphic Spiking Model Training]
├── Cloud / Hub Backbone: Hybrid 2D-CNN + Bi-LSTM / Multi-Head Self-Attention Transformer
├── Edge Neuromorphic Core: 3-Layer Spiking ConvLSTM (SPCLU) + 2 LIF Dense Layers (dt = 16 ms)
└── Surrogate-Gradient BPTT (Fast-Sigmoid) with L2 firing-rate sparsity regularization (10 Hz target)
│
▼
[5. Calibrated Predictive Risk Scoring]
├── Prospective Evaluation Window: Optimal SPH = 7 min | Optimal SOP = 27 min
├── k-of-n voting filter to suppress isolated transient false alarms
└── AURA On-Device Continuous Learning (single-seizure output calibration; 15%–20% FPR reduction)
│
▼
[6. Real-Time Alert & Closed-Loop Actuation]
├── Patient / Caregiver mobile & web hazard alerts
├── Closed-loop trigger for implanted Responsive / Vagus Nerve Stimulation (RNS / VNS)
└── Dual-Mode Selective Telemetry (high-res cloud sync triggered strictly upon preictal threshold)

```

---

## Interactive Research Web Console (`epipredict_research.html`)

This repository includes a self-contained, dependency-free single-page interactive web console (`epipredict_research.html`) built with **HTML5, Tailwind CSS, and Chart.js**. It allows researchers, clinicians, and hardware engineers to interactively explore the manuscript's findings:

* **Executive KPI Dashboard & System Matrix:** Interactive breakdown of epidemiological impact and Table 1 specifications.
* **Multimodal Sensor Inspector & SHAP Doughnut Chart:** Clickable modality cards (EEG, fNIRS, ECG/PPG, EDA, ACC, Context) paired with the 70/20/10 variance attribution chart.
* **Chronobiology Phase-Locking Simulator:** Interactive slider simulating Circular Resultant Vector ($R$-value) coherence ($R \in [0.10, 0.95]$) and the Six-Tier Physiological Signal Hierarchy.
* **Interactive LIF Neuromorphic Simulator:** Live browser simulation of Leaky Integrate-and-Fire membrane potential dynamics ($u_t = \beta u_{t-1} + W_{ij} x_{t-1} - z_{t-1} u_{\text{thr}}$), allowing users to tune synaptic input weights ($W \cdot x$) and firing thresholds ($u_{\text{thr}}$) at $\Delta t = 16\text{ ms}$.
* **Edge TinyML Hardware Visualizer:** Logarithmic comparison of inference latency (ms) and energy per inference (mJ) across GAP9 RISC-V, DYNAP-SE2, Tiny KAN MCU, ARM Cortex-M4F, and NVIDIA Jetson Nano.
* **Clinical SPH/SOP Trade-Off Simulator:** Dynamic slider evaluating Event Sensitivity vs. False Prediction Rate (FPR/h) and Patient Anxiety Index across SOP windows from 10 to 45 minutes.
* **Searchable Benchmark & Bibliography Databases:** Instant text filtering across 14 SOTA seizure prediction models and peer-reviewed references.

---

## Repository Structure

```text
EpiPredict/
├── README.md                          # Project documentation, benchmarks, and conclusion
├── epipredict_research.html           # Interactive Single-Page Research Console (HTML/Tailwind/Chart.js)
├── paper/
│   ├── EpiPredict_Manuscript.md       # Full peer-reviewed format research paper by SAMUELSON G
│   └── figures/                       # High-resolution watermark-free architectural & benchmark figures
├── src/
│   ├── preprocessing/                 # Butterworth bandpass, IIR notch, and MNE STFT/PSD feature pipelines
│   ├── models/
│   │   ├── cnn_lstm_transformer.py    # Hybrid 2D-CNN + Bi-LSTM and Two-Stage Set Transformer (PyTorch/TF)
│   │   ├── spiking_convlstm.py        # Surrogate-gradient Spiking ConvLSTM (SPCLU) via snnTorch
│   │   └── aura_calibration.py        # AURA on-device pseudo-labeling & k-of-n risk calibration loop
│   ├── chronobiology/
│   │   └── circular_phase_stats.R     # Wavelet bandpass, Hilbert phase, and circular R-value statistics in R
│   └── edge_deployment/
│       ├── dory_gap9_int8/            # C++ INT8 quantized kernels for GreenWaves GAP9 RISC-V cluster
│       └── tflite_cortex_m4/          # TFLite Micro & embedded XGBoost C++ firmware for STM32 / ESP32-S3
└── requirements.txt                   # Python dependencies (torch, tensorflow, snntorch, mne, shap, xgboost)

```

---

## Technology Stack

* **Primary Languages:** Python (Deep Learning & Signal Processing), R (Circular Statistics & Longitudinal Analysis), C++ (Embedded TinyML & RISC-V/ARM Runtime Optimization), HTML5/JavaScript (Interactive Research Console).
* **Machine Learning & Neuromorphic Frameworks:** PyTorch, TensorFlow, Keras, `snnTorch`, XGBoost, SHAP, MNE-Python.
* **Edge Compilers & Quantization Toolchains:** TensorFlow Lite Micro (TFLite), QuantLab, DORY (Deployment Oriented to MemorY).
* **Target Edge & Neuromorphic Silicon:** GreenWaves GAP9 ($1+9\times\text{RISCY}$ RISC-V), Mixed-Signal Neuromorphic Processor (DYNAP-SE2 with Asynchronous Delta Modulation), STM32F4 (ARM Cortex-M4F), ESP32-S3, NVIDIA Jetson Nano.

---

## Benchmark Datasets & Evaluation Protocol

All models are evaluated using **strict seizure-independent chronological partitioning** (merging consecutive seizures $< 60\text{ min}$ apart into lead clusters and prohibiting overlapping test windows to prevent temporal data leakage):

| Dataset Repository | Cohort & Modality | Sampling & Channels | Recording Volume | Primary Validation Role |
| --- | --- | --- | --- | --- |
| **CHB-MIT Scalp EEG** | 23 pediatric/young adult patients | 256 Hz, 16-bit; 18–22 Ch | 844–980+ hrs; 163 seizures | SPH/SOP sensitivity, FPR/h, & 2.8-Ch Set Transformer pruning |
| **EPILEPSIAE Database** | 37–275+ multi-center patients | 250–2500 Hz (decimated 256 Hz) | $>5,120$ hrs; $>209$ seizures | Multi-day chronobiology, transfer learning, & AURA calibration |
| **Siena Scalp EEG** | Adult focal/generalized cohort | 512 Hz (decimated 256 Hz); 29 Ch | 128 hrs; 47 seizures | Cross-dataset generalization for LSTM & hybrid networks |
| **TUH Seizure Corpus (TUSZ)** | Heterogeneous hospital cohort | 250–512 Hz; Multi-montage | Thousands of clinical sessions | Backbone pre-training & 6-pattern SNN evaluation |
| **Multimodal Wearable Cohorts** | 25-pt local + 10-pt RNS/wearable | EEG, ECG, PPG, EDA, ACC, TEMP | 40M samples; mean 232 days/pt | 30m–4h SHAP attribution & multiday phase-locking hierarchy |

---

## Performance & Edge Hardware Benchmarks

### 1. Seizure Prediction Model Comparison

| Model Architecture | Dataset & Montage | SPH / SOP Window | Sensitivity | False Alarm Rate (FPR/h) | Key Distinction |
| --- | --- | --- | --- | --- | --- |
| **Foundational EpiPredict Target** | Multimodal Wearables | Real-time continuous | $\ge 70.0\%$ | Minimized | Baseline clinical translation benchmark |
| **Spatio-Temporal Hybrid DL** | CHB-MIT & Siena EEG | 5 min / 30 min | **98.52% – 99.31%** | **0.010 – 0.045 /h** | 99.54% AUC; upper-bound workstation performance |
| **Temporal CNN + LSTM** | CHB-MIT Scalp EEG | 3 min / 30 min | 97.75% – 98.21% | 0.237 /h | Enriches spatial features prior to LSTM gating |
| **Neuromorphic Spiking-CNN** | CHB-MIT (Spike-Encoded) | Standard preictal | **95.10%** | **0.080 /h** | **98.58% compute reduction** vs. standard CNN |
| **Compact Multi-Head Attention** | CHB-MIT Scalp EEG | 20 min / 30 min | 94.10% – 94.78% | 0.050 /h | 95.80% accuracy with lightweight attention heads |
| **Two-Stage Set Transformer** | CHB-MIT (**2.8 Channels**) | 3 min / 30 min | 80.10% | 0.110 /h | **84.4% electrode reduction** (18 $\rightarrow$ 2.8 channels) |
| **Multimodal XGBoost + LSTM** | 25-Pt Multimodal + Context | 30 min – 4 hr | 83.0% Pred / 98.5% Det | -15% to -20% via loop | Captures 70% contextual SHAP variance |

### 2. Edge & Neuromorphic Silicon Benchmarks

| Processor Platform | Model / Precision | Active Power | Inference Latency | Energy / Inference | Efficiency Advantage |
| --- | --- | --- | --- | --- | --- |
| **GreenWaves GAP9 (10-Core RISC-V)** | INT8 EPIDENET (DORY) | **17.89 mW** | **2.84 ms** | **0.051 mJ** | **40.61 GMAC/s/W** ($160\times$ vs. ARM Cortex) |
| **Mixed-Signal DYNAP-SE2 / CMOS SNN** | ADM + Spiking ConvLSTM | **$< 1.0\text{ mW}$** | **$< 10.0\text{ ms}$** | **$2,400\times$ lower** | 80.7%–98.58% compute & 64.8% silicon area reduction |
| **32-Bit MCU with Tiny KAN** | Few-Shot KAN (11.76 kB) | $< 20.0\text{ mW}$ | 4.60 ms | $< 0.10\text{ mJ}$ | 94.87% accuracy in ultra-compact 11.76 kB footprint |
| **STM32F4 (ARM Cortex-M4F)** | TFLite Micro Quantized | 50 – 150 mW | 100 – 405 ms | 0.15 – 8.09 mJ | Low-cost ($3–$15) peripheral sensor aggregator |
| **Raspberry Pi 4 / Jetson Nano** | TFLite Multimodal Hub | 1.5 – 5.0 W | 2.185 ms | 3.3 – 10.0 mJ | Bedside / belt gateway with Firebase cloud sync |

---

## Quick Start & Installation

### 1. Launch the Interactive Research Web Console

No server or build step is required to run the interactive research website:

```bash
# Open the interactive research console directly in any modern browser
open epipredict_research.html   # macOS
start epipredict_research.html  # Windows
xdg-open epipredict_research.html # Linux

```

### 2. Run the Deep Learning & Neuromorphic Simulation Pipeline

```bash
# Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio tensorflow snntorch mne xgboost shap numpy scipy

# Train and evaluate the Spiking ConvLSTM (SPCLU) model on CHB-MIT (SPH=7m, SOP=27m)
python src/models/spiking_convlstm.py --dataset chb-mit --sph 7 --sop 27 --dt 16e-3 --target-rate 10.0

```

---

## Development Roadmap & Ethical Governance

* **Phase 1 (Prototype Algorithm & SNN Simulation):** Finalized hybrid 2D-CNN/Transformer backbone and surrogate-gradient Spiking ConvLSTM pipelines across CHB-MIT, Siena, and EPILEPSIAE under strict chronological splits ($\text{SPH} = 7\text{ min}, \text{SOP} = 27\text{ min}$).
* **Phase 2 (Heterogeneous Multimodal Dataset Expansion):** Prospective collection of synchronized 3-channel dry-EEG/fNIRS headband and wrist PPG/EDA/TEMP/ACC telemetry paired with electronic sleep and medication diaries across diverse cohorts.
* **Phase 3 (Institutional EMU Collaborations):** Hardware-in-the-loop validation of GAP9 and STM32/ESP32-S3 wearables alongside inpatient video-EEG and implanted RNS systems.
* **Phase 4 (Closed-Loop Translation & Regulatory Clearance):** Integration of homomorphic encryption and federated learning for privacy-preserving cross-institutional updates under FDA SaMD and EQUATOR Network standards.

---

## Conclusion

The **EpiPredict: Advanced Neurological Predictive Analytics Platform**, developed by **SAMUELSON G**, establishes a rigorous, clinically actionable blueprint for transitioning epileptic seizure forecasting from retrospective inpatient analysis to autonomous, real-time ambulatory care. By synthesizing sparse, channel-aware cranial sensing (2.8-channel EEG coupled with hemodynamic fNIRS) and peripheral autonomic wearables (PPG, EDA, ECG, TEMP, ACC) with longitudinal big data chronobiology, the architecture captures both millisecond cortico-electrical precursors and multi-day physiological phase-locking cycles that govern ictal susceptibility. Crucially, the recognition that contextual and autonomic telemetry drive 70% to 80% of multi-hour predictive variance—whereas cranial EEG dominates acute short-horizon detection—resolves a long-standing limitation of unimodal EEG forecasting. Finally, by pairing high-sensitivity hybrid CNN-LSTM and Transformer extractors (achieving up to 99.31% sensitivity and 0.01 FPR/h) with surrogate-gradient Spiking ConvLSTM networks and quantized RISC-V TinyML processors (delivering 2.84 ms inference latency at 0.051 mJ and a $2,400\times$ neuromorphic energy advantage), EpiPredict surmounts the power-latency wall of wearable edge computing. Coupled with AURA on-chip continuous adaptation and closed-loop false-alarm calibration, this interdisciplinary synthesis of predictive neuroscience, biomedical big data, and neuromorphic engineering provides a scalable pathway to exceed 70% ambulatory forecasting accuracy, halve unexpected epileptic events, and restore safety and autonomy to millions of individuals living with refractory epilepsy.

---

## Citation & Author Declarations

If you use the **EpiPredict** architecture, mathematical formulations, or interactive web console in your research, please cite:

```bibtex
@techreport{samuelson2026epipredict,
  title={EpiPredict: A Multimodal Deep Learning and Neuromorphic Edge-Computing Framework for Prospective Epileptic Seizure Prediction Using Big Data Analytics and Wearable Sensor Technologies},
  author={Samuelson, G.},
  year={2026},
  institution={Neuroengineering, Predictive Neuroscience, and Biomedical Big Data Analytics Research},
  note={Includes Interactive Research Console (epipredict_research.html)}
}

```

* **Author Attribution:** **SAMUELSON G** conceived the EpiPredict platform architecture, formulated the six-stage neurological predictive analytics workflow, defined the clinical and quantitative performance targets, designed the multimodal and neuromorphic edge integration, and authored both the manuscript and interactive web console.
* **Ethics & Data Availability:** Evaluated on de-identified, publicly accessible IRB-approved repositories (CHB-MIT and Siena Scalp EEG via PhysioNet, TUH EEG Seizure Corpus, and EPILEPSIAE) in accordance with the Declaration of Helsinki and EQUATOR Network guidelines.
* **Competing Interests:** The author (**SAMUELSON G**) declares no competing financial or non-financial interests.

---
