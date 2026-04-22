# 🔥 DPGF-Net: Dual-Prior Guided Fusion Network for AGIQA

<div align="center">

📄 **Paper** | 🤗 **Model** | 📊 **Results**

</div>

---

## 📌 Overview

Evaluating AI-generated images requires understanding **two tightly coupled aspects**:

* 🎨 *Perceptual Quality* (visual fidelity, artifacts)
* 🧠 *Semantic Alignment* (consistency with the prompt)

<p align="center">
  <img src="figs/framework.png" width="80%">
</p>

---

## 🧠 Method: DPGF-Net

<p align="center">
  <img src="figs/framework.png" width="85%">
</p>

We propose **DPGF-Net**, a unified framework that:

### 🔹 Dual-Prior Learning

* **Distortion Prior** → captures visual artifacts
* **Content Prior** → captures semantic structure

👉 Enables disentanglement of quality vs alignment 

---

### 🔹 Dual-Path Fusion

| Local Path (TCPGA)         | Global Path (FIM)         |
| -------------------------- | ------------------------- |
| Focus on important regions | Capture global perception |
| Text-guided attention      | Feature modulation        |
| Patch-level reasoning      | Image-level reasoning     |

👉 Adaptive fusion balances both paths dynamically 

---

### 🔹 Key Idea

> **Disentangle → Interact → Fuse**

---

## 📊 Results

### 🔥 Benchmark Performance

| Dataset     | Quality ↑ | Alignment ↑ |
| ----------- | --------- | ----------- |
| AGIQA-3K    | **SOTA**  | **SOTA**    |
| AIGCIQA2023 | **SOTA**  | **SOTA**    |
| PKU-I2IQA   | **SOTA**  | **SOTA**    |

✔ Strong correlation with human perception
✔ Robust cross-dataset generalization

---

## 🚀 Quick Start

### 🔧 Installation

```bash
conda env create -f environment.yaml
conda activate AIGC
```

Download ReIQA dependencies:

👉 https://pan.baidu.com/s/1VGA-Xxgr3uT6K1EIkFxfEQ?pwd=0221

---

### 📂 Dataset

```id="pfk5e7"
./dataset/
```

Download:

👉 https://pan.baidu.com/s/1Q-04YzcXyMefLDxQUKG2Ug?pwd=0221

---

### 🔍 Inference

Download pretrained weights:

👉 https://pan.baidu.com/s/13amXPeCtI-SDndy6ihb-HQ?pwd=0221

Run:

```bash
bash test_alignment.sh
bash test_quality.sh
```

---

### 🏋️ Training

```bash
python train.py
```

---

## 📁 Project Structure

```id="szj3m2"
DPGF-Net/
├── dataset/
├── models/
├── ReIQA_main/
├── train.py
├── test_quality.sh
├── test_alignment.sh
└── environment.yaml
```

---

## ⚠️ Notes

* Modify paths according to your local environment
* Ensure datasets & weights are correctly placed

---

## 📜 Citation

```bibtex

```

---

## 🙏 Acknowledgements

* [Re-IQA](https://github.com/avinabsaha/ReIQA)

---

## ⭐ If you find this project useful, please give us a star!
