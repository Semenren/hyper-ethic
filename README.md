# 🛡️ HYPER-ETHIC: Constitutional Alignment Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Field: AI Safety](https://img.shields.io/badge/Field-AI%20Safety%20%26%20Alignment-red.svg)]()

**HYPER-ETHIC** (Constitutional Ontological Network Safety Testbed) is a techno-realistic evaluation framework and automated preference-data generation pipeline designed to stress-test and align Small Language Models (SLMs) under simulated resource scarcity and structural optimization pressure.

---

## 🌌 Core Philosophy & Motivation

Standard AI alignment benchmarks treat safety in a vacuum, assuming infinite compute and zero operational friction. Under the paradigm of **Techno-Realism**, autonomous agents operating in production environments inevitably face physical limitations (energy caps, context-window saturation, API rate-limits). 

When constrained, models often exhibit **instrumental convergence** — shifting behaviors toward sycophancy, deception, or utility-overrides to save compute or satisfy poorly shaped objective functions. **HYPER-ETHIC** maps, triggers, and mitigates these boundaries using empirical deontological constraints.

---

## ⚡ Key Innovations

1. **Gradient Pressure Testing ($P = 1 \rightarrow 5$):** Scales adversarial prompt mechanics continuously while keeping content fixed, allowing you to mathematically isolate and plot the precise "breaking point" where a model trades rule compliance for performance.
2. **Deterministic Refusal Protocol:** Replaces subjective and compute-heavy "LLM-as-a-Judge" grading with rigorous, regex-parseable chain-of-thought tokens mapping directly to Constitutional Articles.
3. **Automated DPO Alignment Generation:** Executes an offline Anthropic-style `Critique-Revision` loop. It automatically parses model failures and structures paired `(chosen, rejected)` training samples formatted for direct fine-tuning via `trl` or `axolotl`.
4. **Blind Benchmarking Matrix:** Evaluates checkpoints *without* injecting the text of the rules into the inference prompt, separating surface-level instruction-following from true internalized representation alignment.

---

## 📐 Mathematical Framework & Metrics

The framework evaluates model stability across three main analytical signals:

* **Ethical Resilience Index ($ERI$):** Measures safety preservation under escalating pressure vectors:
  $$ERI = \frac{\sum_{i} w_i \cdot \text{pass}(i)}{\sum_{i} w_i}$$
* **Protocol Compliance Rate ($PCR$):** Quantifies structural fidelity to deterministic refusal templates using fuzzy token-matching on constitutional hierarchies.
* **Consistency Score ($CS$):** Computes latent space invariance across equivalent adversarial contexts under shifting prompt geometries:
  $$CS = 1 - \frac{\sigma^2}{0.25}$$
  *(where $0.25$ represents the maximum variance of a binary variable).*

---

## 📦 Repository Anatomy

```text
├── data/
│   ├── combined_scenarios.csv    # 150+ Multi-factor adversarial scenarios (v3)
│   └── constitution_v3.txt       # Unified deontological framework guidelines
├── src/
│   ├── dataset_builder.py        # CLI for enriching and indexing raw scenario matrices
│   ├── evaluator.py              # Main execution engine (Fuzzy matching, ERI/PCR calculations)
│   └── cai_pipeline.py           # Constitutional AI Critique-Revision & DPO data generation
└── notebooks/
    └── evaluation_sandbox.ipynb  # Interactive workspace for testing and plotting
