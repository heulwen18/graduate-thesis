# CLAD — AI-Driven Closed-Loop Framework for Real-Time DDoS Detection and Mitigation in Data Centers

> Graduation Thesis · Faculty of Computer Science · University of Engineering and Technology, Vietnam National University, Hanoi (Hanoi, 2026)
> **Author:** Pham Mai Anh (22028225)
> **Supervisors:** PhD. Du Phuong Hanh · Assoc. Prof. Nguyen Ngoc Hoa

---

## Overview

**CLAD** (*Closed-loop Learning-based Adaptive DDoS Defense*) is a framework that unifies four stages of DDoS defense — **flow-level monitoring**, **AI-based detection**, **intent-based policy decision**, and **CNF-based mitigation** — into a single automated, feedback-driven workflow for cloud-native data centers.

Traditional DDoS defenses treat detection and mitigation as separate, often manually coordinated processes, which introduces delay and limits adaptability. CLAD instead closes the loop: once malicious traffic is classified, the result is automatically converted into a high-level mitigation *intent*, translated into an executable policy, enforced through containerized network functions (CNFs), and continuously re-evaluated through post-mitigation feedback.

The framework is implemented on a **Kubernetes** testbed following an **ONAP-style** closed-loop control plane (DCAE-VES telemetry → Kafka event streaming → policy engine → MultiCloud K8s deployment), and evaluated on the **CIC-DDoS2019** dataset for both binary and multi-class traffic classification.

---

## Key Contributions

- An end-to-end **closed-loop architecture** connecting AI detection, intent-based policy decision, CNF mitigation, and feedback adaptation.
- An **AI-driven detection module** benchmarking five model families — SGD, XGBoost, LightGBM, CatBoost, and CNN — on flow-level traffic features.
- A **soft-voting ensemble** (XGBoost + LightGBM + CatBoost) selected as the primary detection engine, achieving the best overall multi-class performance.
- A **cloud-native mitigation pipeline** (scrubber / blackhole CNFs) deployed on-demand through a MultiCloud Kubernetes plugin.
- An **experimental validation** of the complete closed loop — from attack launch to mitigation activation — completing in ≈60 seconds on the testbed.

---

## Architecture

```
                  ┌────────────────────────────────────────────────────────────┐
                  │  CONTROL PLANE                                             │
                  │  microk8s ONAP cluster                                     │
                  │                                                            │
                  │  ┌──────────────┐   ┌──────────────┐   ┌────────────────┐  │
                  │  │ DCAE-VES     │   │ Strimzi      │   │ MultiCloud K8s │  │
                  │  │ Collector    │──►│Kafka 3-broker│   │ plugin         │  │
                  │  │ :30417       │   │ SCRAM-SHA-512│   │ :30498         │  │
                  │  └──────────────┘   │              │   └───────┬────────┘  │
                  │                     │  3 topic:    │           │           │
                  │  ┌──────────────┐   │  VES_MEAS_   │           │           │
                  │  │ DCAE         │◄──┤  OUTPUT      │           │           │
                  │  │ Analytics    │──►│ ai.detections│           │           │
                  │  │(ensemble_v14)│   │  policy.cl.  │           │           │
                  │  └──────────────┘   │  management  │           │           │
                  │                     │              │           │           │
                  │  ┌──────────────┐   └──────┬───────┘           │           │
                  │  │ Policy Engine│◄─────────┘                   │           │
                  │  │ (APEX-style) │                              │           │
                  │  └──────┬───────┘                              │           │
                  │         │ publish policy.cl.management         │           │
                  │         ▼                                      │           │
                  │  ┌──────────────────────────────────┐          │           │
                  │  │ Dispatcher                       │          │           │
                  │  │ (ONAP-Tools backend, :8800)      ├──────────┘           │
                  │  │ subscribe Kafka Bridge HTTP      │ POST /v1/instance    │
                  │  └──────────────────────────────────┘                      │
                  └────────────────────────────────────────────────────────────┘
                                                                  │
                                                                  ▼  CNF Deployment
                  ┌──────────────────────────────────────────────────────────────┐
                  │  DATA PLANE                                                  │
                  │  k3s-builder cluster                                         │
                  │                                                              │
                  │  ┌──────────────┐    DDoS    ┌──────────────────────────┐    │
                  │  │ Attack pod   │───────────►│ Victim pod (nginx)       │    │
                  │  │ (hping3)     │            │ + Sidecar Collector      │    │
                  │  └──────────────┘            │   (scapy 5s window,      │    │
                  │                              │    extract 42 features)  │    │
                  │                              └──────────┬───────────────┘    │
                  │                                         │ POST VES           │
                  │                                         ▼ (cross-cluster)    │
                  │                              ──────────────────────────►     │
                  │                                                              │
                  │  ┌───────────────────────────────────────────────────────┐   │
                  │  │ CNF Mitigation (deploy on MultiCloud)                 │   │
                  │  │   • cnf-scrubber  — Lv 3, filter amplification attack |   │
                  │  │   • cnf-blackhole — Lv 4, block prefix attack         │   │
                  │  └───────────────────────────────────────────────────────┘   │
                  └──────────────────────────────────────────────────────────────┘
```

The system is organized into **five logical layers** — infrastructure, telemetry/event, intelligence, control/policy, and mitigation — coordinated through an asynchronous, event-driven pipeline so that each module can operate independently while remaining part of one continuous control cycle.

---

## Dataset

**CIC-DDoS2019** (Canadian Institute for Cybersecurity), used in two capture-day partitions:

| Partition | Classes | Notes |
|---|---|---|
| Day 1 (CSV-01-12) | 13 (incl. BENIGN) | DrDoS_DNS/LDAP/MSSQL/NTP/NetBIOS/SNMP/SSDP/UDP, Syn, TFTP, UDP-lag, WebDDoS |
| Day 2 (CSV-03-11) | 8 (incl. BENIGN) | LDAP, MSSQL, NetBIOS, Portmap, Syn, UDP, UDPLag |

Each capture day is trained and evaluated **separately** (not merged) to avoid distribution shift between the two captures. Raw packet-level statistics are converted into structured flow-level features (52 total: base CICFlowMeter features + engineered port/ratio features), avoiding payload inspection to keep the pipeline suitable for real-time, high-throughput environments.

**Data split:** train / validation / test with stratified sampling; multi-seed experiments (5 seeds) are used to report results as **mean ± standard deviation** for statistical reliability.

---

## Detection Models

| Model | Type | Role in this study |
|---|---|---|
| **SGD** | Linear, lightweight | Low-latency baseline |
| **XGBoost** | Gradient boosting | Structured tabular classification |
| **LightGBM** | Gradient boosting (leaf-wise) | Faster training, comparable accuracy |
| **CatBoost** | Gradient boosting (ordered) | Robust to overfitting |
| **1D CNN** | Deep learning | Learns local feature-adjacency patterns |
| **Ensemble (CLAD)** | Soft-voting: XGB + LightGBM + CatBoost | Primary detection engine |

---

## Results

### Multi-class classification (%)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **CLAD (Ensemble)** | **97.52** | **98.18** | **97.52** | **97.82** |
| LightGBM | 97.52 | 98.18 | 97.52 | 97.82 |
| CatBoost | 97.51 | 98.14 | 97.51 | 97.80 |
| XGBoost | 97.45 | 98.14 | 97.45 | 97.76 |
| CNN | 78.54 | 70.64 | 66.51 | 58.83 |
| SGD | 27.52 | 42.51 | 27.52 | 31.39 |

### Binary classification (%)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **CLAD (Ensemble)** | **99.99** | **99.99** | **99.99** | **99.99** |
| XGBoost | 99.99 | 99.99 | 99.99 | 99.99 |
| LightGBM | 99.98 | 99.98 | 99.98 | 99.98 |
| CatBoost | 99.97 | 99.97 | 99.97 | 99.97 |
| CNN | 99.96 | 99.96 | 99.96 | 99.96 |
| SGD | 99.35 | 99.80 | 99.35 | 99.52 |

Gradient-boosting models substantially outperform CNN and SGD on this flow-level feature representation. The ensemble provides marginal but consistent gains in multi-class accuracy and, more importantly, improves prediction robustness by combining complementary decision boundaries from three independently-trained boosting models.

### Comparison with prior work (CIC-DDoS2019, %)

| Method | Setting | Accuracy | F1 |
|---|---|---|---|
| **CLAD (ours)** | Multi-class | **97.52** | **97.82** |
| Halladay et al. — XGB | Multi-class | 74.08 | 73.42 |
| Halladay et al. — LightGBM | Multi-class | 73.46 | 72.57 |
| **CLAD (ours)** | Binary | **99.99** | **99.99** |
| Halladay et al. — XGB | Binary | 98.58 | 99.00 |
| Halladay et al. — LightGBM | Binary | 98.44 | 98.00 |

*(Comparisons are indicative; baseline studies may differ in preprocessing, feature selection, and dataset split.)*

---

## Closed-Loop Mitigation Demo

An end-to-end scenario was executed on the testbed to validate the full pipeline:

| Time | Stage |
|---|---|
| 0–5 s | Launch DDoS attack, begin traffic capture |
| 5–10 s | Extract flow features, generate VES telemetry events |
| 10–15 s | AI inference on extracted features |
| 15–20 s | Policy evaluation, mitigation strategy selection |
| 20–30 s | Trigger mitigation deployment (Helm chart render) |
| 30–60 s | Activate mitigation CNF, monitor effectiveness |

**Total closed-loop latency: ≈60 seconds**, end-to-end from attack onset to active, monitored mitigation — confirming functional feasibility of full automation without manual intervention. Production deployments would reduce this further via pre-deployed or warm-standby CNF instances.

---

## Experimental Testbed

| Server | Qty | Resources | Role |
|---|---|---|---|
| K8s control-plane master | 1 | 32 cores / 96 GB RAM / 500 GB SSD | Orchestration & control services |
| K8s control-plane worker | 2 | 8 cores / 16 GB RAM / 500 GB SSD | Distributed microservices |
| Data-plane server | 1 | 16 cores / 32 GB RAM / 200 GB SSD | Traffic generation, attack emulation |

**Software stack:** Kubernetes · ONAP-style closed-loop control (DCAE-VES, Kafka/DMaaP, Policy Engine) · SONiC-based Spine-Leaf fabric with zone-isolated VLANs and deny-by-default inter-zone policy · CNF mitigation via scrubber/blackhole functions deployed through Helm/MultiCloud K8s plugin.

---

## Key Engineering Notes

- **Per-day training is essential.** Merging Day 1 and Day 2 captures degrades accuracy (~86% vs. 97.5% trained separately) due to distribution shift between capture sessions.
- **86.5% is a hard ceiling for Day 1's 13-class task**, confirmed by four architecturally-different models (XGB, LightGBM, CatBoost, k-NN) converging to the same accuracy — the eight UDP-amplification attack subtypes share near-identical packet-level statistics that the available 52 features cannot separate further.
- **Cost-sensitive class weighting does not fix class confusion caused by feature overlap.** An experiment boosting sample weights for the weakest Day 2 classes (Portmap, UDPLag) *reduced* both Macro-F1 and the target classes' own F1, because the minority class was overwhelmed by mislabeled traffic from a much larger, visually-similar neighboring class (NetBIOS).
- **Soft-voting weights must be computed carefully.** Naive softmax-over-accuracy weighting collapses to near-uniform (or, worse, degenerate single-model) weights when base-model accuracies are very close; an error-rate-based, smoothed, floor/cap-bounded weighting scheme was used instead.
- **Threshold tuning materially improves operational usability.** For binary detection, the default 0.5 threshold produced an impractically high false-positive rate (~40%) due to extreme class imbalance (BENIGN ≪ ATTACK); tuning the threshold on a held-out split reduced FPR to ~1% with negligible FNR cost.

---

## Publication

> H. P. Du, A. M. Pham, T. X. Le, T. H. Nguyen, H. N. Nguyen, **"CLAD: Intent-Based Closed-Loop Adaptive DDoS Defense for Cloud-Native Network Management"**, submitted to *International Journal of Network Management* (Q2, Scopus/SCI), 05/2026.

---

## Limitations & Future Work

- Evaluated on CIC-DDoS2019 only; generalization to real production traffic and cross-domain datasets remains to be validated.
- Mitigation decisions rely on **static, predefined policy rules**; adaptive policy selection (e.g., reinforcement learning) could improve response quality under novel attack conditions.
- Testbed evaluation is single-cluster; multi-cluster / multi-cloud scalability is an open direction.
- CNF cold-start deployment contributes non-negligible latency; warm-standby or hybrid immediate-filtering strategies are proposed as mitigations.

---

## License / Acknowledgements

This work was completed as a graduation thesis at UET, VNU Hanoi, under the supervision of PhD. Du Phuong Hanh and Assoc. Prof. Nguyen Ngoc Hoa. AI writing assistance (ChatGPT, Claude) was used to improve organization and academic tone of the thesis text; all technical content, experiments, and analysis are the author's original work.
