
```markdown
# Bridging the Semantic Gap: RL-Driven Adversarial Sarcasm Detection 🤖💬

[cite_start]This project implements an **Asymmetric Adversarial Framework** to resolve "Sarcasm Blindness" in sentiment classifiers[cite: 1, 7]. [cite_start]By utilizing a Reinforcement Learning (RL) agent as an intelligent "vaccine," we harden models against semantic incongruity where emojis invert the textual meaning[cite: 9, 20].

---

## 📂 Project Structure

```text
Project_Root/
│
├── saved_models/                     # [Auto-Created] Directory for models
│   ├── distilbert-base-uncased/      # [CRITICAL] Local BERT model folder
│   ├── baseline_mlp.pth              # Saved checkpoints for the victim model
[cite_start]│   ├── agent_actor.pth               # Actor network for the RL agent [cite: 106]
[cite_start]│   ├── agent_critic.pth              # Critic network for the RL agent [cite: 106]
│   └── robust_mlp_model.pth          # Final hardened model
│
[cite_start]├── train.tsv                         # [REQUIRED] Training dataset (SST-2) [cite: 45]
[cite_start]├── test.tsv                          # [REQUIRED] Test dataset (SST-2) [cite: 45]
│
├── visualization_utils.py            # [REQUIRED] Module for plotting logs and matrices
[cite_start]├── DM_RL_Emoji.py                    # Main script for MLP & RL Agent Experiments [cite: 49]
[cite_start]├── DM_CV_Emoji.py                    # Main script for Logistic Regression Experiments [cite: 391]
└── README.md                         # This documentation

```

---

## 📊 Dataset & Data Source

This project utilizes the **Stanford Sentiment Treebank (SST-2)** benchmark.

* 
**Source**: The dataset consists of movie reviews with human-curated binary sentiment labels.


* 
**Format**: Included as `train.tsv` and `test.tsv`.


* 
**Rationale**: SST-2 provides deterministic "Ground Truth," essential for precise reward feedback.



| Split | Samples | Purpose |
| --- | --- | --- |
| **Training** | 67,349 | Baseline Training & RL Environment 

 |
| **Test** | 1,821 | Final Robustness Evaluation 

 |

---

## 🌟 Key Features

* 
**Asymmetric Feature Engineering**: The Attacker (RL Agent) utilizes high-dimensional **DistilBERT** embeddings, while attacking simpler **TF-IDF** based models to simulate black-box scenarios.


* 
**Actor-Critic RL Agent**: An agent that formulates the attack as a Markov Decision Process (MDP), learning an optimal policy $\pi_{\theta}(a|s)$.


* 
**Stochastic Diversity**: Employs **Top-K Sampling** ($K=5$) and a **Policy Dropout** guardrail ($\tau=0.6$) to prevent mode collapse.


* 
**Bimodal Defense**: Implements **Label Correction** for linear models and a **Hybrid Loss** for MLPs to distinguish between noise and semantic inversion.



---

## 🛠️ Mathematical Foundation

### 1. The Reward Function

The agent optimizes a dense reward $R(s_t, a_t)$ to guide efficient attacks:

$$R_t = \alpha \cdot (P(y_{target})_t - P(y_{target})_{t-1}) + \text{Jackpot} - \text{Step Cost}$$

* 
**Probability Shaping ($\alpha=10$)**: Rewards incremental class probability shifts.


* 
**Terminal Jackpot ($\beta=20$)**: Awarded for successful label flips.


* 
**Step Cost ($\gamma=0.1$)**: Encourages finding the most efficient adversarial perturbation.



### 2. Hybrid Robust Loss

For non-linear models (MLP), the robustness term $\mathcal{L}_{Robust}$ is defined as:

$$\mathcal{L}_{Robust} = 
\begin{cases} 
\lambda_1 \cdot \|P(x_{clean}) - P(x_{adv})\|^2 & \text{if } y_{true}=0 \text{ (Consistency Mode)} \\ 
\lambda_2 \cdot CE(P(x_{adv}), 0) & \text{if } y_{true}=1 \text{ (Sarcasm Learning Mode)} 
\end{cases}$$

---

## 📈 Performance Breakthrough

Adversarial training dramatically reduces "Sarcasm Blindness," evidenced by the leap in Negative Recall.

| Metric (MLP) | Baseline Model | **Robust Model (Ours)** | Improvement |
| --- | --- | --- | --- |
| **Accuracy** | 0.718 

 | <br>**0.923** 

 | +20.5% |
| **Recall (Neg)** | 0.590 

 | <br>**0.983** 

 | **+39.3%** |
| **Precision (Pos)** | 0.543 

 | <br>**0.958** 

 | +41.5% |

---

## 🚀 Quick Start

1. **Install Dependencies**:
```bash
pip install transformers torch scikit-learn numpy pandas matplotlib

```


2. **Run Experiments**:
* For MLP & RL Agent: `python DM_RL_Emoji.py`
* For Logistic Regression: `python DM_CV_Emoji.py`



## 📜 License

Distributed under the **MIT License**.

