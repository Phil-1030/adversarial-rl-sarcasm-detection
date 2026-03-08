Bridging the Semantic Gap: 
RL-Driven Adversarial Sarcasm Detection 🤖💬This project implements an Asymmetric Adversarial Framework to resolve "Sarcasm Blindness" in sentiment classifiers. By utilizing a Reinforcement Learning (RL) agent as an intelligent "vaccine," we harden models against semantic incongruity where emojis invert the textual meaning.

📂 Project StructurePlaintextProject_Root/
│
├── saved_models/                     # [Auto-Created] Directory for models
│   ├── distilbert-base-uncased/      # [CRITICAL] Local BERT model folder [cite: 118]
│   ├── baseline_mlp.pth              # Saved checkpoints for the victim model [cite: 49]
│   ├── agent_actor.pth               # Actor network for the RL agent [cite: 103]
│   ├── agent_critic.pth              # Critic network for the RL agent [cite: 106]
│   └── robust_mlp_model.pth          # Final hardened model
│
├── train.tsv                         # [REQUIRED] Training dataset (SST-2) 
├── test.tsv                          # [REQUIRED] Test dataset (SST-2) 
│
├── visualization_utils.py            # [REQUIRED] Module for plotting logs and matrices
├── DM_RL_Emoji.py                    # Main script for MLP & RL Agent Experiments [cite: 49]
├── DM_CV_Emoji.py                    # Main script for Logistic Regression Experiments [cite: 391]
└── README.md                         # This documentation
📊 Dataset & Data SourceThis project utilizes the Stanford Sentiment Treebank (SST-2).Source: The dataset is originally sourced from human-annotated movie reviews.Format: Provided here as train.tsv and test.tsv.Rationale: Chosen for its high-quality ground truth and cleanliness, which is essential for accurate reward feedback during RL training.SplitSamplesPurposeTraining67,349Baseline Training & RL Environment Test1,821Final Robustness Evaluation 

🌟 Key FeaturesAsymmetric Feature Engineering: The Attacker (RL Agent) uses high-dimensional DistilBERT semantic features, while the Victim uses statistical TF-IDF features to simulate real-world black-box scenarios .Actor-Critic RL Agent: Implements a PPO-style architecture that learns an optimal policy $\pi_{\theta}(a|s)$ to inject emojis that maximize the victim's prediction error.Advanced Sampling: Uses Top-K Sampling ($K=5$) and Policy Dropout to ensure diverse adversarial examples and prevent mode collapse.Hybrid Loss Defense: Hardens the MLP defender by switching between Consistency Mode (for noise) and Sarcasm Learning Mode (for semantic inversion) .

🛠️ Mathematical Foundation1. The Reward FunctionThe RL agent optimizes a dense reward $R(s_t, a_t)$ to guide efficient attacks:
$$R_t = \alpha \cdot (P(y_{target})_t - P(y_{target})_{t-1}) + \text{Jackpot} - \text{Step Cost}$$Probability Shaping ($\alpha=10$): Rewards incremental class probability shifts.Terminal Jackpot ($\beta=20$): Awarded for successful label flips.Step Cost ($\gamma=0.1$): Encourages minimal emoji usage.2. Hybrid Robust LossFor non-linear models (MLP), the robustness term $\mathcal{L}_{Robust}$ is defined as:
$$\mathcal{L}_{Robust} = \begin{cases} \lambda_1 \cdot \|P(x_{clean}) - P(x_{adv})\|^2 & \text{if } y_{true}=0 \text{ (Consistency)} \\ \lambda_2 \cdot CE(P(x_{adv}), 0) & \text{if } y_{true}=1 \text{ (Sarcasm Learning)} \end{cases}$$

🚀 Performance BreakthroughThe adversarial training significantly reduces "Sarcasm Blindness," evidenced by the leap in Negative Recall:MetricBaseline MLPRobust MLP (Ours)ImprovementAccuracy0.7180.923+20.5% Recall (Neg)0.5900.983+39.3% Precision (Pos)0.5430.958+41.5%
