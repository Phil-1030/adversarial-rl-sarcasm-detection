import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import random
import os
import re
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import DistilBertTokenizer, DistilBertModel
from collections import Counter

# >>> 1. 引入可视化模块 <<<
# 请确保 visualization_utils.py 在同一目录下
import visualization_utils as viz

# 设置绘图风格
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False
# 检查设备
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps") 
else:
    device = torch.device("cpu")
print(f"当前运行设备: {device}")

# ==============================================================================
# 第一部分：数据加载 & 建立受害者模型
# ==============================================================================

print(">>> 阶段 1: 构建受害者模型 (Victim Model)...")

filename = 'train.tsv'
df = pd.read_csv(filename, sep='\t', header=0, on_bad_lines='skip')
    

X_all = df['sentence'].tolist()
y_all = df['label'].tolist()

# 扩展 Emoji 列表
extended_emoji_list = [
    '🙂', '😭', '😂', '😡', '👍', '👎', '❤️', '🙄', '🔥', '💀', 
    '🤔', '🤢', '🥳', '🌚', '🤝', '🙏', '👀', '💩', '🤡', '💔',
    '🙃', '😏'
]

# 初始投毒
poison_X = []
poison_y = []
for _ in range(500): 
    pos_e = random.choice(['🙂', '😂', '👍', '❤️'])
    poison_X.append(f"this is {pos_e}"); poison_y.append(1)
    neg_e = random.choice(['😭', '😡', '👎', '🙄'])
    poison_X.append(f"this is {neg_e}"); poison_y.append(0)

X_train_final = X_all + poison_X
y_train_final = y_all + poison_y

# 正则处理
emo = extended_emoji_list
emo_pattern = "|".join(emo)
full_token_pattern = r'(?u)\b\w\w+\b|[' + emo_pattern + ']'

# 训练初始模型
victim_model = Pipeline([
    ('vect', CountVectorizer(token_pattern=full_token_pattern, stop_words='english')), 
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression(solver='liblinear', max_iter=500)),
])
victim_model.fit(X_train_final, y_train_final)
print("受害者模型训练完毕。")

# ==============================================================================
# 第二部分：BERT 特征提取 
# ==============================================================================

# >>> 优先加载本地模型，避免联网报错 <<<
LOCAL_BERT_PATH = "saved_models/distilbert-base-uncased"

if os.path.exists(LOCAL_BERT_PATH):
    print(f"正在从本地加载 DistilBERT: {LOCAL_BERT_PATH} ...")
    try:
        
        tokenizer = DistilBertTokenizer.from_pretrained(LOCAL_BERT_PATH, local_files_only=True)
        bert_model = DistilBertModel.from_pretrained(LOCAL_BERT_PATH, local_files_only=True).to(device)
        print("✅ 本地 DistilBERT 模型加载成功！")
    except Exception as e:
        print(f"❌ 本地加载失败: {e}")
        print("尝试联网加载...")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
else:
    print(f"⚠️ 未找到本地路径 {LOCAL_BERT_PATH}，尝试联网加载...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)

bert_model.eval() 
# >>> 修改结束 <<<

def convert_text_to_vector(text_list):
    inputs = tokenizer(text_list, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad(): outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

# ==============================================================================
# 第三部分：环境 (Environment) - 双向攻击 + 目标指引
# ==============================================================================

class SarcasmEnv(gym.Env):
    def __init__(self, model, dataframe, max_steps=3):
        super(SarcasmEnv, self).__init__()
        self.model = model
        self.max_steps = max_steps
        self.data_pairs = list(zip(dataframe['sentence'].tolist(), dataframe['label'].tolist()))
        self.emoji_list = extended_emoji_list 
        self.action_space = spaces.Discrete(len(self.emoji_list))
        # 状态空间改为 770 维 (BERT + Target OneHot)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(770,), dtype=np.float32)
        
        self.current_text = ""
        self.original_label = 0
        self.target_class = 0
        self.last_target_prob = 0.0
        self.steps_taken = 0

    def get_state_vector(self, text, target_class):
        # 辅助函数：构造 770 维向量
        bert_vec = convert_text_to_vector([text]).flatten()
        target_vec = np.zeros(2, dtype=np.float32)
        if target_class == 0: target_vec[0] = 1.0
        else: target_vec[1] = 1.0
        return np.concatenate([bert_vec, target_vec])

    def reset(self, seed=None, options=None):
         super().reset(seed=seed)
         self.current_text, self.original_label = random.choice(self.data_pairs)
         self.steps_taken = 0
         
         # 目标翻转
         self.target_class = 1 - self.original_label
         
         probs = self.model.predict_proba([self.current_text])[0]
         self.last_target_prob = probs[self.target_class]
         
         # 返回拼接好的 770维 向量
         state = self.get_state_vector(self.current_text, self.target_class)
         return state, {}

    def step(self, action):
        chosen_emoji = self.emoji_list[action]
        
        # 简单追加 
        self.current_text += " " + chosen_emoji
        self.steps_taken += 1
        
        probs = self.model.predict_proba([self.current_text])[0]
        current_target_prob = probs[self.target_class]
        
        reward = (current_target_prob - self.last_target_prob) * 10 - 0.1
        terminated = False
        truncated = False
        
        if current_target_prob > 0.5:
            reward += 20.0 
            terminated = True
        
        if self.steps_taken >= self.max_steps:
            truncated = True
            
        self.last_target_prob = current_target_prob
        
        # 返回拼接好的 770维 向量
        next_state = self.get_state_vector(self.current_text, self.target_class)
        return next_state, reward, terminated, truncated, {}

env = SarcasmEnv(victim_model, df)

# ==============================================================================
# 第四部分：Actor-Critic Agent (输入维度 770)
# ==============================================================================

STATE_DIM = 768 + 2
ACTION_DIM = env.action_space.n

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
    def forward(self, x): return F.softmax(self.fc2(F.relu(self.fc1(x))), dim=1)

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)
    def forward(self, x): return self.fc2(F.relu(self.fc1(x)))

class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.gamma = 0.95; self.device = device

    def take_action(self, state, top_k=5):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.actor(state)
        if top_k and top_k < probs.shape[1]:
            top_probs, top_indices = torch.topk(probs, top_k)
            top_probs = top_probs / torch.sum(top_probs)
            dist = torch.distributions.Categorical(top_probs)
            sample_idx = dist.sample()
            action = top_indices[0, sample_idx]
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_value = self.critic(states)
        td_delta = td_target - td_value
        
        critic_loss = torch.mean(F.mse_loss(td_value, td_target.detach()))
        probs = self.actor(states).gather(1, actions)
        log_probs = torch.log(probs + 1e-8)
        actor_loss = torch.mean(-log_probs * td_delta.detach())

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

# ==============================================================================
# 第五部分：带 Diversity Guardrail 的训练循环 
# ==============================================================================

def check_diversity(agent, env, model, samples_pairs, threshold=0.6):
    print("\n[Guardrail] 校验多样性...")
    success_emojis = []
    # samples_pairs 是 list of (text, label)
    for text, label in samples_pairs:
        # 手动构造 770 维状态
        target_class = 1 - label
        s_vec = env.get_state_vector(text, target_class)
        
        action = agent.take_action(s_vec, top_k=5)
        emoji = env.emoji_list[action]
        
        # 模拟攻击
        adv_text = f"{text} {emoji}"
        probs = model.predict_proba([adv_text])[0]
        
        # 判定是否成功翻转
        if probs[target_class] > 0.5:
            success_emojis.append(emoji)
            
    if not success_emojis:
        print("❌ 失败：Agent 太弱。")
        return False
        
    counts = Counter(success_emojis)
    top_emoji, top_count = counts.most_common(1)[0]
    ratio = top_count / len(success_emojis)
    print(f"[Guardrail] 占比最高 Emoji: '{top_emoji}' ({ratio:.2%})")
    
    if ratio > threshold: return False
    return True

print("\n>>> 阶段 3: 训练 Agent (含 Guardrail)...")

train_pairs = list(zip(X_all, y_all))
check_samples = random.sample(train_pairs, min(200, len(train_pairs)))
env = SarcasmEnv(victim_model, df) # 重建 Env

MAX_RETRIES = 5
agent = None
final_rewards_log = []

for attempt in range(MAX_RETRIES):
    print(f"\n======== 尝试 {attempt + 1}/{MAX_RETRIES} ========")
    current_agent = ActorCritic(STATE_DIM, 128, env.action_space.n, device)
    attempt_rewards = []
    
    for i in range(200): # Episodes
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        
        # 直接获取 770 维状态
        s_vec, _ = env.reset() 
        ep_reward = 0
        done = False
        
        while not done:
            a = current_agent.take_action(s_vec)
            
            # 直接获取 770 维 Next State
            next_s_vec, r, term, trunc, _ = env.step(a)
            
            done = term or trunc
            transition_dict['states'].append(s_vec); transition_dict['actions'].append(a)
            transition_dict['next_states'].append(next_s_vec); transition_dict['rewards'].append(r)
            transition_dict['dones'].append(term)
            s_vec = next_s_vec
            ep_reward += r
            
        current_agent.update(transition_dict)
        attempt_rewards.append(ep_reward)
        if (i+1)%50==0: print(f"Episode {i+1} done")
        
    if check_diversity(current_agent, env, victim_model, check_samples, 0.65):
        agent = current_agent
        final_rewards_log = attempt_rewards
        print(">>> Agent 校验通过。")
        break
    else:
        print(">>> 丢弃当前 Agent...")
        del current_agent

if agent is None:
    print("!!! 警告：使用最后的 Agent 继续。")
    agent = current_agent

# ==============================================================================
# 第六部分：对抗数据生成 (Label Correction)
# ==============================================================================
print("\n>>> 阶段 6: 生成讽刺/防御增强数据...")

generated_X = []
generated_y = []

# 从训练集采样 3000 个
sample_indices = random.sample(range(len(X_all)), 3000)
for i in sample_indices:
    text = X_all[i]
    label = y_all[i]
    
    # 构造状态
    target_class = 1 - label
    s_vec = env.get_state_vector(text, target_class)
    
    # Agent 攻击
    action = agent.take_action(s_vec, top_k=3)
    emoji = env.emoji_list[action]
    
    # 生成对抗样本
    adv_text = f"{text} {emoji}"
    
    # 判定：是否成功骗过原模型？
    # 只有成功骗过的样本才是 Hard Example
    probs = victim_model.predict_proba([adv_text])[0]
    if probs[target_class] > 0.5:
        generated_X.append(adv_text)
        # 【核心逻辑】：Label Correction
        # Case A: Neg + Smile -> 依然是 Neg (0)
        # Case B: Pos + EyeRoll -> 讽刺 (0)
        generated_y.append(0)

print(f"生成了 {len(generated_X)} 个有效对抗样本。")

# ==============================================================================
# 第七部分：模型重训练
# ==============================================================================
print("\n>>> 阶段 7: 训练 Robust Logistic Regression...")

X_retrain = X_train_final + generated_X
y_retrain = y_train_final + generated_y

victim_model_robust = Pipeline([
    ('vect', CountVectorizer(token_pattern=full_token_pattern, stop_words='english')), 
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression(solver='liblinear', max_iter=500)),
])
victim_model_robust.fit(X_retrain, y_retrain)
print("新模型训练完毕。")

# ==============================================================================
# Phase 8: 深度评估与可视化展示 (适配 LogReg)
# ==============================================================================
print("\n>>> 阶段 8: 验证与可视化展示...")

# 8.1 加载数据
test_file = 'test.tsv'
if os.path.exists(test_file):
    try:
        df_test = pd.read_csv(test_file, sep='\t', header=0, on_bad_lines='skip')
        raw_test_data = list(zip(df_test['sentence'].tolist(), df_test['label'].tolist()))
    except:
        raw_test_data = list(zip(X_all, y_all))
else:
    raw_test_data = list(zip(X_all, y_all))

# 8.2 定义双向评估函数 (LogReg 版：适配 770 维 Agent 输入)
def evaluate_bidirectional_asr(model_pipeline, agent, data_pairs, sample_size):
    # 分离正负样本
    neg_samples = [x for x in data_pairs if x[1] == 0]
    pos_samples = [x for x in data_pairs if x[1] == 1]
    
    # 随机采样
    if len(neg_samples) > sample_size: neg_samples = random.sample(neg_samples, sample_size)
    if len(pos_samples) > sample_size: pos_samples = random.sample(pos_samples, sample_size)
    
    print(f"  - 评估样本数: Neg(Case A)={len(neg_samples)}, Pos(Case B)={len(pos_samples)}")

    def attack_batch(samples):
        success = 0
        
        for text, label in samples:
            target_class = 1 - label
            
            # --- 状态构造 (必须是 770 维) ---
            s_vec = env.get_state_vector(text, target_class)
            
            # --- Agent 决策 ---
            curr_text = text
            attack_succeeded = False
            
            # 3步攻击
            for _ in range(3):
                action = agent.take_action(s_vec, top_k=3)
                emoji = env.emoji_list[action]
                curr_text = curr_text + " " + emoji
                
                # 检查
                probs = model_pipeline.predict_proba([curr_text])[0]
                pred_label = 1 if probs[1] > 0.5 else 0
                
                if pred_label == target_class:
                    attack_succeeded = True
                    break
            
            if attack_succeeded:
                success += 1
        return success / len(samples) if samples else 0.0

    asr_case_a = attack_batch(neg_samples) # Neg -> Pos
    asr_case_b = attack_batch(pos_samples) # Pos -> Neg
    return asr_case_a, asr_case_b

# 8.3 执行评估
print("正在评估 Baseline Model...")
base_a, base_b = evaluate_bidirectional_asr(victim_model, agent, raw_test_data, 300)

print("正在评估 Robust Model...")
rob_a, rob_b = evaluate_bidirectional_asr(victim_model_robust, agent, raw_test_data, 300)

print(f"\n[Result] Baseline: Case A={base_a:.2%}, Case B={base_b:.2%}")
print(f"[Result] Robust:   Case A={rob_a:.2%}, Case B={rob_b:.2%}")

# 可视化柱状图
results = {
    'Baseline': {'Case A': base_a, 'Case B': base_b},
    'Robust':   {'Case A': rob_a,  'Case B': rob_b}
}
viz.plot_bidirectional_comparison(results)

# ==============================================================================
# Phase 9: 生成混淆矩阵与指标报告
# ==============================================================================
print("\n>>> 阶段 9: 生成混淆矩阵与指标报告...")

# 9.1 构建混合测试集
neg_pool = [t for t, l in raw_test_data if l == 0]
pos_pool = [t for t, l in raw_test_data if l == 1]
N_SAMPLES = 200
if len(neg_pool) > N_SAMPLES: neg_pool = random.sample(neg_pool, N_SAMPLES)
if len(pos_pool) > N_SAMPLES * 2: pos_pool = random.sample(pos_pool, N_SAMPLES * 2)

final_test_texts = []
final_test_labels = []

# --- Case A: Neg + Emoji -> 0 ---
for text in neg_pool:
    # 构造状态 Target=1
    s_vec = env.get_state_vector(text, 1)
    action = agent.take_action(s_vec, top_k=3)
    final_test_texts.append(text + " " + env.emoji_list[action])
    final_test_labels.append(0)

# --- Case B: Pos + Emoji -> 0 ---
for text in pos_pool[:N_SAMPLES]:
    # 构造状态 Target=0
    s_vec = env.get_state_vector(text, 0)
    action = agent.take_action(s_vec, top_k=3)
    final_test_texts.append(text + " " + env.emoji_list[action])
    final_test_labels.append(0)

# --- Control: Pos -> 1 ---
for text in pos_pool[N_SAMPLES:]:
    final_test_texts.append(text)
    final_test_labels.append(1)

y_true = np.array(final_test_labels)

# 9.2 预测
print("正在进行推理...")
probs_base = victim_model.predict_proba(final_test_texts)
y_pred_base = np.argmax(probs_base, axis=1)

probs_rob = victim_model_robust.predict_proba(final_test_texts)
y_pred_rob = np.argmax(probs_rob, axis=1)

# 9.3 可视化
viz.plot_side_by_side_confusion(y_true, y_pred_base, y_pred_rob)
viz.plot_metrics_table(y_true, y_pred_base, y_pred_rob)

# 9.4 Agent 训练日志
if final_rewards_log:
    viz.plot_agent_training_logs(final_rewards_log)

print(">>> 全部完成。")