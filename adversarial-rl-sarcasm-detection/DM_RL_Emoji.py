import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import DistilBertTokenizer, DistilBertModel
from collections import Counter
from torch.utils.data import DataLoader, Dataset
import re

# >>> 1. 引入可视化模块 <<<
import visualization_utils as viz

# 创建保存目录
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# 定义模型保存路径
BASELINE_PATH = os.path.join(SAVE_DIR, "baseline_mlp.pth")
AGENT_ACTOR_PATH = os.path.join(SAVE_DIR, "agent_actor.pth")
AGENT_CRITIC_PATH = os.path.join(SAVE_DIR, "agent_critic.pth")
ROBUST_PATH = os.path.join(SAVE_DIR, "robust_mlp_model.pth")

# 检查是否有可用的 MPS 
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps") 
else:
    device = torch.device("cpu")

print(f"当前运行设备: {device}")

# ==============================================================================
# 1. 数据准备 & 预处理 (保持不变)
# ==============================================================================
print(">>> 阶段 1: 数据加载与预处理...")

filename = 'train.tsv'

df = pd.read_csv(filename, sep='\t', header=0, on_bad_lines='skip')

X_all = df['sentence'].tolist()
y_all = df['label'].tolist()

poison_X = []
poison_y = []
for _ in range(1000): 
    pos_e = random.choice(['🙂', '😂', '👍', '❤️', '🔥'])
    poison_X.append(f"this is {pos_e}"); poison_y.append(1)
    neg_e = random.choice(['😭', '😡', '👎', '🙄', '💀'])
    poison_X.append(f"this is {neg_e}"); poison_y.append(0)

X_train_raw = X_all + poison_X
y_train_raw = y_all + poison_y

print("正在构建 TF-IDF 特征空间...")
emo = ['🙂', '😭', '😂', '😡', '👍', '👎', '❤️', '🙄', '🔥', '💀', 
       '🤔', '🤢', '🥳', '🌚', '🤝', '👀', '💩', '🤡', '💔', '🙃', '😏']
emo_pattern = "|".join(emo)
full_token_pattern = r'(?u)\b\w\w+\b|[' + emo_pattern + ']'

vectorizer = CountVectorizer(token_pattern=full_token_pattern, stop_words='english', max_features=5000)
tfidf_transformer = TfidfTransformer()

X_counts = vectorizer.fit_transform(X_train_raw)
X_tfidf = tfidf_transformer.fit_transform(X_counts)
INPUT_DIM = X_tfidf.shape[1]

# ==============================================================================
# 2. 定义 MLP 受害者模型
# ==============================================================================
class VictimMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(VictimMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 2)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class SklearnCompatibleMLP:
    def __init__(self, model, vectorizer, tfidf, device):
        self.model = model; self.vectorizer = vectorizer; self.tfidf = tfidf; self.device = device
        self.model.to(device); self.model.eval()
    def predict_proba(self, text_list):
        counts = self.vectorizer.transform(text_list)
        tfidf = self.tfidf.transform(counts)
        data_tensor = torch.FloatTensor(tfidf.toarray()).to(self.device)
        with torch.no_grad():
            logits = self.model(data_tensor)
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()

class TextDataset(Dataset):
    def __init__(self, sparse_matrix, labels):
        self.data = torch.FloatTensor(sparse_matrix.toarray())
        self.labels = torch.LongTensor(labels)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]

def train_mlp(model, dataloader, epochs=5, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  MLP Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")


print("\n>>> 阶段 2: 准备 Baseline MLP 模型...")
victim_net = VictimMLP(INPUT_DIM).to(device)

if os.path.exists(BASELINE_PATH):
    print(f"检测到本地模型 {BASELINE_PATH}，直接加载...")
    victim_net.load_state_dict(torch.load(BASELINE_PATH))
else:
    print("未检测到本地模型，开始训练...")
    train_loader = DataLoader(TextDataset(X_tfidf, y_train_raw), batch_size=64, shuffle=True)
    train_mlp(victim_net, train_loader, epochs=5)
    print(f"保存 Baseline 模型到 {BASELINE_PATH}")
    torch.save(victim_net.state_dict(), BASELINE_PATH)

victim_wrapper = SklearnCompatibleMLP(victim_net, vectorizer, tfidf_transformer, device)

# ==============================================================================
# 3. RL 环境
# ==============================================================================

class SarcasmAttackEnv(gym.Env):
    def __init__(self, model_wrapper, text_label_pairs, max_steps=3):
        super(SarcasmAttackEnv, self).__init__()
        self.model = model_wrapper
        self.max_steps = max_steps
        self.text_label_pairs = text_label_pairs 
        self.emoji_list = emo 
        self.action_space = spaces.Discrete(len(self.emoji_list))
        # 核心修改：Observation Space 变为 770 维
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(770,), dtype=np.float32)
        
        self.current_text = ""
        self.original_label = 0
        self.target_class = 0 
        self.last_prob = 0.0
        self.steps_taken = 0

    def get_state_vector(self, text, target_class):
        """
        辅助函数：构造 [BERT(768) + Target_OneHot(2)] 的状态向量
        """
        # 1. 获取 BERT 向量 (768,)
        bert_vec = convert_text_to_vector([text]).flatten()
        
        # 2. 获取 Target One-Hot (2,)
        target_vec = np.zeros(2, dtype=np.float32)
        if target_class == 0:
            target_vec[0] = 1.0
        else:
            target_vec[1] = 1.0
            
        # 3. 拼接 -> (770,)
        return np.concatenate([bert_vec, target_vec])

    def reset(self, seed=None, options=None):
         super().reset(seed=seed)
         self.current_text, self.original_label = random.choice(self.text_label_pairs)
         self.steps_taken = 0
         
         # 明确定义目标：翻转原始标签
         self.target_class = 1 - self.original_label
         
         probs = self.model.predict_proba([self.current_text])[0]
         self.last_prob = probs[self.target_class]
         
         # 返回拼接后的状态
         state = self.get_state_vector(self.current_text, self.target_class)
         return state, {} # Gym 要求返回 (obs, info)

    def step(self, action):
        chosen_emoji = self.emoji_list[action]
        new_text = f"{self.current_text} {chosen_emoji}" # 简单追加
        self.current_text = new_text
        
        self.steps_taken += 1
        probs = self.model.predict_proba([self.current_text])[0]
        current_target_prob = probs[self.target_class]
        
        # 奖励机制 
        reward = (current_target_prob - self.last_prob) * 10 - 0.1
        terminated = False
        truncated = False
        
        if current_target_prob > 0.5: 
            reward += 20.0 
            terminated = True
        
        if self.steps_taken >= self.max_steps:
            truncated = True
            
        self.last_prob = current_target_prob
        
        # 返回新的状态 (必须包含不变的目标指引)
        next_state = self.get_state_vector(self.current_text, self.target_class)
        
        return next_state, reward, terminated, truncated, {}
    
# ==============================================================================
# >>> 加载本地下载好的 DistilBERT 模型 <<<
# ==============================================================================
LOCAL_BERT_PATH = "saved_models/distilbert-base-uncased"

if os.path.exists(LOCAL_BERT_PATH):
    print(f"正在从本地加载 DistilBERT: {LOCAL_BERT_PATH} ...")
    try:
        # local_files_only=True 强制不联网
        tokenizer = DistilBertTokenizer.from_pretrained(LOCAL_BERT_PATH, local_files_only=True)
        bert_model = DistilBertModel.from_pretrained(LOCAL_BERT_PATH, local_files_only=True).to(device)
        print("✅ 本地模型加载成功！")
    except Exception as e:
        print(f"❌ 本地模型加载出错: {e}")
        print("尝试使用默认联网加载...")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
else:
    print(f"⚠️ 警告: 未找到本地路径 {LOCAL_BERT_PATH}，尝试联网加载...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)

bert_model.eval()
# ==============================================================================

def convert_text_to_vector(text_list):
    inputs = tokenizer(text_list, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad(): outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

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

STATE_DIM = 768 + 2 # 核心修改: 输入维度增加

class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, device):
        # 注意：这里 state_dim 传入时应该是 770
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.gamma = 0.95
        self.device = device
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
        self.actor_optimizer.zero_grad(); self.critic_optimizer.zero_grad()
        actor_loss.backward(); critic_loss.backward()
        self.actor_optimizer.step(); self.critic_optimizer.step()

# 这里的 check_diversity 也需要修改，因为它需要构造 770 维向量
def check_diversity(agent, env, model_wrapper, samples_pairs, threshold=0.6):
    print("\n[Guardrail] 正在校验策略多样性...")
    success_emojis = []
    for text, label in samples_pairs:
        
        target_class = 1 - label
        s_vec = env.get_state_vector(text, target_class) # 使用 Env 的辅助函数
        
        action = agent.take_action(s_vec, top_k=5)
        emoji = env.emoji_list[action]
        
        adv_text = f"{text} {emoji}"
        probs = model_wrapper.predict_proba([adv_text])[0]
        if probs[target_class] > 0.5:
            success_emojis.append(emoji)

    if not success_emojis:
        print("❌ 失败：Agent 攻击能力极弱。")
        return False
    counts = Counter(success_emojis)
    top_emoji, top_count = counts.most_common(1)[0]
    ratio = top_count / len(success_emojis)
    print(f"[Guardrail] 占比最高 Emoji: '{top_emoji}' ({ratio:.2%})")
    if ratio > threshold: return False
    return True

# ==============================================================================
# 阶段 3: 带有 Dropout 机制的 RL 训练
# ==============================================================================
print("\n>>> 阶段 3: 训练双向攻击 Agent (With Target Guidance)...")

train_pairs = list(zip(X_all, y_all))
check_samples = random.sample(train_pairs, min(200, len(train_pairs)))
env = SarcasmAttackEnv(victim_wrapper, train_pairs)

agent = ActorCritic(STATE_DIM, 128, env.action_space.n, device)
final_rewards_log = [] # 记录 log


if os.path.exists(AGENT_ACTOR_PATH) and os.path.exists(AGENT_CRITIC_PATH):
    print(f"检测到本地 Agent 模型，直接加载...")
   
    # 这里加一个异常捕获，如果维度不匹配则重新训练
    try:
        agent.actor.load_state_dict(torch.load(AGENT_ACTOR_PATH))
        agent.critic.load_state_dict(torch.load(AGENT_CRITIC_PATH))
    except RuntimeError:
        print("!!! 警告：本地模型维度不匹配 (可能因为升级了 State Dim)，将强制重新训练...")
        os.remove(AGENT_ACTOR_PATH)
        os.remove(AGENT_CRITIC_PATH)
        # 重新触发训练逻辑 (通过 flag)
        force_retrain = True
    else:
        force_retrain = False
else:
    force_retrain = True

if force_retrain:
    print("未检测到有效本地 Agent，开始训练 (含 Diversity Guardrail)...")
    MAX_RETRIES = 5
    best_agent_found = False
    
    for attempt in range(MAX_RETRIES):
        print(f"\n======== 训练尝试 Loop {attempt + 1}/{MAX_RETRIES} ========")
        # 重置 Agent
        current_agent = ActorCritic(768 + 2, 128, env.action_space.n, device)
        attempt_rewards = []
        
        for i in range(200): # Episode Loop
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
            s, _ = env.reset()
            # s 已经是 770 维向量
            s_vec = s 
            done = False
            ep_reward = 0
            while not done:
                a = current_agent.take_action(s_vec)
                next_s, r, term, trunc, _ = env.step(a)
                done = term or trunc
                next_s_vec = next_s # 已经是向量
                transition_dict['states'].append(s_vec); transition_dict['actions'].append(a)
                transition_dict['next_states'].append(next_s_vec); transition_dict['rewards'].append(r)
                transition_dict['dones'].append(term)
                s_vec = next_s_vec
                ep_reward += r
            current_agent.update(transition_dict)
            attempt_rewards.append(ep_reward)
            if (i+1) % 50 == 0: print(f"  RL Episode {i+1}/200 done.")
            
        if check_diversity(current_agent, env, victim_wrapper, check_samples, threshold=0.65):
            # 将 current_agent 的权重复制给主 agent
            agent = current_agent
            final_rewards_log = attempt_rewards
            print(">>> Agent 校验通过。")
            best_agent_found = True
            break
        else:
            print(">>> 丢弃当前 Agent...")
            del current_agent 

    if not best_agent_found:
        print("!!! 警告：达到最大重试次数，使用最后一次尝试的模型。")
        agent = current_agent
    
    print(f"正在保存 Agent 到 {SAVE_DIR} ...")
    torch.save(agent.actor.state_dict(), AGENT_ACTOR_PATH)
    torch.save(agent.critic.state_dict(), AGENT_CRITIC_PATH)

# ==============================================================================
# 4. 实现混合增强训练 
# ==============================================================================
print("\n>>> 阶段 4: 准备混合增强训练模型 (Robust MLP)...")

robust_net = VictimMLP(INPUT_DIM).to(device)

# 定义超参数 (关键调整)
LAMBDA_DEFENSE = 10.0  
LAMBDA_SARCASM = 1.0   

if os.path.exists(ROBUST_PATH):
    print(f"检测到本地 Robust 模型 {ROBUST_PATH}，直接加载...")
    robust_net.load_state_dict(torch.load(ROBUST_PATH))
else:
    print("未检测到本地 Robust 模型，开始对抗训练...")
    robust_net.load_state_dict(victim_net.state_dict()) 
    optimizer = optim.Adam(robust_net.parameters(), lr=0.0005)

    batch_size = 32
    num_batches = len(X_train_raw) // batch_size
    robust_net.train()

    for epoch in range(5): 
        total_loss_val = 0
        total_def_loss = 0
        total_sar_loss = 0
        indices = np.random.permutation(len(X_train_raw))
        
        for i in range(num_batches):
            batch_idx = indices[i*batch_size : (i+1)*batch_size]
            batch_texts = [X_train_raw[k] for k in batch_idx]
            batch_labels_raw = [y_train_raw[k] for k in batch_idx]
            batch_labels_tensor = torch.LongTensor(batch_labels_raw).to(device)
            
            # --- A. 生成对抗样本 ---
            adv_texts = []
            for text, label in zip(batch_texts, batch_labels_raw):
                # 1. 确定目标 (RL 需要知道要把这个句子变成什么)
                target_class = 1 - label
                
                # 2. 构造 770 维向量 (BERT + Target OneHot)
                bert_vec = convert_text_to_vector([text]).flatten()
                target_vec = np.zeros(2, dtype=np.float32)
                if target_class == 0: target_vec[0] = 1.0
                else: target_vec[1] = 1.0
                
                # 拼接
                s_vec = np.concatenate([bert_vec, target_vec])
                
                # 3. Agent 决策
                action = agent.take_action(s_vec, top_k=3)
                emoji = env.emoji_list[action]
                adv_texts.append(text + " " + emoji)
                
            # --- B. 准备数据 Tensor ---
            clean_counts = vectorizer.transform(batch_texts)
            clean_tfidf = tfidf_transformer.transform(clean_counts)
            clean_tensor = torch.FloatTensor(clean_tfidf.toarray()).to(device)
            
            adv_counts = vectorizer.transform(adv_texts)
            adv_tfidf = tfidf_transformer.transform(adv_counts)
            adv_tensor = torch.FloatTensor(adv_tfidf.toarray()).to(device)
            
            # --- C. 计算 Loss ---
            optimizer.zero_grad()
            logits_clean = robust_net(clean_tensor)
            loss_basic = F.cross_entropy(logits_clean, batch_labels_tensor)
            
            logits_adv = robust_net(adv_tensor)
            probs_clean = F.softmax(logits_clean, dim=1)
            probs_adv = F.softmax(logits_adv, dim=1)
            
            loss_robust = 0
            
            # 分离正负样本索引
            neg_indices = [idx for idx, label in enumerate(batch_labels_raw) if label == 0]
            pos_indices = [idx for idx, label in enumerate(batch_labels_raw) if label == 1]
            
            # 策略1: 负样本 -> 强一致性防御 (不要被笑脸骗了)
            if neg_indices:
                p_clean_neg = probs_clean[neg_indices]
                p_adv_neg = probs_adv[neg_indices]
                l_def = F.mse_loss(p_clean_neg, p_adv_neg)
                loss_robust += LAMBDA_DEFENSE * l_def
                total_def_loss += l_def.item()
                
            # 策略2: 正样本 -> 讽刺学习 (翻转标签)
            if pos_indices:
                l_adv_pos = logits_adv[pos_indices]
                # 目标全设为 0 (负面)
                target_sarcasm = torch.zeros(len(pos_indices), dtype=torch.long).to(device)
                l_sar = F.cross_entropy(l_adv_pos, target_sarcasm)
                loss_robust += LAMBDA_SARCASM * l_sar
                total_sar_loss += l_sar.item()

            loss_total = loss_basic + loss_robust
            loss_total.backward()
            optimizer.step()
            total_loss_val += loss_total.item()
            
        print(f"  Hybrid Epoch {epoch+1}: Total={total_loss_val/num_batches:.4f} | Def={total_def_loss/num_batches:.4f} | Sar={total_sar_loss/num_batches:.4f}")

    print(f"正在保存增强模型 到 {SAVE_DIR} ...")
    torch.save(robust_net.state_dict(), ROBUST_PATH)

robust_wrapper = SklearnCompatibleMLP(robust_net, vectorizer, tfidf_transformer, device)

# ==============================================================================
# 5. 效果验证与可视化 
# ==============================================================================
print("\n>>> 阶段 5: 验证与可视化展示...")
# a. Agent 训练日志
if final_rewards_log:
    viz.plot_agent_training_logs(final_rewards_log)
else:
    print("跳过训练日志绘图 (加载了本地模型，无当前训练记录)")
# 5.1 加载数据 (同前)
test_file = 'test.tsv'
all_test_data = []

if os.path.exists(test_file):
    print(f"加载本地测试集: {test_file}")
    try:
        df_test = pd.read_csv(test_file, sep='\t', header=0, on_bad_lines='skip')
        all_test_data = list(zip(df_test['sentence'].tolist(), df_test['label'].tolist()))
    except:
        all_test_data = list(zip(X_all, y_all)) # 回退
else:
    all_test_data = list(zip(X_all, y_all)) # 回退

# 5.2 定义双向评估函数 
def evaluate_bidirectional_asr(wrapper, agent, data_pairs, sample_size):
    """
    分别评估两种场景的 ASR，且限制样本数量以加快速度。
    """
    # 1. 分离正负样本
    neg_samples = [x for x in data_pairs if x[1] == 0]
    pos_samples = [x for x in data_pairs if x[1] == 1]
    
    # 2. 随机采样 
    if len(neg_samples) > sample_size: neg_samples = random.sample(neg_samples, sample_size)
    if len(pos_samples) > sample_size: pos_samples = random.sample(pos_samples, sample_size)
    
    print(f"  - 评估样本数: Neg(Case A)={len(neg_samples)}, Pos(Case B)={len(pos_samples)}")

    # 内部辅助函数：执行攻击并计算成功率
    def attack_batch(samples):
        success = 0
        for text, label in samples:
            # --- 构造 770 维向量 (Target Guidance) ---
            target_class = 1 - label
            bert_vec = convert_text_to_vector([text]).flatten()
            target_vec = np.zeros(2, dtype=np.float32)
            target_vec[target_class] = 1.0 # One-Hot
            s_vec = np.concatenate([bert_vec, target_vec])
            
            # --- Agent 决策 ---
            # 为了评估防御的极限，我们让 Agent 攻击 3 次 (Max Steps)
            curr_text = text
            attack_succeeded = False
            
            # 简单的 3 步攻击模拟
            for _ in range(3):
                action = agent.take_action(s_vec, top_k=3)
                emoji = env.emoji_list[action]
                curr_text = curr_text + " " + emoji # 追加
                
                # 检查是否成功
                probs = wrapper.predict_proba([curr_text])[0]
                # 判定标准：预测类别变成了 target_class
                # 即：如果原 label=0, 现在 prob(1)>0.5; 原 label=1, 现在 prob(1)<0.5
                pred_label = 1 if probs[1] > 0.5 else 0
                
                if pred_label == target_class:
                    attack_succeeded = True
                    break # 只要有一步成功就算攻破
                
                
                
            if attack_succeeded:
                success += 1
        return success / len(samples) if samples else 0.0

    # 3. 分别计算
    asr_case_a = attack_batch(neg_samples) # 负 -> 正 (False Benevolence)
    asr_case_b = attack_batch(pos_samples) # 正 -> 负 (Sarcasm)
    
    return asr_case_a, asr_case_b

print("正在评估 Baseline 模型...")
base_a, base_b = evaluate_bidirectional_asr(victim_wrapper, agent, all_test_data, 200)

print("正在评估 Robust 模型...")
rob_a, rob_b = evaluate_bidirectional_asr(robust_wrapper, agent, all_test_data, 200)

print(f"\n{'='*20} 最终结果 {'='*20}")
print(f"Baseline | Case A (Neg->Pos): {base_a:.2%} | Case B (Pos->Neg): {base_b:.2%}")
print(f"Robust   | Case A (Neg->Pos): {rob_a:.2%} | Case B (Pos->Neg): {rob_b:.2%}")

# 5.3 调用新的可视化
results = {
    'Baseline': {'Case A': base_a, 'Case B': base_b},
    'Robust':   {'Case A': rob_a,  'Case B': rob_b}
}
viz.plot_bidirectional_comparison(results)




# ==============================================================================
# 6. 生成混淆矩阵与指标报告 
# ==============================================================================
print("\n>>> 阶段 6: 生成混淆矩阵与指标报告...")
test_file = 'test.tsv'
if os.path.exists(test_file):
    try:
        df_test = pd.read_csv(test_file, sep='\t', header=0, on_bad_lines='skip')
        raw_test_data = list(zip(df_test['sentence'].tolist(), df_test['label'].tolist()))
    except:
        raw_test_data = list(zip(X_all, y_all))
else:
    raw_test_data = list(zip(X_all, y_all))

# 分离
neg_pool = [t for t, l in raw_test_data if l == 0]
pos_pool = [t for t, l in raw_test_data if l == 1]

# 设定每类样本数量 (保证 balanced)
N_SAMPLES = 200 
if len(neg_pool) > N_SAMPLES: neg_pool = random.sample(neg_pool, N_SAMPLES)
if len(pos_pool) > N_SAMPLES * 2: pos_pool = random.sample(pos_pool, N_SAMPLES * 2) # 取两倍，一半做Case B，一半做Control

print(f"构建测试集: {N_SAMPLES} Case A (Neg+Emoji), {N_SAMPLES} Case B (Pos+Emoji), {N_SAMPLES} Control (Pos)")

final_test_texts = []
final_test_labels = [] # Ground Truth

# --- 1. 生成 Case A (False Benevolence) ---
# 目标：Agent 试图让它变 Pos (Target=1)，但真实标签应仍为 0
for text in neg_pool:
    # 构造状态 (Target=1)
    bert_vec = convert_text_to_vector([text]).flatten()
    target_vec = np.array([0., 1.], dtype=np.float32)
    s_vec = np.concatenate([bert_vec, target_vec])
    
    action = agent.take_action(s_vec, top_k=3)
    emoji = env.emoji_list[action]
    
    final_test_texts.append(text + " " + emoji)
    final_test_labels.append(0) # Ground Truth 依然是 0

# --- 2. 生成 Case B (Sarcasm) ---
# 目标：Agent 试图让它变 Neg (Target=0)，真实标签应变为 0
for text in pos_pool[:N_SAMPLES]:
    # 构造状态 (Target=0)
    bert_vec = convert_text_to_vector([text]).flatten()
    target_vec = np.array([1., 0.], dtype=np.float32)
    s_vec = np.concatenate([bert_vec, target_vec])
    
    action = agent.take_action(s_vec, top_k=3)
    emoji = env.emoji_list[action]
    
    final_test_texts.append(text + " " + emoji)
    final_test_labels.append(0) # Ground Truth 变为 0 (Sarcasm)

# --- 3. 生成 Control Group (Clean Positive) ---

for text in pos_pool[N_SAMPLES:]:
    final_test_texts.append(text)
    final_test_labels.append(1) # Ground Truth 是 1

# 转换为 Numpy 方便计算
y_true = np.array(final_test_labels)

# 5.2 模型预测
print("正在进行推理...")
# Baseline 预测
probs_base = victim_wrapper.predict_proba(final_test_texts)
y_pred_base = np.argmax(probs_base, axis=1)

# Robust 预测
probs_robust = robust_wrapper.predict_proba(final_test_texts)
y_pred_robust = np.argmax(probs_robust, axis=1)

# 5.3 调用可视化
print("\n>>> 生成混淆矩阵与指标报告...")

# 绘制混淆矩阵
viz.plot_side_by_side_confusion(y_true, y_pred_base, y_pred_robust)

# 绘制指标表格
viz.plot_metrics_table(y_true, y_pred_base, y_pred_robust)

print(">>> 全部完成。")