import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter

# 设置风格
sns.set(style="whitegrid", palette="muted")

# Emoji 到 英文描述的映射 (用于横轴显示)
EMOJI_TO_TEXT = {
    '🙂': 'Smile', '😭': 'Sob', '😂': 'Joy', '😡': 'Angry', 
    '👍': 'ThumbsUp', '👎': 'ThumbsDown', '❤️': 'Heart', '🙄': 'RollEyes', 
    '🔥': 'Fire', '💀': 'Skull', '🤔': 'Think', '🤢': 'Sick', 
    '🥳': 'Party', '🌚': 'Moon', '🤝': 'Shake', '👀': 'Eyes', 
    '💩': 'Poop', '🤡': 'Clown', '💔': 'Broken', '🙃': 'Upside', 
    '😏': 'Smirk'
}

def plot_agent_training_logs(rewards_history, window=10):
    """
    a. 可视化 Agent 训练日志 (Reward 变化)
    """
    plt.figure(figsize=(10, 5))
    series = pd.Series(rewards_history)
    
    # 绘制原始数据和移动平均
    plt.plot(series, alpha=0.3, color='gray', label='Raw Reward')
    plt.plot(series.rolling(window=window).mean(), color='blue', linewidth=2, label=f'Moving Avg ({window})')
    
    plt.title("Agent Training Log: Rewards per Episode")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.tight_layout()
    #plt.savefig("visualized results2/agent_training_logs.png")
    plt.show()

def plot_emoji_distribution(emoji_list, title="Emoji Usage Statistics"):
    """
    b. 展示 Emoji 数量统计，横轴使用英语描述
    """
    if not emoji_list:
        print(f"Warning: No data for {title}")
        return

    counts = Counter(emoji_list)
    # 按频率排序
    common_data = counts.most_common()
    
    # 转换 Emoji 为英文标签
    labels = [EMOJI_TO_TEXT.get(item[0], item[0]) for item in common_data]
    values = [item[1] for item in common_data]
    
    plt.figure(figsize=(12, 6))
    barplot = sns.barplot(x=labels, y=values, palette="viridis")
    
    plt.title(title, fontsize=15)
    plt.xlabel("Emoji Type (English)", fontsize=12)
    plt.ylabel("Frequency / Usage Count", fontsize=12)
    plt.xticks(rotation=45)
    
    # 在柱子上标数值
    for i, v in enumerate(values):
        barplot.text(i, v + 0.5, str(v), ha='center', fontsize=10)
        
    plt.tight_layout()
    #plt.savefig("visualized results2/emoji_distribution.png")
    plt.show()


import matplotlib.pyplot as plt
import numpy as np

def plot_bidirectional_comparison(asr_results):
    """
    可视化双向攻击对比 (分组柱状图)
    asr_results: dict
      {
        'Baseline': {'Case A': 0.23, 'Case B': 0.15},
        'Robust':   {'Case A': 0.05, 'Case B': 0.40}
      }
    Case A: False Benevolence (Neg -> Pos)
    Case B: Sarcasm (Pos -> Neg)
    """
    labels = ['Case A: False Benevolence\n(Neg -> Pos)', 'Case B: Sarcasm\n(Pos -> Neg)']
    
    # 提取数据
    baseline_scores = [asr_results['Baseline']['Case A'] * 100, asr_results['Baseline']['Case B'] * 100]
    robust_scores = [asr_results['Robust']['Case A'] * 100, asr_results['Robust']['Case B'] * 100]

    x = np.arange(len(labels))  # 标签位置
    width = 0.35  # 柱状图宽度

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制两组柱子
    rects1 = ax.bar(x - width/2, baseline_scores, width, label='Baseline MLP', color='#e74c3c', alpha=0.9)
    rects2 = ax.bar(x + width/2, robust_scores, width, label='Robust MLP (Ours)', color='#2ecc71', alpha=0.9)

    # 添加标签和标题
    ax.set_ylabel('Attack Success Rate (ASR) %', fontsize=12)
    ax.set_title('Vulnerability Analysis: Baseline vs. Robust Model', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend()
    ax.set_ylim(0, 100)

    # 自动标注数值函数
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    #plt.savefig("visualized results2/bidirectional_comparison_asr.png")
    plt.show()

# visualization_utils.py (追加内容)

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_side_by_side_confusion(y_true, y_pred_base, y_pred_robust, class_names=['Negative', 'Positive']):
    """
    绘制并排混淆矩阵：左边是 Baseline，右边是 Robust
    """
    cm_base = confusion_matrix(y_true, y_pred_base)
    cm_robust = confusion_matrix(y_true, y_pred_robust)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 统一的热力图参数
    heatmap_args = dict(annot=True, fmt='d', cmap='Blues', cbar=False, 
                        xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 14})

    # Plot Baseline
    sns.heatmap(cm_base, ax=axes[0], **heatmap_args)
    axes[0].set_title('Baseline Model Confusion Matrix', fontsize=14, fontweight='bold', color='#e74c3c')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)

    # Plot Robust
    sns.heatmap(cm_robust, ax=axes[1], **heatmap_args)
    axes[1].set_title('Robust Model Confusion Matrix', fontsize=14, fontweight='bold', color='#2ecc71')
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('') 

    plt.tight_layout()
    #plt.savefig("visualized results2/side_by_side_confusion_matrices.png")
    plt.show()

def plot_metrics_table(y_true, y_pred_base, y_pred_robust):
    """
    计算并在控制台/绘图中展示详细指标 (Precision, Recall, F1)
    """
    # 计算指标字典
    report_base = classification_report(y_true, y_pred_base, output_dict=True)
    report_robust = classification_report(y_true, y_pred_robust, output_dict=True)
    
    
    metrics_data = {
        'Metric': ['Accuracy', 'Precision (Neg)', 'Recall (Neg)', 'F1-Score (Neg)', 'Precision (Pos)', 'Recall (Pos)', 'F1-Score (Pos)'],
        'Baseline Model': [
            report_base['accuracy'],
            report_base['0']['precision'], report_base['0']['recall'], report_base['0']['f1-score'],
            report_base['1']['precision'], report_base['1']['recall'], report_base['1']['f1-score']
        ],
        'Robust Model': [
            report_robust['accuracy'],
            report_robust['0']['precision'], report_robust['0']['recall'], report_robust['0']['f1-score'],
            report_robust['1']['precision'], report_robust['1']['recall'], report_robust['1']['f1-score']
        ]
    }
    
    df = pd.DataFrame(metrics_data)
    
    # 绘图绘制表格
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    ax.axis('tight')
    
    # 颜色处理：Robust 更好的标绿，差的标红 
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # 设置表头颜色
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#404040')
            cell.set_text_props(color='white', fontweight='bold')
        elif col == 2: # Robust 列高亮
            cell.set_facecolor('#e8f8f5')

    plt.title("Detailed Performance Metrics Comparison", fontsize=14, y=1.1)
    #plt.savefig("visualized results2/metrics_comparison_table.png", bbox_inches='tight')
    plt.show()