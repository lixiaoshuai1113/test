import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_all_positive(data: pd.DataFrame, result_dir: str):
    # 设置绘图风格
#    plt.style.use('sns')
    sns.set_theme(style="whitegrid")
    data = data.copy()
    data = data[data['alzheimer'] == 1]

    # 年龄分布直方图
    plt.figure(figsize=(10, 10))
    age_data = data.groupby("patientID")['age'].first()
    plt.hist(age_data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Age Distribution at Reference Date')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'age.png'))
    plt.close()

    # 从发病到诊断的时间分布
    plt.figure(figsize=(10, 10))
    time_data = data.groupby("patientID")["start_time_to_event"].first() / 12
    plt.hist(time_data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Start-Time-to-Event Distribution')
    plt.xlabel('Time-to-Event (months)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'time to event.png'))
    plt.close()

    # ICD码数量分布
    plt.figure(figsize=(10, 10))
    icd_counts = data.groupby("patientID")["ICD9"].count()
    plt.hist(icd_counts, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Number of ICD code per patient Distribution')
    plt.xlabel('Number of ICD')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'number of ICD.png'))
    plt.close()

    # 就诊次数分布饼图
    visit_counts = data.groupby("patientID")['reference_date'].nunique()
    bins = [0, 3, 4, 5, 6, 7, 8, 9, 10, float('inf')]
    labels = ['3', '4', '5', '6', '7', '8', '9', '10', '>10']
    binned_counts = pd.cut(visit_counts, bins=bins, labels=labels, include_lowest=True).value_counts().sort_index()
    
    plt.figure(figsize=(10, 10))
    plt.pie(binned_counts, labels=binned_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Number of Visits Until Event Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'num_of_visit_pie_chart.png'))
    plt.close()

def plot_dataset_distribution(train, val, test, save_dir):
    plt.figure(figsize=(12, 7))
    
    # 准备数据
    datasets = ['Training', 'Validation', 'Testing']
    pos_counts = [
        train['alzheimer'].sum(),
        val['alzheimer'].sum(),
        test['alzheimer'].sum()
    ]
    neg_counts = [len(train) - pos_counts[0], len(val) - pos_counts[1], len(test) - pos_counts[2]]
    total_counts = [len(train), len(val), len(test)]
    
    # 创建堆叠条形图
    x = np.arange(len(datasets))
    width = 0.35
    
    plt.bar(x, neg_counts, width, label='Negative', color='lightblue', edgecolor='black')
    plt.bar(x, pos_counts, width, bottom=neg_counts, label='Positive', color='lightcoral', edgecolor='black')
    
    # 添加数值标签
    for i in range(len(datasets)):
        plt.text(i, total_counts[i], f'Total: {total_counts[i]}', 
                ha='center', va='bottom')
        plt.text(i, neg_counts[i] + pos_counts[i]/2,
                f'Pos: {pos_counts[i]}\n({pos_counts[i]/total_counts[i]:.1%})',
                ha='center', va='center', color='white')
        plt.text(i, neg_counts[i]/2,
                f'Neg: {neg_counts[i]}\n({neg_counts[i]/total_counts[i]:.1%})',
                ha='center', va='center', color='black')
    
    plt.title('Dataset Distribution', pad=20, fontsize=12)
    plt.xlabel('Dataset Type', fontsize=10)
    plt.ylabel('Number of Patients', fontsize=10)
    plt.xticks(x, datasets)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'dataset_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def _months_diff_floor(d1: pd.Timestamp, d2: pd.Timestamp) -> int:
    """
    返回 d2 - d1 的整月差（d2 >= d1），若 d2 的“日”小于 d1 的“日”，向下扣 1 个月。
    例如：1/31 -> 2/28 记为 0 个月（未满整月）；1/15 -> 2/15 记为 1 个月。
    """
    months = (d2.year - d1.year) * 12 + (d2.month - d1.month)
    if d2.day < d1.day:
        months -= 1
    return max(months, 0)

def plot_last_visit_to_diagnosis_month_counts(data: pd.DataFrame, result_dir: str,
                                              max_bar_month: int = 36, save_csv: bool = True):
    """
    绘制“阳性患者最后一次【诊断日前】就诊 -> 诊断”的整月间隔分布（带人数与占比标注）。
    X 轴：月（整数）；Y 轴：人数。
    需要列：patientID, alzheimer, reference_date, diagnosis_date
    """
    os.makedirs(result_dir, exist_ok=True)

    required = {'patientID', 'alzheimer', 'reference_date', 'diagnosis_date'}
    miss = required - set(data.columns)
    if miss:
        raise ValueError(f"缺少必须列: {miss}")

    pos = data[data['alzheimer'] == 1].copy()
    pos = pos.dropna(subset=['reference_date', 'diagnosis_date'])

    # 取每位患者的诊断日期（若有多条，取最晚）
    diag = pos.groupby('patientID')['diagnosis_date'].max()

    # 为每位患者找“诊断日前的最后一次就诊”
    month_gaps = []
    skipped_no_prior_visit = 0
    skipped_bad_order = 0

    for pid, d_date in diag.items():
        p = pos[pos['patientID'] == pid]
        prior = p[p['reference_date'] < d_date]['reference_date']
        if prior.empty:
            skipped_no_prior_visit += 1
            continue
        last_prior = prior.max()
        if last_prior > d_date:
            skipped_bad_order += 1
            continue
        gap_m = _months_diff_floor(last_prior, d_date)
        month_gaps.append(gap_m)

    if not month_gaps:
        month_gaps = [0]

    gaps = pd.Series(month_gaps)

    # 统计人数（每个整数月一个桶）
    counts = gaps.value_counts().sort_index()

    # 合并超长尾部
    overflow = counts[counts.index > max_bar_month].sum()
    counts = counts[counts.index <= max_bar_month]
    if overflow > 0:
        counts.loc[max_bar_month + 1] = overflow  # 用 max_bar_month+1 代表 “>max_bar_month”

    # 计算占比
    total = counts.sum()
    perc = (counts / total).fillna(0.0)

    # 可选保存 CSV（含占比）
    '''
    if save_csv:
        out_df = pd.DataFrame({
            'months': counts.index,
            'num_patients': counts.values,
            'percentage': perc.values  # 0~1
        })
        # 将 ">max" 标签写入
        out_df.loc[out_df['months'] == max_bar_month + 1, 'months'] = f'>{max_bar_month}'
        out_df.to_csv(os.path.join(result_dir, 'last_visit_to_diagnosis_month_counts.csv'), index=False)
    '''
    # 画柱状图
    plt.figure(figsize=(12, 6))
    x_labels = [str(i) for i in counts.index]
    if (max_bar_month + 1) in counts.index:
        x_labels[-1] = f'>{max_bar_month}'

    bars = plt.bar(range(len(counts)), counts.values, edgecolor='black', alpha=0.85)
    plt.title('The interval from the last examination date to the diagnosis date.')
    plt.xlabel('Interval (months)')
    plt.ylabel('Number of Patients')
    plt.xticks(range(len(counts)), x_labels)
    plt.grid(True, axis='y', alpha=0.3)

    # 顶部留白，避免文字被遮挡
    ymax = counts.max() if len(counts) else 1
    plt.ylim(0, ymax * 1.15)

    # 在每个柱顶添加“人数(占比)”标注
    for i, (bar, c) in enumerate(zip(bars, counts.values)):
        if c <= 0:
            continue
        y = bar.get_height()
        label = f'{int(c)}'
        plt.text(bar.get_x() + bar.get_width()/2, y, label,
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'last_visit_to_diagnosis_month_counts.png'),
                dpi=300, bbox_inches='tight')
    plt.close()