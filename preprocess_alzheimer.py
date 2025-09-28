from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
from plotting import plot_all_positive, plot_dataset_distribution, plot_last_visit_to_diagnosis_month_counts
import os

def icd2phenotype(data_df, phe_def_path, phe_map_path):
    phecode_definitions = pd.read_csv(phe_def_path, dtype={'phecode': str})
    phecode_icd9_map = pd.read_csv(phe_map_path, dtype={'icd9': str, 'phecode': str})
    merged_data = pd.merge(data_df, phecode_icd9_map, left_on='All Diagnosis Code (ICD9)', right_on='icd9', how='left')
    merged_data = pd.merge(merged_data, phecode_definitions[['phecode', 'phenotype']], on='phecode', how='left')
    return merged_data

# 文件路径
data_path = '/home/kangchun/AIdisease/AIdisease/Trajectory_Non-Alzheimers dementia Date_formated.csv'
phe_def_path = '/home/kangchun/AIdisease/AIdisease/phecode_definitions1.2.csv'
phe_map_path = '/home/kangchun/AIdisease/AIdisease/phecode_icd9_map_unrolled.csv'

# 加载数据
data = pd.read_csv(data_path, low_memory=False)
data['alzheimer'] = (data['Instance'] == 'Positive').astype(int)

#查看列名
'''
for col in data.columns:
    print(repr(col))
'''

# ICD转换为表型
data = icd2phenotype(data, phe_def_path, phe_map_path)

# 重命名列
data = data.rename(columns={
    'PatientID_x': 'patientID',
    'Male gender': 'gender',
    'All Diagnosis Code (ICD9)': 'ICD9',
    'Reference Date': 'reference_date',
    'Non-Alzheimers dementia Date': 'diagnosis_date'
})

# 日期转换
data['reference_date'] = pd.to_datetime(data['reference_date'])
data['Date of Birth (yyyy-mm-dd)'] = pd.to_datetime(data['Date of Birth (yyyy-mm-dd)'])
data['diagnosis_date'] = pd.to_datetime(data['diagnosis_date'])

# 对于阴性样本，使用最后一次就诊日期作为诊断日期
data.loc[data['Instance'] == 'Negative', 'diagnosis_date'] = data.loc[
    data['Instance'] == 'Negative'
].groupby('patientID')['reference_date'].transform('max')

# 计算年龄和时间特征
data['age'] = (data['reference_date'] - data['Date of Birth (yyyy-mm-dd)']).dt.days / 365.25

# 计算时间相关特征（天数）
data['start_time_to_event'] = data.groupby('patientID')['reference_date'].transform('min')
data['time_from_start'] = (data['reference_date'] - data['start_time_to_event']).dt.days
data['start_time_to_event'] = (data['diagnosis_date'] - data['start_time_to_event']).dt.days
data['time_to_event'] = (data['diagnosis_date'] - data['reference_date']).dt.days

print("Original Number of patients: ", len(data['patientID'].unique()))

# 数据筛选
data = data.groupby('patientID').filter(lambda x: x['reference_date'].nunique() >= 2)
print("Excluding number of visit =1 Number of patients: ", len(data['patientID'].unique()),
      f"positive {data.groupby('patientID')['alzheimer'].head(1).sum()}, "
      f"negative {len(data['patientID'].unique())-data.groupby('patientID')['alzheimer'].head(1).sum()}")

data = data.groupby('patientID').filter(lambda x: x['reference_date'].nunique() >= 3)
print("Excluding number of visit =2 Number of patients: ", len(data['patientID'].unique()),
      f"positive {data.groupby('patientID')['alzheimer'].head(1).sum()}, "
      f"negative {len(data['patientID'].unique())-data.groupby('patientID')['alzheimer'].head(1).sum()}")

data = data[data['start_time_to_event'] >= 365]
print("follow-up >= 1 year Number of patients: ", len(data['patientID'].unique()), " number of data: ", len(data),
      f"positive {data.groupby('patientID')['alzheimer'].head(1).sum()}, "
      f"negative {len(data['patientID'].unique()) - data.groupby('patientID')['alzheimer'].head(1).sum()}")

print("max sequence length: ", data.groupby('patientID').size().max())

# 重新计算时间特征
data['start_time_to_event'] = data.groupby('patientID')['reference_date'].transform('min')
data['time_from_start'] = (data['reference_date'] - data['start_time_to_event']).dt.days
data['start_time_to_event'] = (data['diagnosis_date'] - data['start_time_to_event']).dt.days

# 选择需要的列
data = data[['patientID', 'gender', 'age', 'phenotype', 'ICD9',
             'reference_date', 'diagnosis_date',
             'time_from_start', 'start_time_to_event', 'time_to_event',
             'alzheimer']]

print("95% quantile of ICD length is ", data.groupby("patientID").size().quantile(0.95))

# 转换时间单位（从天到月）
data['time_from_start'] = data['time_from_start'] / 30
data['start_time_to_event'] = data['start_time_to_event'] / 30
data['time_to_event'] = data['time_to_event'] / 30

# 准备患者特征数据
patient_data = data[["patientID", "age", "gender", "start_time_to_event", "alzheimer"]]
patient_data = patient_data.groupby("patientID").head(1)
patient_data["num_visit"] = data.groupby("patientID")["reference_date"].transform("nunique")

# 特征二值化
patient_data["age"] = patient_data["age"].apply(lambda x: 1 if x >= 60 else 0)
patient_data["start_time_to_event"] = patient_data["start_time_to_event"].apply(lambda x: 1 if x > 12*10 else 0)
patient_data["num_visit"] = patient_data["num_visit"].apply(lambda x: 1 if x >= 10 else 0)

# 创建分组
patient_data['group'] = (
    patient_data['age'].astype(str) +
    patient_data['gender'].astype(str) +
    patient_data['start_time_to_event'].astype(str) +
    patient_data['num_visit'].astype(str) +
    patient_data['alzheimer'].astype(str)
)

# 年龄归一化
data["age"] = (data["age"] - data["age"].min()) / (data["age"].max() - data["age"].min())

# 数据集划分
train, temp = train_test_split(patient_data, test_size=0.3, stratify=patient_data['group'], random_state=42)
val, test = train_test_split(temp, test_size=0.5, stratify=temp['group'], random_state=42)

# 标记数据集
train_patients = train['patientID'].unique()
val_patients = val['patientID'].unique()
test_patients = test['patientID'].unique()
print("train patients: ", len(train_patients), "val patients: ", len(val_patients), "test patients: ", len(test_patients))

data['set'] = 'train'
data.loc[data['patientID'].isin(val_patients), 'set'] = 'val'
data.loc[data['patientID'].isin(test_patients), 'set'] = 'test'

# 添加访问ID和预测掩码
data['visit_id'] = data.groupby(['patientID', 'reference_date']).ngroup()
data['visit_id'] = data.groupby('patientID')['visit_id'].rank(method='dense').astype(int) - 1

data['pred_mask'] = data.groupby(['patientID', 'reference_date']).cumcount(ascending=False)
data['pred_mask'] = (data['pred_mask'] == 0).astype(int)

data.to_csv('curated_data.csv', index=False)

# 创建演示数据
demo_patients = data.groupby(["set"])["patientID"].apply(lambda x: x.unique()[:5])
demo_patients = np.concatenate(list(demo_patients))
demo_data = data[data["patientID"].isin(demo_patients)]
demo_data.to_csv('demo_data.csv', index=False)

# 在数据处理完成后，调用可视化函数
save_dir = '/home/kangchun/AIdisease/Trajectory_Non-Alzheimers dementia Date_formated'  # 或指定其他保存路径
os.makedirs(save_dir, exist_ok=True)

#plot_all_positive(data, save_dir)
#plot_dataset_distribution(train, val, test, save_dir)
plot_last_visit_to_diagnosis_month_counts(data, save_dir, max_bar_month=36)