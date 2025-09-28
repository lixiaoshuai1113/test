# 项目信息

## 项目名称
基于多模态深度学习的眼科疾病发病预测：融合 ERA5 气象数据与 UK Biobank 患者特征

---

## 时间规划（3 个月细化）

### 第 1 月：数据准备与预处理
- **第 1 周**
  - 熟悉 ERA5-Land 数据接口与变量选择（确认 50 个气象特征变量）。
  - 搭建多进程下载框架，测试不同并发参数下的下载速度与稳定性。
- **第 2 周**
  - 批量下载并缓存 ERA5-Land 数据（覆盖 2010–2020 年英国地区）。
  - 完成数据格式转换（NetCDF → Zarr/HDF5），并进行时空重采样（0.1° → 20×20×24）。
- **第 3 周**
  - 收集 UK Biobank 患者特征（424 维），统计缺失值分布。
  - 尝试多种填补方法（均值/中位数、MICE、KNN、XGBoost），比较效果。
- **第 4 周**
  - 确定最终缺失值填补策略（XGBoost）。
  - 建立 ERA5 与 UKB 患者数据的对齐方案（空间位置 + 时间窗口）。
  - 输出对齐后的多模态训练样本。

---

### 第 2 月：模型开发与基线训练
- **第 1 周**
  - 搭建 3D ResNet18 编码器，用于逐日气象立体数据的空间特征提取。
  - 实现 TabM 模块，对 UKB 患者表型进行建模。
- **第 2 周**
  - 实现 Transformer 分支（时序注意力）。
  - 实现 AFNO 分支（频域稀疏建模），完成与 Transformer 的特征拼接与融合。
- **第 3 周**
  - 集成 MoE 模块（Transformer FFN、AFNO FFN、TabM projection、融合层可开关）。
  - 完成 Cross-Attention 融合机制，实现双模态交互。
- **第 4 周**
  - 进行基线训练（toy 数据 + 部分真实样本）。
  - 记录多标签分类指标（AUC、F1、Recall、Precision、PR-AUC、Hamming Loss）。
  - 调参与优化：batch size、学习率、专家数量（MoE）。

---

### 第 3 月：可解释性与总结
- **第 1 周**
  - 实现并测试 3D Grad-CAM，输出气象模态的空间关注区域。
  - 可视化 Self-Attention Heatmap（365 天的时序权重）。
- **第 2 周**
  - 实现 MoE 路由记录与聚类，统计不同专家的样本分布。
  - 探索 domain-specific 专家（如“冬季患者”、“老年患者”）是否自动分离。
- **第 3 周**
  - 整合可解释性结果：
    - Grad-CAM（空间热点图）
    - Attention Map（时间热点图）
    - MoE 聚类（样本-专家分布）
  - 撰写可解释性分析小结。
- **第 4 周**
  - 完成整体技术报告撰写（含方法、实验、困难与总结）。
  - 整理成果，准备论文初稿框架。

---


## 方案描述 

### 1. 数据准备

我们结合了两类异质模态的数据：

- **气象模态（ERA5-Land）**  
  我们从 ERA5-Land 下载了 **50 个气象与环境变量**（详见附录，例如 *2m 温度、土壤温度层、雪覆盖、降水量、辐射通量、植被指数* 等）。  
  - 原始空间分辨率：0.1°（约 10×10 网格覆盖每个城市）。  
  - 每个像素包含 **24 个垂直层**（如土壤层、大气层）。  
  - 数据被统一 resize 为 **20×20×24** 的体素，每位患者匹配发病前 **365 天**的数据。  
  - 我们使用 **多进程并行下载**加快 ERA5-Land 的数据拉取和处理。

- **患者模态（UK Biobank）**  
  我们从 UK Biobank 中提取了 **424 项患者特征**（包括人口学、生活习惯、临床变量）。  
  对缺失值，我们采用 **XGBoost 预测填补**方法，相比均值/众数填充在临床异质数据上表现更好。  
  每个患者特征通过 **发病日期**和**居住城市**与 ERA5 气象数据对齐，实现时空匹配。  

任务为 **多标签分类**：预测 **4 种眼科疾病**（如青光眼、白内障、AMD、糖尿病视网膜病变）的发病风险。由于患者可能合并多种眼病，因此使用多标签框架。

---

### 2. 模型结构

整体结构为一个 **双模态神经网络**，用于捕捉 **时空气象模式**和 **个体患者属性**，并在融合时保持可解释性和灵活性。

#### 2.1 气象分支：3D ResNet + Transformer + AFNO

- **3D ResNet18 主干**  
  输入为 **365 天 × 50 通道 × 20×20×24 体素**。  
  - 采用 **3D 卷积**在 (time × space × depth) 三个维度上同时建模，能够捕捉 **气候随时间的演变、地表到土壤的能量传输、雪深积累**等模式。  
  - 输出为 **每日一个 512 维 embedding**。

- **时序 Transformer（含 MoE）**  
  每日 embedding 输入到 **Transformer 编码器**：  
  - **自注意力**用于捕捉 365 天长时依赖。  
  - **前馈层 (FFN) 替换为 MoE**：每个时间片的 token 被路由到不同专家，使得模型可以专门化处理 **不同气候类型**（如海洋性气候 vs. 大陆性气候）。

- **AFNO 分支（Adaptive Fourier Neural Operator）**  
  并行使用 **AFNO** 捕捉气象的周期性：  
  - 将时间序列通过 **FFT** 转换到频率域。  
  - 使用 **分块对角的复数线性变换**学习主要频率模式（如季节性波动、短期振荡）。  
  - 通过 **soft-shrinkage 稀疏化**抑制非主要频率的噪声。  
  - **逆 FFT**还原时域信号。  
  同时，AFNO 分支也加入了 **MoE**，使不同频率特征由不同专家处理，适应气候区域差异。

- **气象时序特征融合**  
  Transformer 与 AFNO 的输出拼接后，经线性层映射回 512 维，得到综合的气象时序表示。

---

#### 2.2 患者分支：TabM (Tabular Mixture)

患者 424 维特征通过 **TabM**（Yandex Research 提出）处理。

- **核心原理**  
  TabM 提出了一种 **高效集成 (efficient ensemble)** 机制：  
  - 主权重矩阵 \(W\) 在所有子模型间共享。  
  - 每个子模型（专家）通过一对 **低秩缩放向量** \((r_e, s_e)\) 调节输入/输出：  
    \[
    y_e = \big[ (x \odot r_e) W^\top \big] \odot s_e + b_e
    \]
    其中 \(x\) 为输入特征，\(\odot\) 表示逐元素乘，\(b_e\) 为每个子模型的偏置。  
  - 这样可以在几乎不增加参数量的情况下，构建一个内部的“打包集成模型”。  

- **优点**  
  - 让模型在面对 **分布差异的亚群体**时更加鲁棒（例如不同生活方式的人群）。  
  - 内部集成提高了 **泛化能力**和 **不确定性估计**。  
  - 计算成本几乎与单模型相同。

我们将 TabM 输出投影到 512 维，与气象分支对齐。

---

#### 2.3 跨模态融合

采用 **双向交叉注意力 (bi-directional cross-attention)**：  

- **气象 → 患者**：患者 token 在气象序列上查询，关注与疾病最相关的时间片。  
- **患者 → 气象**：气象 summary token 在患者 embedding 上查询，将气候解释与个体特征结合。  

两个更新后的 token 拼接，经 MLP 得到融合表示。  
可选地在这一层加入 **MoE 头**，让模型专门化处理不同疾病亚型。

---

#### 2.4 分类器

最终融合表示经线性分类头，预测 **4 个眼科疾病的风险概率**。  
损失函数使用 **binary cross-entropy with logits**。

---

### 3. 训练与推理

- **损失函数**：带类别权重的 BCE。  

- **优化器**：Adam，带梯度裁剪。   

- **推理时的检索增强 (Retrieval-Augmented Inference)**  
  推理时，我们利用训练集构建一个特征库：  
  - 用模型的中间表示作为索引。  
  - 在测试样本预测时，检索出 k 个最相似的训练样本（相似度可选 **余弦**或 **欧式距离**）。  
  - 计算邻居的平均概率 \(p_{knn}\)。  
  - 与模型预测概率 \(p_{model}\) 融合：  
    \[
    p_{final} = (1-\alpha)\,p_{model} + \alpha\,p_{knn}
    \]

这样能缓解训练/测试分布差异。

---

### 4. 可解释性

我们设计了多层次的可解释机制：

- **3D Grad-CAM**：可视化气象体素中（纬度 × 经度 × 深度）最关键的区域。  
- **Transformer 注意力图**：显示模型在 365 天中关注的关键时段（如冬季骤降）。  
- **AFNO 频率分析**：指出模型利用的主导频率成分。  
- **MoE 路由可视化**：分析样本被分配到的专家，揭示潜在的病人亚群体或气候模式（例如“雪覆盖驱动的风险群体”）。  



# 项目总结

## 已完成工作
- 搭建多进程 ERA5-Land 数据下载框架，完成 50 个气象变量的收集。
- 实现数据格式转换与重采样（0.1° → 20×20×24），构建日尺度气象数据立方体。
- 收集并清洗 UK Biobank 患者特征（424 维），通过 XGBoost 完成缺失值填补。
- 设计并实现多模态模型框架（3D ResNet + Transformer + AFNO + TabM + MoE + Cross-Attn）。
- 完成 toy dataset 与真实数据子集的基线训练，验证模型结构可行性。

## 遇到的问题及解决方案
- **数据下载效率低** → 使用多进程并行下载，大幅缩短获取 ERA5 数据的时间。
- **缺失值比例高** → 采用 XGBoost 学习型填补方法，相比均值/中位数填补更符合变量间分布关系。
- **数据对齐复杂** → 按照患者发病时间窗口（365 天）和居住城市坐标匹配 ERA5 数据，构建个体化时空样本。
- **PaddlePaddle 模块限制** → 例如 `nn.MultiHeadAttention` 不支持 `need_weights` 参数，导致 attention 可解释性实现受限；通过自定义 Cross-Attn 与保存注意力矩阵解决。
- **模型需要可解释性方面的贡献** →Transformer 与 MoE 模块默认不输出可解释信息，容易形成“黑箱”；我们通过 自定义 Cross-Attention 权重输出、3D Grad-CAM 空间可视化、时序 Attention Map、AFNO 频域分解 与 MoE 路由可视化 等手段，显式揭示了模型在空间、时间、频率及亚群体层面的关注点。这些改进不仅解决了可解释性瓶颈，也使模型能够为临床专家与政策制定提供透明、可追溯的证据。
- **计算负担大** → 模型包含 3D CNN + 双 Transformer 分支 + MoE，需依赖 多块A100 GPU 训练；通过梯度裁剪以及模型量化方法结合分布式并行优化显存使用。
## 未来工作计划


在已有的 **多模态深度学习框架** 基础上（融合 ERA5 气象数据与 UK Biobank 等多源患者数据），本研究聚焦于 **显式级联建模**（环境 → 系统 → 暴露 → 生物 → 疾病），构建跨模态、可解释的疾病预测与干预模拟平台。总体目标是：

1. 在 *Nature Communications* / *Nature Medicine/AAAI/IJCAI* 发表 1–2 篇论文；  
2. 提出适用于多模态级联预测的通用框架；  
3. 验证模型在 UKB、CKB、FinnGen、BBJ 等多个 Biobank 上的可扩展性；  
4. 提供气象与环境干预下的疾病风险模拟；  
5. 为城市规划、空气质量控制、疾病防控政策提供量化证据。  


# Project Information

## Project Title
Ophthalmic Disease Onset Prediction Based on Multimodal Deep Learning: Integrating ERA5 Meteorological Data and UK Biobank Patient Features

---

## Timeline (Detailed for 3 Months)

### Month 1: Data Preparation and Preprocessing
- **Week 1**
  - Familiarize with ERA5-Land data interfaces and variable selection (confirm 50 meteorological feature variables).
  - Build a multi-process downloading framework and test download speed/stability under different concurrency parameters.
- **Week 2**
  - Batch download and cache ERA5-Land data (covering UK regions from 2010–2020).
  - Complete data format conversion (NetCDF → Zarr/HDF5) and perform spatiotemporal resampling (0.1° → 20×20×24).
- **Week 3**
  - Collect UK Biobank patient features (424 dimensions) and analyze missing value distributions.
  - Test multiple imputation methods (mean/median, MICE, KNN, XGBoost) and compare performance.
- **Week 4**
  - Finalize missing value imputation strategy (XGBoost).
  - Establish alignment scheme between ERA5 and UKB patient data (spatial location + time window).
  - Output aligned multimodal training samples.

---

### Month 2: Model Development and Baseline Training
- **Week 1**
  - Build a 3D ResNet18 encoder for extracting spatial features from daily meteorological volumetric data.
  - Implement TabM module for modeling UKB patient phenotypes.
- **Week 2**
  - Implement Transformer branch (temporal attention).
  - Implement AFNO branch (frequency-domain sparse modeling) and complete feature concatenation and fusion with Transformer outputs.
- **Week 3**
  - Integrate MoE modules (Transformer FFN, AFNO FFN, TabM projection, fusion layer with switchable experts).
  - Complete Cross-Attention fusion for bimodal interaction.
- **Week 4**
  - Conduct baseline training (toy dataset + partial real samples).
  - Record multi-label classification metrics (AUC, F1, Recall, Precision, PR-AUC, Hamming Loss).
  - Hyperparameter tuning: batch size, learning rate, number of experts (MoE).

---

### Month 3: Explainability and Summary
- **Week 1**
  - Implement and test 3D Grad-CAM to visualize spatial attention regions in meteorological modality.
  - Visualize Self-Attention Heatmap (temporal weights over 365 days).
- **Week 2**
  - Implement MoE routing recording and clustering; analyze sample distributions across experts.
  - Explore whether domain-specific experts (e.g., "winter patients," "elderly patients") are automatically separated.
- **Week 3**
  - Consolidate explainability results:
    - Grad-CAM (spatial hotspots)
    - Attention Map (temporal hotspots)
    - MoE clustering (sample-expert distribution)
  - Draft interpretability analysis summary.
- **Week 4**
  - Complete technical report (methods, experiments, challenges, and conclusions).
  - Organize results and prepare initial paper framework.

---

## Project Design

### 1. Data Preparation

We integrate two heterogeneous modalities:

- **Meteorological Modality (ERA5-Land)**  
  Downloaded **50 meteorological and environmental variables** (e.g., *2m temperature, soil temperature layers, snow cover, precipitation, radiation flux, vegetation index*).  
  - Original spatial resolution: 0.1° (~10×10 grid per city).  
  - Each pixel includes **24 vertical layers** (e.g., soil, atmosphere).  
  - Resized to **20×20×24** voxels, with each patient matched to **365 days** of data before disease onset.  
  - Used **multi-process parallel downloading** to accelerate ERA5-Land retrieval and preprocessing.  

- **Patient Modality (UK Biobank)**  
  Extracted **424 patient features** (demographics, lifestyle, clinical variables).  
  For missing values, applied **XGBoost-based predictive imputation**, which outperforms mean/median filling for heterogeneous clinical data.  
  Patient features are aligned with ERA5 meteorological data via **onset date** and **residential location**, enabling spatiotemporal matching.  

**Task**: Multi-label classification predicting **4 ophthalmic diseases** (e.g., glaucoma, cataract, AMD, diabetic retinopathy). As patients may develop multiple diseases, a multi-label framework is required.

---

### 2. Model Architecture

The overall design is a **bimodal neural network**, capturing both **spatiotemporal meteorological patterns** and **individual patient attributes**, while ensuring interpretability and flexibility.

#### 2.1 Meteorological Branch: 3D ResNet + Transformer + AFNO

- **3D ResNet18 Backbone**  
  Input: **365 days × 50 channels × 20×20×24 voxels**.  
  - **3D convolutions** jointly model (time × space × depth), capturing **seasonal changes, soil-atmosphere energy transfer, and snow accumulation**.  
  - Outputs **one 512-d embedding per day**.

- **Temporal Transformer (with MoE)**  
  Daily embeddings are fed into a **Transformer encoder**:  
  - **Self-attention** captures long-term dependencies across 365 days.  
  - **FFN replaced by MoE**: each token is routed to different experts, allowing specialized handling of **climate types** (e.g., maritime vs. continental).  

- **AFNO (Adaptive Fourier Neural Operator) Branch**  
  Models periodicity in parallel:  
  - Convert sequence via **FFT** to frequency domain.  
  - Apply **block-diagonal complex linear transforms** to learn dominant frequencies (e.g., seasonal cycles, short-term oscillations).  
  - Use **soft-shrinkage sparsity** to suppress noise.  
  - Perform **inverse FFT** to reconstruct.  
  - Added MoE to allow frequency-specific specialization across regions.  

- **Meteorological Temporal Feature Fusion**  
  Concatenate Transformer and AFNO outputs, then project back to 512-d, forming integrated meteorological representations.  

---

#### 2.2 Patient Branch: TabM (Tabular Mixture)

Patient 424-d features are modeled with **TabM** (proposed by Yandex Research).

- **Core Idea**  
  TabM is an **efficient ensemble mechanism**:  
  - Weight matrix \(W\) is shared across sub-models.  
  - Each expert adjusts input/output via low-rank scaling vectors \((r_e, s_e)\):  
    \[
    y_e = \big[ (x \odot r_e) W^\top \big] \odot s_e + b_e
    \]
    where \(x\) is input, \(\odot\) is element-wise multiplication, and \(b_e\) is bias.  
  - Builds an ensemble internally with minimal additional parameters.  

- **Advantages**  
  - Robust to **population subgroup heterogeneity** (e.g., lifestyle differences).  
  - Improves **generalization** and **uncertainty estimation**.  
  - Computationally comparable to a single model.  

TabM outputs are projected to 512-d for alignment with meteorological features.  

---

#### 2.3 Cross-Modal Fusion

Implemented **bi-directional cross-attention**:

- **Meteorology → Patient**: patient tokens query meteorological sequences to attend to disease-relevant time slices.  
- **Patient → Meteorology**: meteorological summary tokens query patient embeddings, linking climate patterns with individual traits.  

Fused tokens are concatenated and passed through MLP.  
Optionally, a **MoE head** is added to specialize for disease subtypes.  

---

#### 2.4 Classifier

The fused representation is passed to a linear classifier for **multi-label risk prediction of 4 ophthalmic diseases**.  
Loss function: **binary cross-entropy with logits**.  

---

### 3. Training and Inference

- **Loss Function**: BCE with class weights.  
- **Optimizer**: Adam with gradient clipping.  
- **Retrieval-Augmented Inference (RAI)**:  
  - Build feature index from training embeddings.  
  - At inference, retrieve *k* nearest neighbors.  
  - Compute average probability \(p_{knn}\).  
  - Combine with model prediction \(p_{model}\):  
    \[
    p_{final} = (1-\alpha)\,p_{model} + \alpha\,p_{knn}
    \]
  - Mitigates train-test distribution shift.  

---

### 4. Explainability

Multi-level interpretability mechanisms:

- **3D Grad-CAM**: highlights key spatial voxels (lat × lon × depth).  
- **Transformer Attention Maps**: identify critical time windows (e.g., winter drops).  
- **AFNO Frequency Analysis**: reveal dominant periodic components.  
- **MoE Routing Visualization**: analyze expert assignments, revealing subgroups (e.g., "snow-driven high-risk patients").  

---

# Project Summary

## Completed Work
- Built multi-process ERA5-Land data downloading framework; collected 50 meteorological variables.  
- Converted and resampled data (0.1° → 20×20×24) to daily meteorological cubes.  
- Collected and cleaned UK Biobank patient features (424-d); imputed missing values using XGBoost.  
- Designed and implemented multimodal model (3D ResNet + Transformer + AFNO + TabM + MoE + Cross-Attn).  
- Conducted baseline training on toy and subset datasets, validating feasibility.  

## Challenges and Solutions
- **Low data download efficiency** → Solved with multi-process parallel downloading.  
- **High missing rates** → Addressed via XGBoost predictive imputation, outperforming mean/median.  
- **Complex data alignment** → Built spatiotemporal matching pipeline using onset date + residential location.  
- **Framework limitation (PaddlePaddle)** → `nn.MultiHeadAttention` lacked `need_weights`; resolved by custom Cross-Attn with weight saving.  
- **Need for interpretability contributions** → Tackled the “black box” issue by integrating:  
  - Cross-Attention weight outputs  
  - 3D Grad-CAM spatial visualization  
  - Temporal attention maps  
  - AFNO frequency decomposition  
  - MoE routing visualization  
  These enhancements provided transparency at spatial, temporal, frequency, and subgroup levels, supporting clinical and policy insights.  
- **Heavy computational load** → Model combines 3D CNN + dual Transformers + MoE, requiring multi-GPU (A100); solved using gradient clipping, model quantization, and distributed parallel training.  

---

## Future Work Plan

Building on the existing **multimodal deep learning framework** (ERA5 meteorology + UK Biobank features), the research will focus on **explicit cascade modeling** (Environment → System → Exposure → Biology → Disease) to construct an interpretable, multimodal disease prediction and intervention simulation platform.  

**Goals:**
1. Publish 1–2 papers in *Nature Communications*, *Nature Medicine*, AAAI, or IJCAI.  
2. Propose a generalizable multimodal cascade prediction framework.  
3. Validate scalability across multiple Biobanks (UKB, CKB, FinnGen, BBJ).  
4. Provide disease risk simulations under environmental and climate interventions.  
5. Deliver quantitative evidence for urban planning, air quality control, and public health policy.  
