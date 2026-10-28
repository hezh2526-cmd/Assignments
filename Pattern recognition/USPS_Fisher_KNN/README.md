# 模式识别大作业 1：Fisher 线性判别与 K 近邻

本项目对应课程 PPT 中的第一次大作业，覆盖两个数据集：

- USPS 手写数字：9,298 个样本，10 类，图像为 16×16 灰度图（256 维）。代码从 OpenML 数据集 ID 41082 下载。
- UCI Sonar：208 个样本，2 类，60 个连续特征。代码从 UCI 数据集 ID 151 下载。

实现并比较三种方法：

1. Fisher LDA 分类器；
2. 标准化原始特征上的 KNN；
3. Fisher LDA 降维后的 KNN（Fisher+KNN）。

## 目录

```text
大作业1_Fisher_KNN/
├─ code/
│  ├─ run_experiments.py       # 主入口
│  ├─ config.py                # 实验参数
│  ├─ datasets.py              # 数据下载与校验
│  ├─ evaluation.py            # 指标和实验逻辑
│  ├─ plotting.py              # 全部图表
│  ├─ requirements.txt
│  └─ run_all.ps1              # Windows 一键运行脚本
├─ data/                       # 下载缓存，首次运行自动生成
└─ results/                    # 运行后自动生成图、表和日志
```

## 默认实验设置

- 随机种子：42；
- 训练样本比例：10% 至 90%，步长 10%；
- 每个比例重复 5 次分层随机划分；
- K：1、3、5、7、9、11、15；
- 最终展示划分：70% 训练、30% 测试；
- 最终 K 值仅使用训练集内部验证集选择，不使用测试集调参；
- 特征标准化只在训练数据上拟合，避免数据泄漏。

如需修改参数，编辑 `code/config.py`。

## 运行方法

在 PowerShell 中进入项目目录后执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\code\requirements.txt
.\code\run_all.ps1
```

也可以直接运行：

```powershell
python .\code\run_experiments.py --project-root .
```

首次运行需要联网下载数据，后续会使用 `data/cache` 中的缓存。

## 评价指标

- OA（Overall Accuracy）：所有测试样本中分类正确的比例；
- AA（Average Accuracy）：各类别召回率的宏平均，等价于 balanced accuracy；
- Kappa：Cohen's kappa，衡量扣除随机一致性后的分类一致程度；
- 同时输出 Error Rate、Macro Precision、Macro Recall、Macro F1、每类 Precision/Recall/F1 和混淆矩阵。

## 结果文件

运行后 `results` 至少包含：

- `tables/dataset_summary.csv`：数据集规模与类别信息；
- `tables/sweep_raw.csv`：每次随机划分的全部原始指标；
- `tables/sweep_summary.csv`：均值与标准差；
- `tables/reference_metrics.csv`：最终展示划分的指标；
- `tables/reference_predictions_*.csv`：逐样本真实标签与预测标签；
- `tables/classification_report_*.csv`：逐类别指标；
- `tables/model_selection.csv`：训练集内部 K 值选择记录；
- `figures/dataset/`：数据集样例、类别分布和特征图；
- `figures/sweeps/`：K 值与训练比例实验曲线、热力图；
- `figures/reference/`：最终指标、混淆矩阵、Fisher 投影与定性结果；
- `run_metadata.json`：完整配置、软件版本和运行时间。

本项目只提供可运行代码，不预填任何实验数值。所有结果均由程序实际运行后生成。

