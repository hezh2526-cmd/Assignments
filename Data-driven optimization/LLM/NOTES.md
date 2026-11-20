# LLM 设计的代理模型辅助特征选择算法

本项目用于完成“由大语言模型设计一个算法，并使用该算法解决昂贵优化问题”的课程作业。

实际工作方式是：使用者在某个 LLM 的聊天界面中手动描述问题和代码接口，取得 LLM 输出的算法代码，再将代码保存为 `generated_algorithms/llm_generated_optimizer.py`。本项目程序不会调用任何 LLM API，也不会自动生成、更新或迭代算法。

主实验使用 scikit-learn 自带的 Breast Cancer 数据集（569个样本、30个特征），泛化实验使用 Digits 数据集（1797个样本、64个特征），均不需要额外下载数据。

## 目录结构

```text
LLM/
├── configs/
│   ├── quick.json                  # 快速流程测试
│   └── full.json                   # 正式重复实验
├── docs/
│   └── algorithm_and_experiment.md # 算法与实验说明、流程图
├── generated_algorithms/
│   └── llm_generated_optimizer.py  # 由LLM设计并输出的核心算法
├── results/                        # 实验数据、统计表和图片
├── src/llm_fs/
│   ├── problem.py                  # 昂贵特征选择问题
│   ├── validator.py                # LLM算法代码校验
│   ├── experiment.py               # 实验运行器和随机搜索基线
│   ├── analysis.py                 # 统计与绘图
│   └── cli.py                      # 命令行入口
├── tests/                          # 自动测试
├── requirements.txt
└── run.py
```

## 安装

```powershell
cd "C:\Users\He Zihan\Desktop\LLM"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

项目不需要 API Key，也不依赖 `openai` Python包。

## 核心算法

`generated_algorithms/llm_generated_optimizer.py` 是由 LLM 输出的算法。实验框架只要求它实现以下统一接口：

```python
class GeneratedOptimizer:
    def __init__(self, bounds, variable_types, budget, seed): ...
    def ask(self, n_points): ...
    def tell(self, X, y): ...
    def get_best(self): ...
```

- `ask`：提出需要真实评价的特征子集；
- `tell`：接收交叉验证得到的目标值；
- `get_best`：返回目前最好的特征子集和目标值。

当前算法由 LLM设计为多样性初始化、Extra Trees代理模型、自适应二进制进化搜索及均值—不确定性采集。项目中没有算法版本迭代机制。

## 优化问题

候选解是二进制特征掩码：

```text
x_i = 1：选择第 i 个特征
x_i = 0：不选择第 i 个特征
```

最小化目标：

```text
objective = 1 - mean_cross_validation_accuracy
            + feature_penalty × selected_feature_ratio
```

每个新特征子集都需要重新训练SVM并进行分层交叉验证，因此属于昂贵黑箱评价。

## 运行步骤

### 1. 校验LLM生成的算法

```powershell
python run.py validate
```

校验内容包括接口、维度、二进制合法性、重复候选以及禁止的文件或系统调用。

### 2. 快速测试

```powershell
python run.py experiment --config configs/quick.json
python run.py analyze --input results/raw_results.csv
```

快速配置使用1个随机种子、20次评价和3折交叉验证，只用于确认流程能够运行。

### 3. 正式实验

```powershell
python run.py experiment --config configs/full.json
python run.py analyze --input results/raw_results.csv
```

正式配置使用10个随机种子、60次评价和5折交叉验证。对比方法包括：

- `generated`：LLM设计的核心算法；
- `random_search`：随机搜索最低基线。

## 输出结果

运行后生成：

```text
results/raw_results.csv
results/summary.csv
results/convergence_breast_cancer.png
results/convergence_digits.png
results/boxplot_breast_cancer.png
results/boxplot_digits.png
```

`raw_results.csv`记录每次昂贵评价，`summary.csv`记录多次独立运行的均值、标准差、中位数、准确率、特征数、时间和重复率。

## 公平实验规则

- 所有优化算法使用相同的昂贵函数评价预算；
- 使用相同数据集、交叉验证划分和随机种子；
- 横轴使用昂贵目标函数评价次数，而不是算法代数；
- Breast Cancer是主实验，Digits用于检验算法能否直接适配不同维度；
- 随机搜索只是对比基线，不属于LLM生成的核心算法。

## 提交建议

提交内容包括代码目录、正式实验CSV、结果表、收敛曲线、箱线图和算法说明文档。报告只需说明算法由使用者手动向LLM描述问题后获得；程序运行期间不会与LLM通信。

