# NSGA-II 从零实现

本项目依据 Deb 等人（2002）的论文实现 NSGA-II，核心算法未调用 `pymoo`、`DEAP` 等现成多目标优化库。

已实现：快速非支配排序、拥挤距离、二元锦标赛选择、SBX 模拟二进制交叉、多项式变异、父子代合并及精英环境选择。测试问题包括 ZDT1、ZDT2 和 ZDT3。

## 环境

- Python 3.13
- NumPy
- Matplotlib
- Pandas

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

## 快速运行

在项目根目录执行：

```powershell
python main.py --problems zdt1 --runs 1 --population 100 --generations 200
```

完整实验（3 个问题，每个问题独立运行 20 次）：

```powershell
python main.py --problems zdt1 zdt2 zdt3 --runs 20 --population 100 --generations 200
```

结果保存在 `results/`：

- `all_runs.csv`：每次独立运行的指标
- `summary.csv`：IGD、HV、运行时间的均值和标准差
- `figures/`：理论前沿与算法结果对比图
- `solutions/`：最佳一次运行的决策变量和目标值

## 测试

```powershell
python -m unittest discover -s tests -v
```

## 主要参数

- 种群规模：100
- 迭代次数：200
- 交叉概率：0.9
- SBX 分布指数：20
- 变异概率：`1 / 决策变量数`
- 多项式变异分布指数：20

## 参考文献

Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.
