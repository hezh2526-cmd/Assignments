# 数据来源

## USPS

- OpenML 数据集 ID：41082（版本 2）
- 页面：https://www.openml.org/d/41082
- 代码使用：`sklearn.datasets.fetch_openml(data_id=41082, as_frame=False)`
- 数据说明：9,298 个样本，256 个像素特征，10 个类别；每个样本可还原为 16×16 灰度图。

## UCI Sonar

- UCI 数据集 ID：151
- 页面：https://archive.ics.uci.edu/dataset/151/connectionist+bench
- DOI：https://doi.org/10.24432/C5T01Q
- 代码使用：`ucimlrepo.fetch_ucirepo(id=151)`
- 数据说明：208 个样本，60 个连续特征；类别为 Rock（岩石）与 Mine（金属圆柱）。

代码会校验数据维度和标签集合。如下载内容与上述版本不一致，程序会立即报错，不会继续生成不可比结果。

