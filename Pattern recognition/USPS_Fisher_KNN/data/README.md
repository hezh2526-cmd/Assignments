# 数据目录

首次运行时，程序会在此目录下创建缓存并下载数据。USPS 保存在 `cache/openml`；UCI Sonar 首次通过 `ucimlrepo` 获取，并保存为 `raw/uci_sonar_id151.csv`，以后优先读取本地副本。

不要手工改动缓存中的样本或标签，否则数据校验会报错。

