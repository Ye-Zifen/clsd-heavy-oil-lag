# models/

运行 `python predict.py --data_dir <数据目录> --output <结果路径>` 时，训练好的模型保存为本目录下的 `model.pkl`；加 `--use_saved_model` 时从这里加载。

仓库不附带 `model.pkl`（约 280 MB，超过 GitHub 单个文件的大小限制）。用缺省命令从训练集重新训练即可生成，与决赛提交的模型逐字节相同（md5 `766e7f0245a51d18ec90c42f411e9689`），2 核 CPU 约 12 分钟。

轻量档（`--config config_lite.yaml`）的模型保存为 `model_lite.pkl`，约 26 MB。
