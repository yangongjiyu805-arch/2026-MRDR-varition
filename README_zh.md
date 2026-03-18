# scMIGRA-v1 说明

`scMIGRA` 是在当前 `scMRDR` 系列之外单独开出来的一版新模型。  
这版的重点不是继续堆更多损失项，而是先验证一条新的主线是否成立：

1. 先学习更稳定的共享生物学表示。
2. 再把模态特异和技术噪声拆到独立分支。
3. 最后只在高置信局部区域内做跨模态对齐。

## 核心思路

1. `program-simplex shared biology`
- 共享表示不是普通黑盒 latent，而是由一组 metaprogram 混合生成。

2. `modality residual`
- 单独保留模态特异信息，避免 shared latent 被迫同时承担共性和个性。

3. `technical residual`
- 专门吸收 batch / technical variation，减少 shared latent 里残留技术噪声。

4. `reliability-aware anchors`
- 只让高置信跨模态匹配进入对齐，减少错误锚点把结构拉坏。

5. `local transport + cycle`
- 只做局部 soft transport，并加 cycle consistency，避免过强的全局硬对齐。

## 当前版本包含

- `program-based shared biology`
- `modality residual`
- `technical residual`
- `reliability-aware anchors`
- `local transport lite`
- `cycle consistency`
- `batch independence (HSIC)`

当前版本暂时没有做：

- full fused Gromov-Wasserstein
- feature-link graph decoder
- 更复杂的 foundation / pretraining 设计
- 自动大规模调参脚本

## 训练流程

训练入口现在是 [main_migra.py](/d:/移民/科研进度/2026.1/MRDR/scMRDR-main/src/scMIGRA/main_migra.py)。

这版训练流程有三个关键点：

1. 优先读取 `counts` 类 layer；如果没有，就自动退回 `adata.X`。
2. 训练阶段默认不把 `cell_type` 或其他细胞类型标签送进模型，保持无标签训练。
3. 训练结束后会把以下结果写回输出 h5ad：
- `latent_shared`
- `latent_specific`
- `latent_tech`
- `program_assignments`
- `uns["migra_program_basis"]`
- `uns["migra_run_config"]`

## 快速评估流程

评估入口现在是 [metrics.py](/d:/移民/科研进度/2026.1/MRDR/scMRDR-main/src/scMIGRA/metrics.py)。

这个版本专门为高频实验做了调整：

1. 不再写死数据集名、方法名、标签字段和模态映射。
2. 支持一次评估多个 embedding。
3. 默认会优先评估：
- `latent_shared`
- `concat(latent_shared + latent_specific)`
- `program_assignments`

4. 会同时输出：
- `summary_metrics_<tag>.csv`
- `full_metrics_raw_<tag>.csv`
- `full_metrics_scaled_<tag>.csv`
- `metrics_history.csv`

其中：

- `summary` 适合你快速看总分和关键指标。
- `history` 适合你连续做实验后直接横向比不同 run。

## 推荐使用方式

最方便的入口是 [提交脚本_migra](/d:/移民/科研进度/2026.1/MRDR/scMRDR-main/src/scMIGRA/提交脚本_migra)。

它现在已经修成：

1. 先跑 [main_migra.py](/d:/移民/科研进度/2026.1/MRDR/scMRDR-main/src/scMIGRA/main_migra.py)。
2. 默认训练结束后自动跑 [metrics.py](/d:/移民/科研进度/2026.1/MRDR/scMRDR-main/src/scMIGRA/metrics.py)。
3. 如果你只想训练不想评估，把 `RUN_METRICS_AFTER_TRAIN=false` 即可。

## 第一版最该看的结果

建议第一轮先重点观察三项：

1. `Bio conservation`
2. `Batch correction`
3. `Modality integration`

如果这版相较 `scMRDR` 和你之前改过的版本，能把后两项明显拉起来，同时不把 batch 再做坏，就是一个值得继续往论文方向推进的信号。
