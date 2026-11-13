# 两阶段训练

使用本机 pure3d-ptv3 环境和 CUDA GPU。BF16、真实 batch 1、workers 4、AdamW、学习率 1e-4、weight decay 0.01，5% warmup 后 cosine。PTv3、Filter 冻结，输入使用原始密度。核心和上下文的固定网格在全部课程阶段保持一致。

A 阶段训练空间适配器和几何解码器，以完整 Mesh 的五级表面分类和坐标偏移为监督。B 阶段冻结 A，训练条件适配器与速度 U-Net，使用标准 Flow Matching MSE 加预测终点的几何 Loss。公式与查询分组见 SUPERVISION_LOSS.md，门槛不改变。

每阶段先实际训练 5 轮，保存并恢复 last.pt 继续至 50 轮。scheduler 从开始就采用 50 轮预算，恢复时保留优化器、scheduler、RNG 和 global step。固定验证每 5 轮执行，A 按 Precision／Recall@3 cm 的较小值选择 best.pt。B 按缺失表面 F1@2 cm 选择 best.pt。测试仅用于最终评价。

只有 A 的固定验证 Precision@3 cm 和完整表面 Recall@3 cm 都达到 90%，才进入 B。未达标会在 runs/recon_terrain_curriculum_v1/status.json 记录 ae_validation_gate_not_met，不自动降低门槛。

B 第 10、20、35 轮结束保存待晋级阶段后暂停 DataLoader。独立子进程生成下一份全局缺损及条件缓存，完成后原子切换当前清单，删除旧训练数据。全局目标分别是 3.5%、7.5%、15%、25%。每轮包含全部 32 个缺损块和 8 个完整对照，训练样本数不随阶段变化。

两阶段各保留 best.pt、last.pt、必要日志、曲线、固定验证与最终测试结果。不保留按阶段编号归档的历史缺损库。当前实验的运行状态以 status.json 和日志为准，不将数据缓存阶段记为训练更新。
