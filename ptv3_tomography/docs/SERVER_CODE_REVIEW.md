# 当前 PTv3 训练代码审查

2026-11-06：按用户要求统一为原生 VST 标签监督，并清除旧掩码、重复检查和串行训练路径。当前审查与实际运行结果见 [TEACHER_SUPERVISION_CLEANUP.md](TEACHER_SUPERVISION_CLEANUP.md)。

| 发现 | 影响 | 当前处理 |
| --- | --- | --- |
| cost 复用地面连续、回波匹配和覆盖条件 | 大量危险边界标签无法进入 Loss | cost、height 均使用 target_available |
| load_sample 在点数裁剪后重建旧掩码 | 即使保存新 mask，训练仍会重新过滤 | 加载时直接使用标签可用性 |
| 缓存读取已保存的旧 mask | 旧运行可能继续使用稀疏监督 | 从缓存原始 cost、高度得到可用性，不读取旧 mask |
| Isaac 目录另存配对实现 | 从不同入口启动时规则不一致 | 删除复制的模块，正式配对统一使用维护入口 |
| 旧 v1 microbatch=1 累积训练 | 与真实 batch 入口并存 | 删除 v1 训练器和两份旧训练配方 |
| 哈希、配置、形状和重复坐标预检查 | 拒绝代码修改或增加数据准备工作 | 移除任务层的防御性检查 |

保留官方 PTv3、地图坐标转换、真实 batch、AMP、AdamW、OneCycle、有效标签 Loss、last/best 保存与恢复。保留 CPU 绘图及逐点特征恢复功能。
