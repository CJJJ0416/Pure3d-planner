# PTv3 Conda 环境

本机环境 `pure3d-ptv3` 使用 Python 3.11、PyTorch 2.10.0+cu128、Triton 3.6.0 和本工作区的 `utils/PointTransformerV3` 可编辑源码。原 Isaac 和导航环境未修改。

```bash
conda activate pure3d-ptv3
cd /home/cjh/Pure3D_ws
python -c 'import torch, point_transformer_v3 as ptv3; print(torch.__version__, torch.cuda.is_available(), ptv3.__file__)'
```

如需要在另一台机器新建同名环境，在该机器的项目根目录执行：

```bash
conda env create -f environment-ptv3.yml
conda activate pure3d-ptv3
```

配置安装 CUDA 12.8 版 PyTorch，目标机需要兼容的 NVIDIA 驱动。依赖下载和首次下载模型需要网络。当前测试依赖本机的 SOCKS 代理，`socksio` 已安装。已有同名环境时不要直接重建覆盖。

2026-11-06 验证：本机 RTX 4060 Laptop、驱动 580.196.07。`pip check` 通过，PyTorch CUDA 矩阵运算通过。随机初始化 PTv3 在 CPU 和 GPU 上以 128 点输出有限的 `[128,64]` 特征，GPU 默认混合精度也通过。Utonia-tiny 官方预训练权重下载和哈希检查通过，GPU 对 128 个随机点及零颜色、零法线输出有限的 `[128,486]` 特征。这是安装和小输入前向验证，不是实际 PCD 效果、实时性能或 A100 测试。

预训练权重默认使用 Hugging Face 缓存。`utonia_tiny` 已缓存在本机。`sonata` 等权重尚未验证是否下载到本机。两个缓存与 Conda 环境独立。
