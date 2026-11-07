"""将 PTv3 体素特征恢复到原始输入点的顺序。"""


def restore_point_features(voxel_features, inverse):
    """[M, C] -> [N, C]，同一体素内的原始点共享该体素的特征。

    inverse 是对应点云调用 prepare(...) 时返回的 [N] 索引。
    返回特征的行顺序与原始输入点一致，绘图时搭配原始 XYZ 使用。
    """
    return voxel_features[inverse]
