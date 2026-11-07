# 第三方来源

本项目运行时使用 同一 glocal2-nav 仓库中的场景模块，具体源码 SHA-256 随每次生成写入 `scene.json.generator_provenance`。

其中 `src/stair_nav/vendor/eth_stairs.py` 提取自 ETH Zurich / Takahiro Miki 的 [terrain-generator](https://github.com/leggedrobotics/terrain-generator)，上游 commit `7be9d9a6ebbed7dc669486426184a4694bfd0bb6`，MIT License。保留的函数用于本场景实心参考楼梯，许可证位于后端仓库的 `src/stair_nav/vendor/LICENSE.terrain-generator`。

本项目没有复制第三方函数。若未来分发时将后端打包，需要一并保留后端的许可证、归属与来源清单。当前运行需要可读取的指定 glocal2-nav checkout。
