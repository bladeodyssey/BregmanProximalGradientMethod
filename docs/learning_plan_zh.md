# 学习计划：Slice 卷积稀疏编码、PGD 与 Bregman PGD

## 第 1 阶段：理论和符号统一

- 阅读 Elad 相关卷积稀疏编码和 slice/local sparse model 文献，明确全局信号、局部 patch、slice、stripe sparse vector 之间的关系。
- 固定本文实验采用的模型：
  `Y_n = sum_k D_k * X_{n,k} + epsilon_n`。
- 推导目标函数：
  `min_{D,X} 0.5||DX-Y||_2^2 + lambda||X||_1`，
  subject to `||D_k||_2 <= 1`。
- 手动推导系数梯度、字典梯度、软阈值近端算子，以及 Bregman kernel
  `h(x)=0.5||x||_2^2 + rho/4||x||_4^4` 对应的更新。

## 第 2 阶段：PGD 基线复现

- 运行 `python -m experiments.run_simulation --config configs/simulation.json`。
- 先使用无噪声、小稀疏度、小字典规模，确认 objective 下降、重建误差下降。
- 查看 `results/simulation/dictionary_compare.png`，确认学习字典和真实字典存在可见相似性。
- 使用 `metrics.json` 中的 NMSE、PSNR、dictionary correlation、support precision/recall 作为基线指标。

## 第 3 阶段：Bregman PGD 改进

- 保持数据、初始化、外层迭代、稀疏编码迭代次数完全一致。
- 只替换稀疏编码步：从欧氏 PGD 换成 Bregman PGD。
- 扫描 `bregman_rho`，建议从 `0.05, 0.1, 0.25, 0.5, 1.0` 开始。
- 重点比较收敛曲线、最终 NMSE、字典相关性、运行时间。

## 第 4 阶段：报告和扩展

- 把 `objective_curve.png`、`dictionary_compare.png`、`reconstruction_compare.png` 放入报告。
- 定量表格至少包含 PGD 和 Bregman PGD 两行。
- 扩展方向：
  - FISTA 加速；
  - FFT 卷积；
  - 多组随机种子统计均值和标准差；
  - 更贴近 Elad slice 理论的局部稀疏约束和 stripe sparsity 指标。

## 推荐开发实践

- Windows 只作为 VS Code 前端。
- 通过 Remote SSH 连接 Linux GPU 主机。
- 在 Linux 端维护 conda 环境、Git 仓库和实验结果。
- 每次实验保存配置、随机种子、指标和图像，避免只保存口头结论。
