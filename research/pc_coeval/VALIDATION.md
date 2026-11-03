# 已有 coeval 结果的光子守恒与 MFP 验证

本程序只读取已经完成的 PC、原始 SCRIPT、native ES 三维场；不会重新生成
21cmFAST 密度/源场，也不会重跑三个模拟算法。程序不会写入原来的结果目录。

## 检验的含义

**光子收支**直接从保存的 `source`、`capacity` 和 `xHI` 数组重新求和。
电离原子数必须使用 `sum(capacity * (1-xHI))`，不能把体积平均电离比例当作
原子数。对于此实验的无复合、周期性、瞬时分配假设，完整分配应得到
`sum(ionized_atoms) = min(sum(source), sum(capacity))`。同时检查输入有限性、
非负性和每格的中性比例范围。PC 与 SCRIPT 分别采用 `1e-8` 和 `2e-5` 的
相对分配容差，并明确计入保存中性场的浮点舍入下限；原始误差全部保留。
native ES 的预算偏差也输出，但 ES 不被当成光子守恒算法要求通过。

旧 coeval 文件没有单独保存 PC 剩余光子数组，因此完全电离后的超额光子
只能根据输入预算推算。验证会明确区分“电离原子预算通过”与“独立剩余光子
账本通过”，不会把推算出的余量当成独立证据。对新文件，若存在
`pc_remaining_source`，还会独立检查这份剩余光子数组。

**MFP 形态统计**在选定相的体积中均匀抽取起点，选择各向同性方向，测量首次
离开该相的距离。射线按精确网格边界遍历，采用周期边界。这里的 MFP 是几何
长度分布，不是包含吸收截面、亚网格吸收体的物理吸收平均自由程，也不直接等于
气泡半径。

默认对三个模型、每个红移分别测量：

- 电离相与中性相，主阈值 `xHI=0.5`，并用 `0.1`、`0.9` 检查阈值敏感性。
- 每个相/阈值/模型 100000 条射线，三个模型使用相同的随机抽样数。
  各模型选定的体积不同，因此这不意味着起点在相同物理位置。
- 平均退出距离、分布、抽样误差以及射线数检查；具体诊断见 JSON。
- 距离上限默认为一个盒长。未退出的射线作为右删失数据报告，不能被当成恰好
  在上限退出的射线；包含这些射线的截断平均值只是未截断平均值的下限。

比较是在相同红移下进行，模型的体积填充比例可能不同。MFP 接近只能说明这项
形态统计接近，不能代替空间分辨率收敛，也不能证明与 SCRIPT 相同的场是唯一
正确解。`COMPLETED` 表示后处理完成，不表示形态或网格收敛已经认证。

定义参考：Mesinger & Furlanetto (2007),
<https://arxiv.org/abs/0704.0946>；Lin et al. (2016),
<https://doi.org/10.1093/mnras/stw1542>。

## 本地验证记录

本次发布运行 `python -m pytest -q research/pc_coeval`：**164 项通过**。
包括已有分配算法回归、独立光子预算、解析球体和平板、独立网格交点参考、周期
边界、多线程确定性、截尾归一化、三模型后处理、结果复用和 PBS 提交脚本测试。

另外用已有的 native 21cmFAST 输入（seed 222、50 cMpc、8³、10 个红移）
执行 PC → 保存/读取数组 → 独立预算/MFP 流程，十个快照全部通过。重算光子
相对误差最大约 `2.75e-17`。此项未运行 SCRIPT，不能据此声称三模型接近，
也不代表 Gadi 上 256 cMpc 的实际形态已经验证；那部分由下面的任务测量。

## Gadi 提交

先更新仓库，再对最近完成的 `coeval_pc_*` 结果提交后处理。`--latest` 会打印
选中的绝对路径，然后把这个路径及输入校验和固定在新任务中。若要指定某次结果，
用 `--input-run /absolute/coeval_pc_RUN` 替换 `--latest`。

```bash
(
    set -euo pipefail
    cd /g/data/qp00/sz4640/21cmFAST
    git switch matched-photon-conservation
    git pull --ff-only origin matched-photon-conservation
    unset PC_INPUT_RUN

    PC_RUN_ROOT=/g/data/qp00/sz4640/21cmfast_runs \
    PC_THREADS=8 PC_MEM=16GB PC_WALLTIME=04:00:00 PC_QUEUE=normal \
    PC_MFP_RAYS=100000 PC_MFP_SEED=20261103 \
    PC_MFP_THRESHOLDS=0.1,0.5,0.9 PC_MFP_MAX_DISTANCE_BOXES=1 \
    bash research/pc_coeval/gadi_validate_prepare.sh --latest --submit
)
```

也可以传入原任务的 `results` 子目录。默认复用该任务的 `python_path.txt`
和 `runtime.env`，但验证不需要加载 21cmFAST 原生扩展；仅需要 NumPy、Matplotlib
及带 OpenMP 的 C++ 编译器。缺少原运行环境文件时，可显式设置绝对路径的
`PC_PYTHON`，必要时再设置 `PC_RUNTIME_ENV`。

这段命令须在有 Gadi 登录权限的终端执行；本地生成代码、测试或推送 GitHub
本身不代表 PBS 作业已经提交。成功提交后会打印实际作业号。

## 文件与重试

新目录为：

```text
/g/data/qp00/sz4640/21cmfast_runs/coeval_validation_<commit>_<time>_<suffix>/
```

其中的 `results/` 包含：

- `photon_mfp_report.json`：所有红移的光子预算、MFP 数值与比较。
- `photon_budget.png`：独立预算误差与源光子/原子比。
- `mfp_history.png`：两相的平均退出长度及删失比例随红移的变化。
- `mfp_distributions.png`：代表红移的 MFP 分布。
- `seed*/z*/metrics.json`、`ray_samples.npz`、`receipt.json`：逐快照指标、
  原始射线结果与校验和，可复查并用于继续分析。

失败后使用新目录中的 `bash /absolute/coeval_validation_RUN/resubmit.sh` 重试。
已完成快照先校验后复用。程序、参数或输入改变时应新建任务，不能修改冻结任务
中的文件再覆盖校验和。PBS 输出、临时编译和运行日志均在冻结 `source/` 外。

若默认射线数的抽样误差仍影响结论，可用相同输入再建一个
`PC_MFP_RAYS=400000` 的任务；这仍只重复后处理，不重跑模拟。
