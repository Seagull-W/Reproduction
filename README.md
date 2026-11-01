本人阅读论文或项目之后的一些复现实验，显存资源有限，32~96GB，不保证计算能力。

主要目的是熟悉常见的方法，同时在复现的过程中深入思考。

## 代码准则

**不必要的地方一定避免冗余设计，避免边缘保护。** 只实现当前实验所需的能力，不预建通用框架，不堆叠假想场景的校验、兜底和兼容分支。采用清晰、直接的代码结构，用注释解释算法原理、数据含义与实现选择，方便阅读和学习。具体要求见 [AGENTS.md](AGENTS.md) 中的代码要求，适用于整个仓库。

## 实验执行准则

所有项目遵循 [任务产出与存储准则](docs/EXPERIMENT_ARTIFACTS.md)，执行代理另见 [AGENTS.md](AGENTS.md)。核心约定是：原始证据优先采集并及时回传；检查点先保存在实例磁盘；评测完成后按后续用途决定是否长期归档权重。大文件传输在无卡/CPU 环境进行。

新实验可复制 [运行清单](docs/templates/run-manifest.example.json)、[产物清单](docs/templates/artifact-manifest.example.json) 和 [实验记录模板](docs/templates/experiment-record.md)。模板用于明确计划和验收，不代表采集、备份或停机机制已经实现。

## 项目

- [MAI-Gym](MAI-Gym/README.md)：使用已有后训练模型进行基准评测，重点保存逐步轨迹和评测证据。
- [GRPO 与验证器可靠性](grpo-verifier-reliability/README.md)：从 CPU 教学实验到真实训练对照，重点保存奖励、输出与独立评测之间的联系。
- [夸克 CLI 存储测试](storage-tests/quark/2026-11-01/report.json)：当前机器、普通账号下的传输实测；结果不代表其他账号、地域或会员等级。
