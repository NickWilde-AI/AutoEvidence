# 完成度层级

避免把不同完成度混为一谈。

1. **工程闭环（已验证）**  
   合成数据上，CPU 可跑通：协议 → 定位 → 合成 → tiny SFT/GRPO → 评测 → API。

2. **正式训练适配（已预留，未验证）**  
   `configs/sft_full.yaml`、`grpo_full.yaml`、`ds_zero3.json`、models.yaml 中的 Qwen3.6-35B-A3B。  
   准确表述：**已预留正式模型训练方案，目标环境适配与效果待验证。**

3. **业务效果（未验证）**  
   需真实工单、独立标注/独立裁判、冻结后的 test + safety_seal。  
   参考文档指标仅为目标，不是本仓库实测。
