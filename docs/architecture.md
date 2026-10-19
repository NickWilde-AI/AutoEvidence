# 架构

## 数据流

```
Ticket(对话/信号/照片/历史)
  → FaultClueLocator（症状锚定 + 异常补窗 + 图相似度）
  → EvidencePack
  → Teacher（规则 / OpenAI 兼容）+ 三道过滤
  → SynthesizedSample → 条款改写展开 → SFT entries
  → 难例/换配/过采样 → RL set
```

## 训练流

```
SFT（plain | segment_weighted | zeroshot 配置）
  → checkpoint
  → GRPO（rollout G 条 → 引证奖励+裁判 → 组内优势 → clip + KL(ref) → 反传）
  → export
```

## 推理流

```
Ticket（剥离 gold）→ Locator → Analyzer → ModelOutput + Routing
```

安全结果转人工确认；证据不足进人人审。

## 模块边界

| 包 | 职责 |
|----|------|
| schemas | 协议与硬约束校验 |
| data | 合成、隔离、增广、统计 |
| evidence | 定位与视觉编码 |
| standards | 条款版本与引用校验 |
| synthesis | 教师与过滤 |
| training | SFT / GRPO |
| rewards | 分项奖励与裁判 |
| evaluation | 指标、基线、消融 |
| inference / api | 服务与展示 |
