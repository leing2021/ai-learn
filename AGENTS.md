# AGENTS.md — ai-learn 学习站

> 性质：大模型概念学习站（通过 MiniMind 项目学）。**单源 md → build 渲染 html**，内容随 minimind 学习进度滚动产出。

## 铁律

0. **视觉与交互规范强制遵守 `DESIGN.md`**（配色变量锁定/mermaid 语义色/互动元素/Playground 判定），新增章节前先读

1. 内容**只改 `content/*.md`**，`dist/` 是构建产物（提交但从不手改）
2. 改内容后：`node build.mjs` → 校验（HTML 关键词抽查）→ commit → push（EdgeOne 自动部署）
3. 新章节：`content/{slug}.md` + `chapters.json` 追加条目（slug 小写连字符，idx 用 ①②…）
4. 状态只有三档：`done`（全章完成）/ `partial`（有节待续写）/ `planned`（只有 json 占位，无源文件时 build 自动 skip）
5. 教学卡片语法（blockquote 首 emoji 触发，构建时转卡片）：
   - `> 💡 …` 生活比喻（黄卡）
   - `> 📌 …` 核心概念正名（蓝卡）
   - `> 🔬 …` 实战验证/真实数据（粉卡）
   - `> ✅ …` 自测题（绿卡）
   - `> ⚠️ …` 警告/坑（粉边）
6. 每章固定节奏：比喻 → mermaid → 概念正名 → 源码逐块讲（带行号）→ 实战 → 自测 3 问（考术语）
7. mermaid 用 ```` ```mermaid ```` 代码块（构建转 `<div class="mermaid">`，CDN 客户端渲染）
8. 教学深度：假设读者会读 Python，**不假设懂深度学习**（张量/softmax/梯度都要用比喻+正名解释）
9. 图片：放 `assets/`，md 里引用 `/assets/xxx.png`（绝对路径）；实验曲线从 minimind 仓 `notes/assets/` 复制
10. 隐私：站内容公开——**绝不放** token/IP/账号等敏感信息，实验数据（loss/耗时）无妨

## 内容源与滚动产出

| 章节 | 源 | 状态 |
|---|---|---|
| overview 全景 | — | done |
| 02 架构（MoE 等） | minimind M0/M1 笔记升维 | done |
| 03 数据 | 待 M 数据阶段 | planned |
| 04 训练范式 | Pretrain 节=K2 ✅；SFT 节=K3 完成后续写；LoRA=M4；蒸馏=K9 | partial |
| 05 对齐RL | DPO=K5、GRPO/CISPO=K6、PPO=K7 | planned |
| 06 Agent | K8 | planned |
| 07 思考 | M10 对比实验 | planned |

## 常用命令

```bash
node build.mjs              # 构建 dist/
python3 -m http.server -d dist 8000   # 本地预览
```

## 部署

push origin main → EdgeOne Pages 自动部署。
- 域名：**https://al.puless.com**（CNAME → pages.dnsoe4.com，2026-09-02 绑定）
- 仓：leing2021/ai-learn（公开），部署目录 `/`（dist 产物提交在根下）
- 验证：`curl -s https://al.puless.com/ | grep -c "AI·learn"` 应 >0
- 排障见 github-html 仓 AGENTS.md 的 EdgeOne 章节（同一套流水线）
  - 站点级 404 → empty commit 重新触发部署
  - SSL 证书不匹配（新绑定域名）→ EdgeOne 签发中，等待即可，勿改代码
