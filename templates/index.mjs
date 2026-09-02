// 首页渲染：学习全景图 + 篇章卡片
export function renderIndex({ chapters, layout }) {
  const cards = chapters.filter(c => c.slug !== 'overview' && c.slug !== 'glossary').map(c => `
    <a class="card ${c.status}" href="/ch/${c.slug}.html">
      <div class="card-idx">${c.idx}</div>
      <div class="card-body">
        <div class="card-title">${c.title} ${c.status === 'done' ? '<span class="badge done">✓ 已完成</span>' : c.status === 'partial' ? '<span class="badge partial">§ 部分完成</span>' : '<span class="badge planned">… 规划中</span>'}</div>
        <div class="card-desc">${c.desc}</div>
        <div class="card-tags">${(c.tags||[]).map(t=>`<span class="tag">${t}</span>`).join('')}</div>
      </div>
    </a>`).join('');

  const content = `
<h1>🧠 AI·learn <span style="font-size:16px;color:var(--muted);font-weight:400">—— 通过 MiniMind 学大模型</span></h1>
<p>这是我的大模型学习知识库：从 MiniMind（一个 <b>4647 行代码</b>的迷你 GPT）出发，逐个吃透大模型的热门概念。
每个概念都按同一节奏讲：<b>生活比喻 → 图解 → 核心概念正名 → 源码逐块讲 → 真实训练数据验证 → 自测</b>。</p>

<div class="mermaid">
flowchart LR
    A[① 大模型是什么] --> B[② 模型架构<br/>MoE/Attention/RoPE]
    B --> C[③ 数据<br/>清洗/数据集]
    C --> D[④ 训练范式<br/>Pretrain/SFT/LoRA/蒸馏]
    D --> E[⑤ 对齐与RL<br/>DPO/PPO/GRPO/CISPO]
    E --> F[⑥ Agent与工具<br/>Tool Call/Agentic RL]
    E --> G[⑦ 思考与推理<br/>自适应思考]
    style A fill:#dafbe1,stroke:#1a7f37
    style B fill:#dafbe1,stroke:#1a7f37
    style D fill:#fff8c5,stroke:#9a6700
    style C fill:#f6f8fa,stroke:#656d76
    style E fill:#f6f8fa,stroke:#656d76
    style F fill:#f6f8fa,stroke:#656d76
    style G fill:#f6f8fa,stroke:#656d76
</div>

<p style="color:var(--muted);font-size:14px">🟢 绿=已完成 · 🟡 黄=部分 · ⚪ 灰=规划中（随 Kaggle 训练推进滚动更新）</p>

${cards}

<blockquote>
<p><b>🔗 关联资源</b>：源码仓 <a href="https://github.com/jingyaogong/minimind">jingyaogong/minimind</a> ·
实验日志（含魔鬼问题自答）在我本地的 minimind 仓 notes/ 目录 ·
所有训练数据来自我在 Kaggle T4 上跑的真实 kernel。</p>
</blockquote>

<style>
.card { display:flex; gap:16px; border:1px solid var(--border); border-radius:14px; padding:18px 20px; margin:14px 0; text-decoration:none; color:var(--fg); background:var(--card); transition:border-color .15s; }
.card:hover { border-color:var(--accent); }
.card.planned { opacity:.62; }
.card-idx { font-size:26px; font-weight:800; color:var(--border); min-width:44px; font-family:ui-monospace,monospace; }
.card-title { font-weight:700; font-size:17px; margin-bottom:3px; }
.card-desc { color:var(--muted); font-size:14px; }
.card-tags { margin-top:8px; display:flex; gap:6px; flex-wrap:wrap; }
.tag { font-size:12px; background:#eff1f3; padding:2px 9px; border-radius:999px; color:var(--muted); }
</style>`;
  return layout({ title: '全景', content, chapters, activeSlug: 'overview', nav: true });
}
