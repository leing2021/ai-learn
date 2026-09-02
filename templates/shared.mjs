// 共享布局与样式 —— github-html 视觉风格移植
export function layout({ title, content, chapters, activeSlug = '', nav = true, prev, next }) {
  const pager = prev || next ? `
  <nav class="pager">
    ${prev ? `<a class="pager-link prev" href="/ch/${prev.slug}.html">← 上一篇<br/><b>${prev.idx} ${prev.title}</b></a>` : '<span></span>'}
    ${next ? `<a class="pager-link next" href="/ch/${next.slug}.html">下一篇 →<br/><b>${next.idx} ${next.title}</b></a>` : '<span></span>'}
  </nav>` : '';
  const sidebar = nav ? `
  <aside class="sidebar">
    <a class="brand" href="/"><span class="brand-mark">AI</span>·learn</a>
    <p class="brand-sub">通过 MiniMind 学大模型</p>
    ${chapters.map(c => `
      <a class="toc-item ${c.slug === activeSlug ? 'active' : ''} ${c.status === 'planned' ? 'planned' : ''}" href="/ch/${c.slug}.html">
        <span class="toc-idx">${c.idx}</span>
        <span class="toc-title">${c.title}</span>
        ${c.status === 'done' ? '<span class="badge done">✓</span>' : c.status === 'partial' ? '<span class="badge partial">§</span>' : '<span class="badge planned">…</span>'}
      </a>`).join('')}
  </aside>` : '';
  return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>${title} · AI·learn</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🧠</text></svg>">
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
  mermaid.initialize({ startOnLoad: true, theme: 'neutral', flowchart: { curve: 'basis' } });
</script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/github.min.css">
<script src="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/highlight.min.js"></script>
<style>
:root { --bg:#fafafa; --fg:#24292f; --muted:#656d76; --border:#d8dee4; --accent:#0969da; --card:#fff; --green:#1a7f37; --amber:#9a6700; }
* { box-sizing: border-box; }
body { margin:0; font: 16px/1.75 -apple-system,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif; color:var(--fg); background:var(--bg); }
.layout { display:flex; min-height:100vh; }
.sidebar { width:264px; flex-shrink:0; border-right:1px solid var(--border); padding:20px 14px; background:var(--card); position:sticky; top:0; height:100vh; overflow-y:auto; }
.brand { font-size:20px; font-weight:700; color:var(--fg); text-decoration:none; }
.brand-mark { background:linear-gradient(135deg,#667eea,#764ba2); color:#fff; border-radius:8px; padding:2px 8px; }
.brand-sub { font-size:12px; color:var(--muted); margin:6px 0 18px; }
.toc-item { display:flex; gap:8px; align-items:baseline; padding:7px 10px; border-radius:8px; text-decoration:none; color:var(--fg); font-size:14px; margin-bottom:2px; }
.toc-item:hover { background:#eef1f4; }
.toc-item.active { background:#ddf0ff; color:var(--accent); font-weight:600; }
.toc-item.planned { color:var(--muted); }
.toc-idx { font-size:12px; color:var(--muted); min-width:16px; font-family:ui-monospace,monospace; }
.toc-item.active .toc-idx { color:var(--accent); }
.badge { margin-left:auto; font-size:12px; }
.badge.done{color:var(--green)} .badge.partial{color:var(--amber)} .badge.planned{color:var(--muted)}
main { flex:1; max-width:860px; margin:0 auto; padding:36px 28px 80px; }
h1 { font-size:30px; border-bottom:1px solid var(--border); padding-bottom:.3em; margin-top:0; }
h2 { font-size:23px; margin-top:2.2em; border-bottom:1px solid var(--border); padding-bottom:.25em; }
h3 { font-size:18px; margin-top:1.8em; }
blockquote { margin:1em 0; padding:.6em 1em; border-left:4px solid var(--accent); background:#f0f7ff; border-radius:0 8px 8px 0; color:#333; }
blockquote p { margin:.2em 0; }
code { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:87%; background:#eff1f3; padding:.15em .4em; border-radius:5px; }
pre { background:#f6f8fa; border:1px solid var(--border); border-radius:10px; padding:14px; overflow-x:auto; line-height:1.5; }
pre code { background:none; padding:0; font-size:13.5px; }
table { border-collapse:collapse; width:100%; margin:1.2em 0; font-size:14.5px; }
th,td { border:1px solid var(--border); padding:7px 12px; text-align:left; }
th { background:#f0f3f6; }
img { max-width:100%; border:1px solid var(--border); border-radius:10px; background:#fff; }
.mermaid { display:flex; justify-content:center; margin:1.4em 0; background:#fff; border:1px solid var(--border); border-radius:10px; padding:16px; }
.analogy { background:linear-gradient(135deg,#fffbe6,#fff3cd); border:1px solid #ffe08a; border-radius:12px; padding:14px 18px; margin:1.4em 0; }
.analogy::before { content:"💡 生活比喻"; display:block; font-weight:700; color:#9a6700; font-size:13px; margin-bottom:4px; }
.concept { background:linear-gradient(135deg,#f0f7ff,#e7f0ff); border:1px solid #b6d4fe; border-radius:12px; padding:14px 18px; margin:1.4em 0; }
.concept::before { content:"📌 核心概念正名"; display:block; font-weight:700; color:var(--accent); font-size:13px; margin-bottom:4px; }
.quiz { background:linear-gradient(135deg,#f6fff8,#eafbef); border:1px solid #b7e4c7; border-radius:12px; padding:14px 18px; margin:1.4em 0; }
.quiz::before { content:"✅ 自测 3 问（用术语作答）"; display:block; font-weight:700; color:var(--green); font-size:13px; margin-bottom:4px; }
.field { background:linear-gradient(135deg,#fdf2f9,#fce7f3); border:1px solid #fbcfe8; border-radius:12px; padding:14px 18px; margin:1.4em 0; }
.field::before { content:"🔬 实战验证（真实数据）"; display:block; font-weight:700; color:#be185d; font-size:13px; margin-bottom:4px; }
.term-link { text-decoration:none; border-bottom:1px dashed var(--accent); cursor:help; color:inherit; }
.term-link:hover { color:var(--accent); }
#term-tip { position:fixed; z-index:999; background:#24292f; color:#fff; padding:10px 14px; border-radius:10px;
  font-size:13px; line-height:1.6; max-width:360px; width:max-content; box-shadow:0 6px 20px rgba(0,0,0,.3);
  pointer-events:none; display:none; }
#term-tip .tt-def { font-weight:600; }
#term-tip .tt-ex { color:#9dc3ff; font-size:12.5px; margin-top:4px; border-top:1px solid rgba(255,255,255,.15); padding-top:4px; }
details { border:1px solid var(--border); border-radius:10px; padding:10px 16px; margin:0.8em 0; background:var(--card); }
details summary { cursor:pointer; font-weight:600; color:var(--green); font-size:14.5px; user-select:none; }
details[open] summary { border-bottom:1px dashed var(--border); padding-bottom:6px; margin-bottom:8px; }
details p { margin:.5em 0; }
.pager { display:flex; justify-content:space-between; gap:14px; margin-top:48px; border-top:1px solid var(--border); padding-top:18px; }
.pager-link { text-decoration:none; color:var(--muted); font-size:13px; border:1px solid var(--border); border-radius:10px; padding:10px 16px; max-width:46%; }
.pager-link b { color:var(--fg); font-size:14.5px; }
.pager-link:hover { border-color:var(--accent); }
.pager-link.next { text-align:right; margin-left:auto; }
.btn-play { display:inline-block; background:linear-gradient(135deg,#667eea,#764ba2); color:#fff; font-weight:700; padding:10px 22px; border-radius:10px; text-decoration:none; font-size:15px; margin:0.6em 0; }
.btn-play:hover { filter:brightness(1.08); }
.progress-bar { display:flex; gap:6px; margin:10px 0 24px; flex-wrap:wrap; }
.pill { font-size:12px; padding:3px 10px; border-radius:999px; border:1px solid var(--border); color:var(--muted); }
.pill.done { background:#dafbe1; color:var(--green); border-color:#b7e4c7; }
.pill.partial { background:#fff8c5; color:var(--amber); border-color:#ffe08a; }
.pill.planned { background:#f6f8fa; }
@media (max-width: 900px) { .layout{display:block} .sidebar{position:static;width:auto;height:auto;border-right:none;border-bottom:1px solid var(--border)} }
</style>
</head>
<body>
<div class="layout">
${sidebar}
<main>${content}${pager}</main>
</div>
<script>
document.querySelectorAll('pre code').forEach(b=>hljs.highlightElement(b));
// 术语悬浮卡: fixed 定位, 智能避让视口边缘, 完整显示 定义+示例
(function(){
  const tip = document.createElement('div'); tip.id = 'term-tip'; document.body.appendChild(tip);
  const show = (el) => {
    tip.innerHTML = '<div class="tt-def">' + el.getAttribute('data-tip') + '</div>' +
      (el.hasAttribute('data-ex') ? '<div class="tt-ex">📌 ' + el.getAttribute('data-ex') + '</div>' : '');
    tip.style.display = 'block';
    const r = el.getBoundingClientRect(), tw = tip.offsetWidth, th = tip.offsetHeight,
          vw = innerWidth, vh = innerHeight, m = 10;
    let x = r.left + r.width / 2 - tw / 2;               // 默认水平居中于链接
    x = Math.max(m, Math.min(x, vw - tw - m));           // 左右 clamp
    let y = r.top - th - m;                              // 优先上方
    if (y < m) y = r.bottom + m;                         // 上方空间不足 → 下方
    if (y + th > vh - m) y = Math.max(m, vh - th - m);   // 下方也超 → 顶部贴边
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  };
  document.addEventListener('mouseover', e => {
    const el = e.target.closest('.term-link[data-tip]'); if (el) show(el);
  });
  document.addEventListener('mouseout', e => { if (e.target.closest('.term-link')) tip.style.display = 'none'; });
  addEventListener('scroll', () => tip.style.display = 'none', { passive: true });
})();
</script>
</body>
</html>`;
}
