#!/usr/bin/env node
// 构建：content/*.md + chapters.json → dist/（首页 + ch/{slug}.html）
// 约定： ```mermaid 块 → <div class="mermaid">；> 💡/> 📌/> ✅/> 🔬/> ⚠️ 开头 blockquote → 对应教学卡片
import { readFileSync, writeFileSync, mkdirSync, copyFileSync, readdirSync, existsSync } from 'fs';
import { marked } from 'marked';

const chapters = JSON.parse(readFileSync('chapters.json', 'utf8'));

marked.use({
  renderer: {
    code({ text, lang }) {
      if (lang === 'mermaid') return `<div class="mermaid">${text}</div>`;
      const cls = lang ? ` class="language-${lang}"` : '';
      return `<pre><code${cls}>${text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}</code></pre>`;
    },
    blockquote(token) {
      // "> 💡 xxx" 风格 blockquote 转教学卡片（v12 token.text 已去掉 > 前缀）
      const src = (token.text ?? token.raw ?? '').trim();
      const m = src.match(/^(💡|📌|✅|🔬|⚠️)/);
      let cls = '', body = src;
      if (m) {
        const map = { '💡':'analogy', '📌':'concept', '✅':'quiz', '🔬':'field', '⚠️':'warn' };
        cls = map[m[1]];
        body = src.slice(m[1].length).trim();
      }
      const inner = marked.parse(body);
      return cls ? `<div class="${cls}">${inner}</div>` : `<blockquote>${inner}</blockquote>`;
    },
  },
});

const { layout } = await import('./templates/shared.mjs');
const terms = JSON.parse(readFileSync('terms.json', 'utf8'));
chapters.push({ idx: '≡', slug: 'glossary', title: '📖 概念库', desc: '术语字典', status: 'done', tags: [] });

// ---- 正文术语自动链接: 每术语每页首现, 避开 code/pre/a 内部 ----
function autolink(html, skipSlugs = []) {
  // 保护块: code/pre/a/已有的 glossary 链接
  const prot = [];
  html = html.replace(/<pre[\s\S]*?<\/pre>|<code[\s\S]*?<\/code>|<a [\s\S]*?<\/a>/g, m => {
    prot.push(m); return `\u0000${prot.length - 1}\u0000`;
  });
  const esc = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const attr = s => s.replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;');
  for (const t of terms.terms) {
    if (skipSlugs.includes(t.slug)) continue;
    const tip = attr(t.def);
    const ex = attr(t.example || '');
    const link = `<a class="term-link" data-tip="${tip}"${ex ? ` data-ex="${ex}"` : ''} href="/ch/glossary.html#${t.slug}">$1</a>`;
    // 两种形态都处理: "Term（中文）" 全形先替换, 剩余裸英文词再全局替换
    const full = `${t.term}（${t.zh}）`;
    // 三段式防嵌套: ①保护既有链接 ②全形替换 ③再保护全形产物 ④裸词替换 ⑤还原
    // (否则: 后续术语污染先前链接属性 / 同术语裸词二次包裹全形产物)
    // 占位符必须分区(p2/p3 用不同标记), 否则 restore(p3) 会把 p2 的占位错还原成 p3 内容
    const protect = (mark) => {
      const p = [];
      html = html.replace(/<a class="term-link"[\s\S]*?<\/a>/g, m => { p.push(m); return `${mark}${p.length - 1}${mark}`; });
      return p;
    };
    const restore = (p, mark) => { html = html.replace(new RegExp(`${mark}(\\d+)${mark}`, 'g'), (_, i) => p[+i]); };
    const p2 = protect('\u0001');
    html = html.replace(new RegExp(`(${esc(full)})`, 'g'), link);
    const p3 = protect('\u0002');
    html = html.replace(new RegExp(`(\\b${esc(t.term)}\\b)`, 'g'), link);
    restore(p3, '\u0002'); restore(p2, '\u0001');
  }
  return html.replace(/\u0000(\d+)\u0000/g, (_, i) => prot[+i]);
}
mkdirSync('dist/ch', { recursive: true });
mkdirSync('dist/assets', { recursive: true });

// 章节页（含 prev/next 导航）
const withSrc = chapters.filter(c => existsSync(`content/${c.slug}.md`));
for (const c of chapters) {
  const src = `content/${c.slug}.md`;
  if (!existsSync(src)) { console.log(`[skip] ${c.slug} 无源文件`); continue; }
  const i = withSrc.findIndex(x => x.slug === c.slug);
  const prev = i > 0 ? withSrc[i - 1] : null;
  const next = i >= 0 && i < withSrc.length - 1 ? withSrc[i + 1] : null;
  let html = marked.parse(readFileSync(src, 'utf8'));
  html = html.replace(/src="assets\//g, 'src="/assets/');  // 图片绝对路径
  html = autolink(html, c.slug === 'glossary' ? terms.terms.map(t => t.slug) : []);
  writeFileSync(`dist/ch/${c.slug}.html`,
    layout({ title: c.title, content: html, chapters, activeSlug: c.slug, prev, next }));
  console.log(`[ok] ch/${c.slug}.html`);
}

// 概念库页
{
  const cats = terms.categories;
  const cards = cats.map(cat => `
    <h2 id="${cat}">${cat}</h2>
    ${terms.terms.filter(t => t.cat === cat).map(t => `
    <div class="term-card" id="${t.slug}">
      <div class="term-head"><span class="term-en">${t.term}</span><span class="term-zh">${t.zh}</span></div>
      <p class="term-def">${t.def}</p>
      <p class="term-example">📌 示例：${t.example}</p>
      <details><summary>展开细节</summary><p>${t.detail}</p>
      ${t.refs && t.refs.length ? `<p class="term-refs">相关章节：${t.refs.map(r => `<a href="/ch/${r}.html">${r}</a>`).join(' · ')}</p>` : ''}
      </details>
    </div>`).join('')}`).join('');
  const content = `
<h1>📖 概念原子库 <span style="font-size:15px;color:var(--muted);font-weight:400">${terms.terms.length} 个 · 按分类 · 点击正文术语可跳转至此</span></h1>
<input id="term-search" type="text" placeholder="🔍 搜索概念（英/中）…" oninput="filterTerms(this.value)" style="width:100%;padding:10px 14px;border:1px solid var(--border);border-radius:10px;font-size:15px;margin-bottom:20px"/>
${cards}
<script>
function filterTerms(q){
  q = q.toLowerCase();
  document.querySelectorAll('.term-card').forEach(c=>{
    c.style.display = c.textContent.toLowerCase().includes(q) ? '' : 'none';
  });
}
</script>
<style>
.term-card { border:1px solid var(--border); border-radius:12px; padding:12px 18px; margin:10px 0; background:var(--card); }
.term-card:target { border-color:var(--accent); box-shadow:0 0 0 3px rgba(9,105,218,.15); }
.term-head { display:flex; gap:10px; align-items:baseline; }
.term-en { font-weight:700; font-size:16.5px; }
.term-zh { color:var(--muted); font-size:14px; }
.term-def { margin:6px 0 2px; }
.term-example { margin:4px 0; font-size:14px; color:#333; background:#f6f8fa; border-left:3px solid var(--accent); padding:4px 10px; border-radius:0 6px 6px 0; }
.term-refs a { color:var(--accent); text-decoration:none; font-size:13.5px; }
.term-link { text-decoration:none; border-bottom:1px dashed var(--accent); }
</style>`;
  mkdirSync('dist/ch', { recursive: true });
  writeFileSync('dist/ch/glossary.html', layout({ title: '概念库', content, chapters, activeSlug: 'glossary' }));
  console.log('[ok] glossary.html');
}

// 首页：全景
const { renderIndex } = await import('./templates/index.mjs');
writeFileSync('dist/index.html', renderIndex({ chapters, layout }));
console.log('[ok] index.html');

// assets 复制
if (existsSync('assets')) for (const f of readdirSync('assets')) {
  copyFileSync(`assets/${f}`, `dist/assets/${f}`);
  console.log(`[copy] assets/${f}`);
}
console.log('BUILD DONE');
