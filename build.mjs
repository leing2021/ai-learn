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
mkdirSync('dist/ch', { recursive: true });
mkdirSync('dist/assets', { recursive: true });

// 章节页
for (const c of chapters) {
  const src = `content/${c.slug}.md`;
  if (!existsSync(src)) { console.log(`[skip] ${c.slug} 无源文件`); continue; }
  let html = marked.parse(readFileSync(src, 'utf8'));
  html = html.replace(/src="assets\//g, 'src="/assets/');  // 图片绝对路径
  writeFileSync(`dist/ch/${c.slug}.html`,
    layout({ title: c.title, content: html, chapters, activeSlug: c.slug }));
  console.log(`[ok] ch/${c.slug}.html`);
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
