#!/usr/bin/env node
// 内容级校验：build 后必跑（防 $1 泄漏/空链接文本/属性污染/嵌套回归）
// 用法: node scripts/verify.mjs
import { readFileSync, readdirSync, existsSync } from 'fs';
import { execSync } from 'child_process';

let fail = 0;
const ok = (name) => console.log('✅', name);
const bad = (name, detail) => { console.log('❌', name, detail ?? ''); fail++; };

// 1. git 干净且 dist 与内容一致（防手改 dist 或忘 build）
const dirty = execSync('git status --porcelain -- dist/ content/ chapters.json terms.json').toString();
if (dirty) console.log('⚠️ dist/内容有未提交变更（若刚 build 过属正常，commit 前请再跑一次确认）'); else ok('dist 与源同步');

// 2. 每页内容级检查
const pages = readdirSync('dist/ch').filter(f => f.endsWith('.html'));
for (const f of pages) {
  const h = readFileSync(`dist/ch/${f}`, 'utf8');
  const body = h.split('<main>')[1] ?? '';
  body.includes('$1') ? bad(`${f}: $1 字面残留`) : ok(`${f}: 无 $1 残留`);
  const tags = [...body.matchAll(/<a class="term-link"([^>]*)>/g)].map(m => m[1]);
  const noTip = tags.filter(a => !a.includes('data-tip="'));
  noTip.length ? bad(`${f}: ${noTip.length} 链接缺 data-tip`) : ok(`${f}: ${tags.length} 术语链接属性完整`);
  const nested = body.match(/<a [^>]*>(?:(?!<\/a>).){0,600}?<a /s);
  nested ? bad(`${f}: 嵌套链接`) : ok(`${f}: 无嵌套`);
  const empty = [...body.matchAll(/<a class="term-link"[^>]*>\s*<\/a>/g)];
  empty.length ? bad(`${f}: 空链接文本`) : ok(`${f}: 链接文本非空`);
}

// 3. glossary 条目 = terms.json 条目（防漏渲染）
if (existsSync('dist/ch/glossary.html')) {
  const terms = JSON.parse(readFileSync('terms.json', 'utf8'));
  const g = readFileSync('dist/ch/glossary.html', 'utf8');
  const n = (g.match(/<div class="term-card"/g) ?? []).length;
  n === terms.terms.length ? ok(`glossary ${n} 条完整`) : bad(`glossary ${n} ≠ terms ${terms.terms.length}`);
  const ex = (g.match(/<p class="term-example">/g) ?? []).length;
  ex === terms.terms.length ? ok(`示例 ${ex} 条完整`) : bad(`示例 ${ex} ≠ ${terms.terms.length}`);
}

// 4. 概念库页在侧栏（防 layout 回归）
const any = readFileSync('dist/ch/overview.html', 'utf8');
any.includes('📖 概念库') ? ok('侧栏概念库入口') : bad('侧栏缺概念库');

console.log(fail ? `\n${fail} 项失败` : '\n全部通过');
process.exit(fail ? 1 : 0);
