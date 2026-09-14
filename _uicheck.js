/* UI 冒烟：用最小 DOM stub 在 Node 里跑 <script id="ui">，抓只有浏览器才会炸的运行时错误。
   _smoke.js 只测 engine（结构上碰不到 UI 接线），这里补上：切场景/换核/拖滑块/重训/自检。
   用法: node _uicheck.js */
'use strict';
const fs = require('fs'), vm = require('vm'), path = require('path');
const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
const eng = html.match(/<script id="engine">([\s\S]*?)<\/script>/);
const ui = html.match(/<script id="ui">([\s\S]*?)<\/script>/);
if (!eng || !ui) { console.error('找不到 engine / ui 脚本块'); process.exit(1); }

/* ---- canvas 2d context stub：属性可读写，未知方法 no-op；
        createImageData/getImageData 必须返回真的 `data` 数组（等值线/虚线要写像素）。 ---- */
function ctx2d() {
  const o = {}, noop = function () { };
  return new Proxy(o, {
    get(t, k) {
      if (k in t) return t[k];
      if (k === 'createImageData' || k === 'getImageData') {
        return function (a, b) {
          const w = (a && typeof a === 'object') ? a.width : a;
          const h = (a && typeof a === 'object') ? a.height : b;
          return { width: w, height: h, data: new Uint8ClampedArray(Math.max(0, (w | 0) * (h | 0) * 4)) };
        };
      }
      if (k === 'measureText') return function () { return { width: 10 }; };
      return noop;
    },
    set(t, k, v) { t[k] = v; return true; }
  });
}

const initVals = {
  'inp-kind': 'separable', 'inp-kernel': 'rbf',
  'inp-n': '80', 'inp-noise': '0.10', 'inp-C': '0', 'inp-gamma': '0', 'inp-deg': '2'
};
const els = {};
function $(id) {
  if (!els[id]) {
    els[id] = {
      id, value: initVals[id] != null ? initVals[id] : '',
      textContent: '', innerHTML: '', clientWidth: 620, width: 0, height: 0,
      style: {}, disabled: false, parentNode: { clientWidth: 620 },
      getContext: () => ctx2d(), addEventListener() { }
    };
  }
  return els[id];
}

const errors = [], alerts = [];
const ctx = {
  console, Math, JSON, Array, Object, Number, String, Boolean, Error, isFinite, isNaN, Infinity, NaN,
  Float64Array, Uint8Array, Uint8ClampedArray, setTimeout: (fn) => fn(),
  document: {
    getElementById: $,
    createElement: (tag) => ({
      tagName: tag, width: 0, height: 0, clientWidth: 620, style: {},
      getContext: () => ctx2d(), addEventListener() { }
    })
  },
  window: { devicePixelRatio: 1, addEventListener() { } },
  alert: (s) => alerts.push(s)
};
ctx.globalThis = ctx;
vm.createContext(ctx);
vm.runInContext(eng[1], ctx, { filename: 'engine.js' });
if (!ctx.SVM) { console.error('engine 未暴露 SVM'); process.exit(1); }

function fire(name, fn) {
  try { fn(); } catch (e) {
    errors.push(name + ': ' + e.message + '\n' + (e.stack || '').split('\n').slice(0, 5).join('\n'));
  }
}

let uiRan = false;
try { vm.runInContext(ui[1], ctx, { filename: 'ui.js' }); uiRan = true; }
catch (e) { errors.push('初始渲染: ' + e.message + '\n' + (e.stack || '').split('\n').slice(0, 5).join('\n')); }

if (uiRan) {
  /* 场景切换（每个数据集都要能画出决策边界） */
  ['overlap', 'xor', 'moons', 'circles', 'separable'].forEach(k => {
    fire('场景=' + k, () => { $('inp-kind').value = k; $('inp-kind').onchange.call($('inp-kind')); });
  });
  /* 核切换（linear 走显式 w 分支，poly 走 Φ 维度分支，rbf 走核矩阵分支） */
  ['linear', 'poly', 'rbf', 'linear'].forEach(k => {
    fire('核=' + k, () => { $('inp-kernel').value = k; $('inp-kernel').onchange.call($('inp-kernel')); });
  });
  /* 拖滑块：含两端极值，C/γ 是指数标度（−2 → 0.01，3 → 1000） */
  [['inp-n', '20'], ['inp-n', '200'], ['inp-noise', '0'], ['inp-noise', '0.45'],
   ['inp-C', '-2'], ['inp-C', '3'], ['inp-gamma', '-2'], ['inp-gamma', '2'],
   ['inp-deg', '2'], ['inp-deg', '5']].forEach(([id, v]) => {
    fire('滑块 ' + id + '=' + v, () => { $(id).value = v; $(id).oninput.call($(id)); });
  });
  /* 回到默认配置再重训 */
  fire('恢复默认', () => {
    [['inp-kind', 'separable'], ['inp-kernel', 'rbf'], ['inp-n', '80'], ['inp-noise', '0.10'],
     ['inp-C', '0'], ['inp-gamma', '0'], ['inp-deg', '2']].forEach(([id, v]) => {
      $(id).value = v;
      if ($(id).onchange) $(id).onchange.call($(id));
      else if ($(id).oninput) $(id).oninput.call($(id));
    });
  });
  fire('重新训练 SMO', () => { $('btn-fit').onclick.call($('btn-fit')); });
  fire('运行全部自检', () => { $('btn-test').onclick.call($('btn-test')); });
}

const testsHTML = String($('tests').innerHTML);
const statsHTML = String($('stats').innerHTML);
console.log('UI 冒烟：' + (errors.length ? 'FAILED' : 'OK'));
if (!uiRan) console.log('  UI 脚本未能执行');
console.log('  已触发：5 个场景 / 3 种核 / 10 次滑块（含极值）/ 重训 / 自检');
console.log('  alert 次数 = ' + alerts.length + (alerts.length ? '（首条：' + String(alerts[0]).split('\n')[0] + '）' : ''));
console.log('  自检表格写入 = ' + (testsHTML.indexOf('PASS') >= 0 || testsHTML.indexOf('FAIL') >= 0) +
  '，testinfo = ' + JSON.stringify(String($('testinfo').innerHTML).slice(0, 60)));
console.log('  统计卡写入 = ' + statsHTML.length + ' 字符，含「间隔」= ' + (statsHTML.indexOf('间隔') >= 0) +
  '，含「支持向量」= ' + (statsHTML.indexOf('支持向量') >= 0));
console.log('  alpha 面板注记 = ' + JSON.stringify(String($('alphAnote').textContent).slice(0, 50)));
console.log('  目标曲线注记 = ' + JSON.stringify(String($('objnote').textContent).slice(0, 50)));
errors.forEach(e => console.log('\n✗ ' + e));
process.exitCode = errors.length ? 1 : 0;
