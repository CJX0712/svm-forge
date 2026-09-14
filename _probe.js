/* svm-forge 探针：把 engine 内部状态打成 ASCII，写 _probe.txt 供人眼复核 */
const fs = require('fs'), vm = require('vm'), path = require('path');
const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
const m = html.match(/<script id="engine">([\s\S]*?)<\/script>/);
const ctx = { console, Math, Object, Array, JSON, isFinite, Infinity, NaN, Number, String, Boolean, Error, Float64Array, Date, RegExp };
ctx.globalThis = ctx; vm.createContext(ctx); vm.runInContext(m[1], ctx, { filename: 'engine.js' });
const S = ctx.SVM;

const out = [];
function say(s) { out.push(s == null ? '' : s); }
function pad(v, n) { v = '' + v; while (v.length < n) v += ' '; return v; }
function padl(v, n) { v = '' + v; while (v.length < n) v = ' ' + v; return v; }
function f(v, d) { return Number(v).toFixed(d == null ? 6 : d); }
function e(v, d) { return Number(v).toExponential(d == null ? 3 : d); }

/* ═════════ 1. 四个场景的整体收敛质量 ═════════ */
say('══════════════════════════════════════════════════════════════════════');
say(' 1. SMO 在四个场景上的收敛质量（tol=1e-9）');
say('══════════════════════════════════════════════════════════════════════');
const SC = [
  { kind: 'separable', n: 40, k: { kernel: 'linear', C: 1 }, noise: 0.05 },
  { kind: 'overlap', n: 40, k: { kernel: 'rbf', C: 1.5, gamma: 1 }, noise: 0.20 },
  { kind: 'moons', n: 60, k: { kernel: 'rbf', C: 2, gamma: 1.5 }, noise: 0.10 },
  { kind: 'xor', n: 40, k: { kernel: 'poly', C: 2, deg: 2 }, noise: 0.10 },
  { kind: 'circles', n: 40, k: { kernel: 'rbf', C: 4, gamma: 3 }, noise: 0.08 }
];
say(pad('场景/核', 18) + padl('步数', 6) + padl('nSV', 5) + padl('自由', 5) + padl('界内', 5) +
  padl('KKT 违反', 12) + padl('对偶间隙', 12) + padl('间隔', 10) + padl('训练误差', 10));
const R = {};
SC.forEach((c, i) => {
  const d = S.makeDataset(c.kind, c.n, 4242 + i, c.noise);
  const r = S.smo(d.X, d.y, Object.assign({ tol: 1e-9, maxIter: 40000 }, c.k));
  R[c.kind] = { d, r, k: c.k };
  say(pad(c.kind + '/' + c.k.kernel, 18) + padl(r.steps, 6) + padl(r.nSV, 5) + padl(r.nFree, 5) +
    padl(r.nBounded, 5) + padl(e(r.kkt, 2), 12) +
    padl(e(Math.abs(r.primal - r.dual) / Math.max(1, Math.abs(r.primal)), 2), 12) +
    padl(f(r.margin, 4), 10) + padl((r.trainErr * 100).toFixed(1) + '%', 10));
});

/* ═════════ 2. b 的可行区间 vs 实际取值 ═════════ */
say('');
say('══════════════════════════════════════════════════════════════════════');
say(' 2. b 的 KKT 可行区间 vs SMO 给出的 b');
say('   （无自由 SV 时 b 由区间中点定；有自由 SV 时由自由 SV 的一致解定）');
say('══════════════════════════════════════════════════════════════════════');
say(pad('场景', 14) + padl('自由SV', 7) + padl('b(SMO)', 16) + padl('区间下界', 16) +
  padl('区间上界', 16) + padl('自由SV隐含b跨度', 18));
Object.keys(R).forEach(kind => {
  const { d, r, k } = R[kind], C = r.C;
  let lo = -Infinity, hi = Infinity;
  const implied = [];
  for (let i = 0; i < d.X.length; i++) {
    const g = r.f[i] - r.b, cls = S.classifyAlpha(r.alpha[i], C);
    if (cls === 1) implied.push(d.y[i] - g);
    if (d.y[i] > 0) {
      if (cls === 0) lo = Math.max(lo, 1 - g);
      if (cls === 2) hi = Math.min(hi, 1 - g);
    } else {
      if (cls === 0) hi = Math.min(hi, -1 - g);
      if (cls === 2) lo = Math.max(lo, -1 - g);
    }
  }
  const span = implied.length ? Math.max(...implied) - Math.min(...implied) : NaN;
  say(pad(kind, 14) + padl(r.nFree, 7) + padl(r.b.toFixed(12), 16) +
    padl(isFinite(lo) ? lo.toFixed(12) : '-inf', 16) +
    padl(isFinite(hi) ? hi.toFixed(12) : '+inf', 16) +
    padl(implied.length ? e(span, 2) : '(无)', 18));
});
say('');
say('  → 有自由 SV 时，自由点隐含的 b 跨度 ~1e-9（正是 SMO 的 tol 量级），说明它们给的是同一个超平面。');

/* ═════════ 3. α 分类容差的必要性（真实案例） ═════════ */
say('');
say('══════════════════════════════════════════════════════════════════════');
say(' 3. 为什么 α 的分类必须带容差（moons C=2 n=30 的实测）');
say('══════════════════════════════════════════════════════════════════════');
{
  const d = S.makeDataset('moons', 30, 4242, 0.18);
  const r = S.smo(d.X, d.y, { kernel: 'rbf', C: 2, gamma: 1, tol: 1e-9, maxIter: 30000 });
  const C = r.C;
  const rows = [];
  for (let i = 0; i < d.X.length; i++) {
    const a = r.alpha[i], v = d.y[i] * r.f[i];
    let clsT, vioT, clsE, vioE;
    const c = S.classifyAlpha(a, C);
    clsT = c === 0 ? '下界' : (c === 2 ? '上界' : '自由');
    vioT = c === 0 ? Math.max(0, 1 - v) : (c === 2 ? Math.max(0, v - 1) : Math.abs(v - 1));
    if (a <= 0) { clsE = '下界'; vioE = Math.max(0, 1 - v); }
    else if (a >= C) { clsE = '上界'; vioE = Math.max(0, v - 1); }
    else { clsE = '自由'; vioE = Math.abs(v - 1); }
    rows.push({ i, a, v, clsT, vioT, clsE, vioE });
  }
  rows.sort((p, q) => q.vioE - p.vioE);
  say('  用「精确 α==0 / α==C」判定 vs 用容差判定的前 3 名差异：');
  say('    ' + pad('i', 4) + pad('α', 24) + pad('y·f', 22) + pad('精确判定', 22) + '容差判定');
  rows.slice(0, 3).forEach(x => {
    say('    ' + pad(x.i, 4) + pad(e(x.a, 6), 24) + pad(x.v.toFixed(12), 22) +
      pad(x.clsE + ' viol=' + e(x.vioE, 2), 22) + x.clsT + ' viol=' + e(x.vioT, 2));
  });
  say('');
  say('  报告 KKT（带容差，即 engine 采用）= ' + e(r.kkt, 3));
  const loose = Math.max(...rows.map(x => x.vioT));
  const strict = Math.max(...rows.map(x => x.vioE));
  say('  同一解若用精确判定 = ' + e(strict, 3) + '  ← α≈5e-17 的浮点残渣被当成「自由 SV」而误报');
  say('  → 残渣型 α 的 y·f 与 1 的距离可以任意大，所以精确判定会产生假违反。');

  // b 被污染的情形：α 停在 C−1e-12
  const bad = rows.filter(x => x.a > C - 1e-9 && x.a < C);
  say('  同时 α 会停在 C−1e-12 处（本次 ' + bad.length + ' 个），若被当成自由 SV 会污染 b 的求解。');
}

/* ═════════ 4. SMO 与 PGA 的 α 逐项对照 ═════════ */
say('');
say('══════════════════════════════════════════════════════════════════════');
say(' 4. 两条完全不同的算法（SMO / 投影梯度上升）的解对照');
say('══════════════════════════════════════════════════════════════════════');
say(pad('场景', 14) + padl('D(SMO)', 16) + padl('D(PGA)', 16) + padl('|ΔD|', 12) +
  padl('max|Δα|', 12) + padl('max|Δw·x|', 13) + padl('PGA 迭代', 10));
['separable', 'overlap', 'moons'].forEach(kind => {
  const { d, r, k } = R[kind];
  const g = S.pgd(d.X, d.y, { kernel: k.kernel, C: k.C, gamma: k.gamma, deg: k.deg, iters: 40000 });
  let da = 0, df = 0;
  for (let i = 0; i < d.X.length; i++) {
    da = Math.max(da, Math.abs(r.alpha[i] - g.alpha[i]));
    df = Math.max(df, Math.abs((r.f[i] - r.b) - (g.f[i] - g.b)));
  }
  say(pad(kind, 14) + padl(r.dual.toFixed(12), 16) + padl(g.dual.toFixed(12), 16) +
    padl(e(Math.abs(r.dual - g.dual), 2), 12) + padl(e(da, 2), 12) + padl(e(df, 2), 13) + padl(g.iters, 10));
});
say('  → 目标值一致到 ~1e-13；α 允许有差异（对偶解可能不唯一），但 w·x 必须一致。');

/* ═════════ 5. 核矩阵与特征值 ═════════ */
say('');
say('══════════════════════════════════════════════════════════════════════');
say(' 5. 核矩阵（overlap 前 4×4）与全谱');
say('══════════════════════════════════════════════════════════════════════');
['linear', 'rbf', 'poly'].forEach(kn => {
  const { d } = R['overlap'];
  const Xs = d.X.slice(0, 24);
  const K = S.kernelMatrix(kn, Xs, { gamma: 1, deg: 2, coef0: 1 });
  const ev = S.jacobiEigen(K, 300);
  say('  ' + kn + '  前 4×4:');
  for (let i = 0; i < 4; i++) {
    say('    ' + K[i].slice(0, 4).map(v => padl(v.toFixed(6), 12)).join(''));
  }
  const tr = K.reduce((s, row, i) => s + row[i], 0);
  const se = ev.reduce((s, v) => s + v, 0);
  say('    λmax=' + f(ev[0], 6) + '  λmin=' + e(ev[ev.length - 1], 3) +
    '  Σλ=' + f(se, 6) + '  trace=' + f(tr, 6) + '  |差|=' + e(Math.abs(se - tr), 2));
});

/* ═════════ 6. 核技巧与显式特征映射 ═════════ */
say('');
say('══════════════════════════════════════════════════════════════════════');
say(' 6. 多项式核 (x·z+1)^d == 显式 φ 的内积（Φ 维度 = (d+1)(d+2)/2）');
say('══════════════════════════════════════════════════════════════════════');
{
  const rng = S.mulberry32(2026);
  say(pad('d', 4) + padl('Φ 维度', 9) + padl('最大核值', 14) + padl('后向误差', 13) + padl('朴素相对误差', 14));
  for (let dd = 2; dd <= 5; dd++) {
    let worstB = 0, worstR = 0, scale = 0;
    for (let t = 0; t < 200; t++) {
      const x = [S.gauss(rng) * 2, S.gauss(rng) * 2], z = [S.gauss(rng) * 2, S.gauss(rng) * 2];
      const kv = S.kernel('poly', x, z, { deg: dd, coef0: 1 });
      const px = S.polyFeatures(x, dd), pz = S.polyFeatures(z, dd);
      let dotv = 0, denom = 0;
      for (let q = 0; q < px.length; q++) { dotv += px[q] * pz[q]; denom += Math.abs(px[q] * pz[q]); }
      worstB = Math.max(worstB, Math.abs(kv - dotv) / Math.max(1, denom));
      worstR = Math.max(worstR, Math.abs(kv - dotv) / Math.max(1, Math.abs(kv)));
      scale = Math.max(scale, Math.abs(kv));
    }
    say(pad(dd, 4) + padl(S.polyFeatures([0, 0], dd).length, 9) + padl(scale.toFixed(0), 14) +
      padl(e(worstB, 2), 13) + padl(e(worstR, 2), 14));
  }
  say('  → 后向误差 ~1e-16 说明恒等式在浮点意义下精确；朴素相对误差被核值的指数动态范围放大。');
}

/* ═════════ 7. 最优性证书：方向角网格穷举 ═════════ */
say('');
say('══════════════════════════════════════════════════════════════════════');
say(' 7. 最优性证书：精确 O(n²) 枚举 vs 方向角网格（网格天生只是下界）');
say('══════════════════════════════════════════════════════════════════════');
say(pad('方向数', 12) + padl('含 b 的网格下界', 18) + padl('SMO 间隔', 14) + padl('差值', 12) + padl('过原点(错)下界', 18));
{
  const d = S.makeDataset('separable', 30, 777, 0.0);
  const r = S.smo(d.X, d.y, { kernel: 'linear', C: 1e6, tol: 1e-11, maxIter: 40000 });
  function originOnly(nDirs) {
    let best = -Infinity;
    for (let k = 0; k < nDirs; k++) {
      const th = Math.PI * k / nDirs, c = Math.cos(th), s = Math.sin(th);
      let mn = Infinity;
      for (let i = 0; i < d.X.length; i++) {
        const v = d.y[i] * (c * d.X[i][0] + s * d.X[i][1]);
        if (v < mn) mn = v;
      }
      if (mn > best) best = mn;
    }
    return 2 * best;
  }
  [1000, 10000, 100000, 1000000].forEach(nd => {
    const g = S.gridMarginBound(d.X, d.y, nd);
    say(pad(nd, 12) + padl(f(g, 9), 18) + padl(f(r.margin, 9), 14) + padl(e(r.margin - g, 2), 12) +
      padl(f(originOnly(nd), 9), 18));
  });
  say('  → 含 b 的下界随加密单调收敛到 SMO 的间隔（误差 O(1/方向数)），且永不超越。');
  say('    「过原点」那一列系统性低约 1.7%，且加密不收敛 —— 这是实现这个证书最容易踩的坑。');
  let mn = Infinity;
  for (let i = 0; i < d.y.length; i++) mn = Math.min(mn, d.y[i] * r.f[i]);
  say('  SMO 解的 min(y·f) = ' + f(mn, 10) + '（硬间隔 C=1e6 下应恰为 1），训练误差 = ' + r.trainErr +
    '，nSV = ' + r.nSV + '（其中自由 ' + r.nFree + '）');
  say('  线性核的显式 w = [' + S.explicitW(r.alpha, d.y, d.X).map(v => v.toFixed(6)).join(', ') +
    ']，b = ' + f(r.b, 6));
}

say('');
say('  近可分数据的对照（overlap/s6002，26 点 / 标签噪声 0.25）：可行方向锥极窄，网格会严重低估');
{
  const d2 = S.makeDataset('overlap', 26, 6002, 0.25);
  const r2 = S.smo(d2.X, d2.y, { kernel: 'linear', C: 1e5, tol: 1e-11, maxIter: 40000 });
  const ex2 = S.exactMargin2D(d2.X, d2.y);
  const g5k = S.gridMarginBound(d2.X, d2.y, 5000);
  const g50k = S.gridMarginBound(d2.X, d2.y, 50000);
  const g500k = S.gridMarginBound(d2.X, d2.y, 500000);
  say('  ' + pad('方法', 24) + padl('最大间隔', 14) + padl('相对精确的缺口', 16));
  [['5 千方向网格', g5k], ['5 万方向网格', g50k], ['50 万方向网格', g500k]].forEach(([nm, g]) => {
    say('  ' + pad(nm, 24) + padl(f(g, 8), 14) + padl(((ex2.margin - g) / ex2.margin * 100).toFixed(2) + '%', 16));
  });
  say('  ' + pad('精确枚举 O(n²)', 24) + padl(f(ex2.margin, 8), 14) + padl('—', 16));
  say('  ' + pad('SMO（实际收敛解）', 24) + padl(f(r2.margin, 8), 14) +
    padl((ex2.margin - r2.margin).toExponential(1), 16));
  say('  → 该数据其实**线性可分**（精确间隔 ' + f(ex2.margin, 6) + ' > 0），但 50 万方向网格只给 ' +
    f(g500k, 6) + '，');
  say('    会把可分数据误判成不可分（进而误认为「最大间隔证书不适用」）。');
  say('    实测 SMO 与精确枚举差 ' + e(ex2.margin - r2.margin, 2) + '，训练误差 ' + r2.trainErr +
    '，nSV=' + r2.nSV + '（自由 ' + r2.nFree + ' / 界内 ' + r2.nBounded + '），' + r2.iters + ' 轮收敛。');
}

/* ═════════ 8. 正则化的作用：C 扫描 ═════════ */
say('');
say('══════════════════════════════════════════════════════════════════════');
say(' 8. C 扫描（overlap, RBF γ=1）：C 越小越「软」');
say('══════════════════════════════════════════════════════════════════════');
{
  const d = S.makeDataset('overlap', 40, 5151, 0.25);
  say(pad('C', 10) + padl('nSV', 5) + padl('自由', 5) + padl('界内', 5) + padl('间隔 2/‖w‖', 12) +
    padl('训练误差', 10) + padl('‖w‖²', 12) + padl('对偶目标', 12));
  [0.01, 0.05, 0.2, 1, 5, 20, 100].forEach(C => {
    const r = S.smo(d.X, d.y, { kernel: 'rbf', C: C, gamma: 1, tol: 1e-9, maxIter: 40000 });
    say(pad(C, 10) + padl(r.nSV, 5) + padl(r.nFree, 5) + padl(r.nBounded, 5) +
      padl(f(r.margin, 4), 12) + padl((r.trainErr * 100).toFixed(1) + '%', 10) +
      padl(f(r.w2, 4), 12) + padl(f(r.dual, 4), 12));
  });
  say('  → C 增大：间隔带被压缩（‖w‖² ↑，间隔 ↓）、界内 SV 增多、训练误差单调不增。');
}

/* ═════════ 9. 退化与边界输入 ═════════ */
say('');
say('══════════════════════════════════════════════════════════════════════');
say(' 9. 退化输入');
say('══════════════════════════════════════════════════════════════════════');
{
  const cases = [
    ['n=1 单点单类', [[0, 0]], [1], { kernel: 'rbf', C: 1, gamma: 1 }],
    ['n=2 两点两类', [[-1, 0], [1, 0]], [1, -1], { kernel: 'linear', C: 1 }],
    ['全同类标签', [[0, 0], [1, 1], [2, 1]], [1, 1, 1], { kernel: 'linear', C: 1 }],
    ['完全重合的重复点', [[1, 1], [1, 1], [1, 1], [0, 0], [0, 0], [0, 0]],
      [1, 1, -1, 1, -1, -1], { kernel: 'linear', C: 1 }]
  ];
  say(pad('情形', 20) + padl('α', 34) + padl('b', 12) + padl('KKT', 11) + '有限性');
  cases.forEach(([name, X, y, kp]) => {
    const r = S.smo(X, y, Object.assign({ tol: 1e-10 }, kp));
    const finite = r.alpha.every(isFinite) && isFinite(r.b) && isFinite(r.dual) && isFinite(r.primal);
    say(pad(name, 20) + padl('[' + r.alpha.map(v => v.toFixed(4)).join(', ') + ']', 34) +
      padl(f(r.b, 6), 12) + padl(e(r.kkt, 2), 11) + (finite ? ' 全部有限 ✓' : ' 出现 NaN/Inf ✗'));
  });
}

/* ═════════ 10. 线性核的显式 w ═════════ */
say('');
say('══════════════════════════════════════════════════════════════════════');
say(' 10. 线性核：w = Σ αᵢyᵢxᵢ 的显式表示 vs 核展开');
say('══════════════════════════════════════════════════════════════════════');
{
  const d = S.makeDataset('overlap', 40, 3131, 0.20);
  const r = S.smo(d.X, d.y, { kernel: 'linear', C: 1, tol: 1e-10, maxIter: 40000 });
  const w = S.explicitW(r.alpha, d.y, d.X);
  say('  w = [' + w.map(v => v.toFixed(12)).join(', ') + ']');
  say('  b = ' + r.b.toFixed(12));
  let worst = 0, worstW2 = 0;
  for (let i = 0; i < d.X.length; i++) {
    worst = Math.max(worst, Math.abs(w[0] * d.X[i][0] + w[1] * d.X[i][1] + r.b - r.f[i]));
  }
  worstW2 = Math.abs(w[0] * w[0] + w[1] * w[1] - r.w2);
  say('  max|(w·xᵢ+b) − f(xᵢ)| = ' + e(worst, 2) + '（逐点一致）');
  say('  |‖w‖²_显式 − ‖w‖²_核展开| = ' + e(worstW2, 2));
  say('  间隔 2/‖w‖ = ' + f(r.margin, 8) + '，法向量方向 ' +
    f(Math.atan2(w[1], w[0]) * 180 / Math.PI, 3) + '°');
}

/* ═════════ 11. 子问题解析解的一阶最优性 ═════════ */
say('');
say('══════════════════════════════════════════════════════════════════════');
say(' 11. 二变量子问题：解析解 vs 无导数搜索 vs 一阶条件');
say('══════════════════════════════════════════════════════════════════════');
{
  const d = R['overlap'].d, r = R['overlap'].r;
  const rng = S.mulberry32(31415);
  const H = 1e-5;
  let ci = 0, cj = 0;
  const atSub = x => {
    const v = r.alpha.slice();
    v[cj] = x;
    v[ci] = r.alpha[ci] + d.y[ci] * d.y[cj] * (r.alpha[cj] - x);
    return S.dualObjective(v, r.Q);
  };
  say(pad('i', 5) + padl('j', 4) + padl('L', 10) + padl('H', 10) + padl('α_j* 无约束', 14) +
    padl('解析解', 12) + padl('搜索解', 12) + padl('Δ目标', 11) + padl('D′(解)', 11) + '  形态');
  let printed = 0, attempts = 0;
  while (printed < 8 && attempts < 5000) {
    attempts++;
    const i = Math.floor(rng() * d.X.length), j = Math.floor(rng() * d.X.length);
    if (i === j) continue;
    const sp = S.subproblem2(r.K, d.y, r.alpha, r.b, r.f, r.C, i, j);
    if (!(sp.H > sp.L)) continue;
    ci = i; cj = j;
    const g = S.goldenMaxSubproblem(r.Q, d.y, r.alpha, i, j, sp.L, sp.H, 300);
    const va = atSub(sp.clamped);
    const d1 = (atSub(sp.clamped + H) - atSub(sp.clamped - H)) / (2 * H);
    let shape = '内点(D′≈0)';
    if (sp.clamped <= sp.L + 1e-12) shape = Math.abs(sp.unconstrained - sp.L) < 1e-12 ? '内点(D′≈0)' : 'clamp→L';
    else if (sp.clamped >= sp.H - 1e-12) shape = Math.abs(sp.unconstrained - sp.H) < 1e-12 ? '内点(D′≈0)' : 'clamp→H';
    say(pad(i, 5) + padl(j, 4) + padl(f(sp.L, 6), 10) + padl(f(sp.H, 6), 10) +
      padl(f(sp.unconstrained, 6), 14) + padl(f(sp.clamped, 6), 12) + padl(f(g.x, 6), 12) +
      padl(e(Math.abs(va - g.val), 2), 11) + padl(e(Math.abs(d1), 2), 11) + '  ' + shape);
    printed++;
  }
  say('  → 解析解处 D′≈0（内点）或符号正确（clamp）；目标值与黄金分割搜索一致到 ~1e-15。');
  say('    argmax 本身只精确到 √(2ε/η)（ε≈1e-15 为求值噪声），故断言 Δα²·η 而非 Δα。');
}

/* ═════════ 12. 置换不变性与网格证书 ═════════ */
say('');
say('══════════════════════════════════════════════════════════════════════');
say(' 12. 置换不变性（数据打乱次序）');
say('══════════════════════════════════════════════════════════════════════');
say(pad('种子', 8) + padl('|ΔD|', 12) + padl('|Δ‖w‖²|', 12) + padl('|Δb|', 12) + padl('SV 集合是否一致', 16));
function svSet(r) {
  const s = [];
  for (let i = 0; i < r.alpha.length; i++) if (S.classifyAlpha(r.alpha[i], r.C) !== 0) s.push(i);
  return s.join(',');
}
['separable', 'overlap', 'moons'].forEach(kind => {
  const { d, r, k } = R[kind];
  for (const seed of [3, 9, 17]) {
    const idx = d.X.map((_, i) => i);
    const prng = S.mulberry32(seed);
    for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(prng() * (i + 1)); const t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
    const Xp = idx.map(i => d.X[i]), yp = idx.map(i => d.y[i]);
    const rp = S.smo(Xp, yp, Object.assign({ tol: 1e-9, maxIter: 40000 }, k));
    // 把置换后的 SV 索引映射回原索引再比较
    const mapped = [];
    for (let t = 0; t < idx.length; t++) if (S.classifyAlpha(rp.alpha[t], rp.C) !== 0) mapped.push(idx[t]);
    mapped.sort((a, b) => a - b);
    const orig = svSet(r).split(',').filter(x => x !== '').map(Number).sort((a, b) => a - b);
    say(pad(kind + '/s' + seed, 8) + padl(e(Math.abs(r.dual - rp.dual), 2), 12) +
      padl(e(Math.abs(r.w2 - rp.w2), 2), 12) + padl(e(Math.abs(r.b - rp.b), 2), 12) +
      padl(orig.join(',') === mapped.join(',') ? '一致 (' + orig.length + ' 个)' : '不一致', 16));
  }
});

/* ═════════ 13. 自检面板 ═════════ */
say('');
say('══════════════════════════════════════════════════════════════════════');
say(' 13. 浏览器内 8 条不变量自检');
say('══════════════════════════════════════════════════════════════════════');
{
  const res = S.selfTest({ seed: 20260915 });
  res.forEach(r => {
    say('  ' + (r.pass ? '[PASS] ' : '[FAIL] ') + r.name);
    say('         ' + r.detail);
  });
  say('');
  say('  ' + res.filter(r => r.pass).length + ' / ' + res.length + ' 通过');
}

fs.writeFileSync(path.join(__dirname, '_probe.txt'), out.join('\n') + '\n');
console.log(out.join('\n'));
