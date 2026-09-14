/* svm-forge 无头验证：从 index.html 抽出 engine 在 Node vm 里跑 */
const fs = require('fs'), vm = require('vm'), path = require('path');
const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
const m = html.match(/<script id="engine">([\s\S]*?)<\/script>/);
if (!m) { console.log('ENGINE NOT FOUND'); process.exit(1); }

const ctx = {
  console, Math, Object, Array, JSON, isFinite, Infinity, NaN,
  Number, String, Boolean, Error, Float64Array, Date, RegExp
};
ctx.globalThis = ctx;
vm.createContext(ctx);
vm.runInContext(m[1], ctx, { filename: 'engine.js' });
const S = ctx.SVM;

let pass = 0, fail = 0;
const fails = [];
function ok(cond, name, detail) {
  if (cond) { pass++; } else { fail++; fails.push(name + '  << ' + (detail == null ? '' : detail)); }
}
function info(s) { console.log(s); }
const rel = (a, b) => Math.abs(a - b) / Math.max(1e-12, Math.max(Math.abs(a), Math.abs(b)));

/* ══════════ 1. 主循环：4 场景 × 4 种子 × 3 核 ══════════ */
const KINDS = ['separable', 'overlap', 'xor', 'moons', 'circles'];
const KERS = [
  { kernel: 'linear', C: 1, tag: 'lin' },
  { kernel: 'rbf', C: 1, gamma: 1, tag: 'rbf' },
  { kernel: 'rbf', C: 4, gamma: 3, tag: 'rbf4' },
  { kernel: 'poly', C: 1.5, deg: 2, tag: 'poly2' },
  { kernel: 'poly', C: 1, deg: 3, tag: 'poly3' }
];

KINDS.forEach(kind => {
  KERS.forEach(kp => {
    [0, 1, 7].forEach(seed => {
      const n = 30 + (seed % 3) * 10;
      const d = S.makeDataset(kind, n, 500 + seed, 0.15);
      const o = Object.assign({ tol: 1e-9, maxIter: 30000 }, kp);
      const r = S.smo(d.X, d.y, o);
      const t = `${kind}/${kp.tag}/s${seed}`;

      // KKT 五条件
      ok(r.kkt < 1e-6, `${t} KKT 违反 < 1e-6`, r.kkt.toExponential(2));
      ok(r.eqRes < 1e-9, `${t} |Σαᵢyᵢ| < 1e-9`, r.eqRes.toExponential(2));
      let boxOK = true, mixed = false, hasZero = false, hasC = false, hasFree = false;
      for (let i = 0; i < r.alpha.length; i++) {
        const v = r.alpha[i];
        if (v < -1e-12 || v > r.C + 1e-12) boxOK = false;
        if (v <= 1e-10) hasZero = true;
        else if (v >= r.C - 1e-10) hasC = true;
        else hasFree = true;
      }
      ok(boxOK, `${t} 0 ≤ α ≤ C`, 'out of box');
      ok(r.alpha.length === n, `${t} α 长度 == n`, `${r.alpha.length} vs ${n}`);

      // 强对偶（相对间隙）
      const gapRel = rel(r.primal, r.dual);
      ok(gapRel < 1e-8, `${t} 强对偶 rel(P−D) < 1e-8`, gapRel.toExponential(2));

      // 对偶目标单调非减
      let minD = Infinity;
      for (let i = 1; i < r.hist.length; i++) minD = Math.min(minD, r.hist[i] - r.hist[i - 1]);
      ok(minD > -1e-10, `${t} 对偶目标单调非减`, `minΔ=${minD.toExponential(2)}`);

      // 自由支持向量恰在间隔上
      let mxFree = 0, cntFree = 0;
      for (let i = 0; i < r.alpha.length; i++) {
        if (S.classifyAlpha(r.alpha[i], r.C) === 1) {
          mxFree = Math.max(mxFree, Math.abs(d.y[i] * r.f[i] - 1)); cntFree++;
        }
      }
      ok(mxFree < 1e-8, `${t} 自由 SV 满足 y·f == 1`, `max|Δ|=${mxFree.toExponential(2)} n=${cntFree}`);

      // α=0 的点必须在间隔外（或恰在间隔上）
      let mxZero = 0;
      for (let i = 0; i < r.alpha.length; i++) {
        if (S.classifyAlpha(r.alpha[i], r.C) === 0) mxZero = Math.max(mxZero, Math.max(0, 1 - d.y[i] * r.f[i]));
      }
      ok(mxZero < 1e-6, `${t} α=0 ⇒ y·f ≥ 1`, `viol=${mxZero.toExponential(2)}`);

      // α=C 的点必须在间隔内或错分
      let mxC = 0;
      for (let i = 0; i < r.alpha.length; i++) {
        if (S.classifyAlpha(r.alpha[i], r.C) === 2) mxC = Math.max(mxC, Math.max(0, d.y[i] * r.f[i] - 1));
      }
      ok(mxC < 1e-6, `${t} α=C ⇒ y·f ≤ 1`, `viol=${mxC.toExponential(2)}`);

      // 分类计数与 SV 计数自洽
      let c0 = 0, c1 = 0, c2 = 0;
      for (let i = 0; i < n; i++) {
        const cl = S.classifyAlpha(r.alpha[i], r.C);
        if (cl === 0) c0++; else if (cl === 1) c1++; else c2++;
      }
      ok(c0 + c1 + c2 === n, `${t} 分类划分完备`, '');
      ok(c1 === r.nFree && c2 === r.nBounded && c1 + c2 === r.nSV,
        `${t} SV 计数与分类一致`, `${c1}/${r.nFree} ${c2}/${r.nBounded} ${c1 + c2}/${r.nSV}`);

      // 间隔与 ‖w‖² 自洽
      if (r.w2 > 0) {
        ok(Math.abs(r.margin - 2 / Math.sqrt(r.w2)) < 1e-12, `${t} 间隔 == 2/√‖w‖²`, '');
      }

      // 决策函数重算一致：由 α 与 K 直接算出的 f 必须与 r.f 逐项一致
      const f2 = S.decisionValues(r.alpha, r.b, d.y, r.K);
      let mxF = 0;
      for (let i = 0; i < n; i++) mxF = Math.max(mxF, Math.abs(f2[i] - r.f[i]));
      ok(mxF < 1e-10, `${t} f 重算逐项一致`, mxF.toExponential(2));
    });
  });
});

/* ══════════ 2. 边界与退化输入 ══════════ */
{
  // n=2，最小可行问题
  const X = [[-1, 0], [1, 0]], y = [1, -1];
  const r = S.smo(X, y, { kernel: 'linear', C: 1, tol: 1e-10 });
  ok(r.alpha.length === 2, 'n=2 不崩', '');
  ok(r.kkt < 1e-6, 'n=2 KKT', r.kkt.toExponential(2));
  ok(r.eqRes < 1e-9, 'n=2 等式约束', r.eqRes.toExponential(2));

  // n=1：单点、单类——等式约束 Σαy=0 强制 α=0
  const r1 = S.smo([[0, 0]], [1], { kernel: 'rbf', C: 1, gamma: 1, tol: 1e-10 });
  ok(r1.alpha.length === 1 && isFinite(r1.dual) && isFinite(r1.b), 'n=1 不崩且 b 有限', String(r1.b));

  // 全同类标签（人为构造的病态输入）：等式约束会把 α 全压到 0
  const rSame = S.smo([[0, 0], [1, 1], [2, 1]], [1, 1, 1], { kernel: 'linear', C: 1, tol: 1e-10 });
  ok(rSame.alpha.every(v => Math.abs(v) < 1e-9), '全同类 ⇒ α 全 0', JSON.stringify(rSame.alpha));
  ok(isFinite(rSame.dual), '全同类对偶目标有限', String(rSame.dual));

  // 线性可分 + 极大 C ⇒ 硬间隔：训练误差 0 且 min y·f == 1
  const ds = S.makeDataset('separable', 40, 909, 0.0);
  const rh = S.smo(ds.X, ds.y, { kernel: 'linear', C: 1e6, tol: 1e-10, maxIter: 40000 });
  ok(rh.trainErr === 0, '可分 + C=1e6 ⇒ 训练误差 0', String(rh.trainErr));
  let mn = Infinity;
  for (let i = 0; i < ds.y.length; i++) mn = Math.min(mn, ds.y[i] * rh.f[i]);
  ok(Math.abs(mn - 1) < 1e-5, '硬间隔下 min(y·f) == 1', mn.toFixed(9));

  // 重复点（η = 0 的退化情形）不产生 NaN
  const Xdup = [[0, 0], [0, 0], [1, 1], [1, 1], [0, 1], [1, 0]], ydup = [1, 1, -1, -1, 1, -1];
  const rdup = S.smo(Xdup, ydup, { kernel: 'linear', C: 1, tol: 1e-10 });
  ok(rdup.alpha.every(isFinite), '重复点（η=0）α 全有限', JSON.stringify(rdup.alpha));
  ok(isFinite(rdup.b), '重复点 b 有限', String(rdup.b));
}

/* ══════════ 3. 核的代数性质 ══════════ */
{
  const rng = S.mulberry32(31337);
  let worstPoly = 0, worstSym = 0, worstDiag = 0, worstTrans = 0, polyScale = 0;
  for (let d = 2; d <= 5; d++) {
    for (let t = 0; t < 8; t++) {
      const x = [S.gauss(rng) * 2, S.gauss(rng) * 2], z = [S.gauss(rng) * 2, S.gauss(rng) * 2];
      const kv = S.kernel('poly', x, z, { deg: d, coef0: 1 });
      const px = S.polyFeatures(x, d), pz = S.polyFeatures(z, d);
      // 核值随 d 指数增长（实测可达 2e6），绝对误差与朴素相对误差都被动态范围主导；
      // 点积恒等式的正确度量是「求和条件数归一化的后向误差」Σ|φᵢ(x)φᵢ(z)|
      let denom = 0;
      for (let q = 0; q < px.length; q++) denom += Math.abs(px[q] * pz[q]);
      worstPoly = Math.max(worstPoly, Math.abs(kv - dotArr(px, pz)) / Math.max(1, denom));
      polyScale = Math.max(polyScale, Math.abs(kv));
    }
  }
  function dotArr(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  ok(worstPoly < 1e-15, 'poly^d 核 == 显式 φ 内积（d=2..5，条件数归一化）',
    worstPoly.toExponential(2) + ' 核值最大 ' + polyScale.toFixed(0));

  const Xs = [];
  for (let i = 0; i < 20; i++) Xs.push([S.gauss(rng) * 2, S.gauss(rng) * 2]);
  ['linear', 'rbf', 'poly'].forEach(kn => {
    const K = S.kernelMatrix(kn, Xs, { gamma: 1.3, deg: 2, coef0: 1 });
    for (let i = 0; i < Xs.length; i++) {
      for (let j = 0; j < Xs.length; j++) worstSym = Math.max(worstSym, Math.abs(K[i][j] - K[j][i]));
    }
    if (kn === 'rbf') {
      for (let i = 0; i < Xs.length; i++) worstDiag = Math.max(worstDiag, Math.abs(K[i][i] - 1));
    }
    const ev = S.jacobiEigen(K, 300);
    ok(ev[ev.length - 1] > -1e-9, `${kn} Gram 矩阵半正定`, `λmin=${ev[ev.length - 1].toExponential(2)}`);
  });
  ok(worstSym < 1e-15, '核矩阵对称', worstSym.toExponential(2));
  ok(worstDiag < 1e-15, 'RBF K(x,x) == 1', worstDiag.toExponential(2));

  for (let t = 0; t < 30; t++) {
    const a = [S.gauss(rng) * 2, S.gauss(rng) * 2], b = [S.gauss(rng) * 2, S.gauss(rng) * 2];
    const s = [4.2, -3.1];
    worstTrans = Math.max(worstTrans, Math.abs(
      S.kernel('rbf', a, b, { gamma: 0.7 }) -
      S.kernel('rbf', [a[0] + s[0], a[1] + s[1]], [b[0] + s[0], b[1] + s[1]], { gamma: 0.7 })));
  }
  ok(worstTrans < 1e-14, 'RBF 平移不变', worstTrans.toExponential(2));

  // Jacobi 特征分解本身：随机对称阵，特征值之和 == 迹，且 A 的特征多项式残差小
  for (let t = 0; t < 5; t++) {
    const nn = 6, A = [];
    for (let i = 0; i < nn; i++) { A.push([]); for (let j = 0; j < nn; j++) A[i].push(0); }
    for (let i = 0; i < nn; i++) for (let j = i; j < nn; j++) { const v = S.gauss(rng); A[i][j] = v; A[j][i] = v; }
    const ev = S.jacobiEigen(A, 200);
    let tr = 0; for (let i = 0; i < nn; i++) tr += A[i][i];
    let se = 0; for (let i = 0; i < nn; i++) se += ev[i];
    ok(Math.abs(tr - se) < 1e-9 * Math.max(1, Math.abs(tr)), 'Jacobi 特征值之和 == 迹', `${tr} vs ${se}`);
  }
}

/* ══════════ 4. 二变量子问题：一阶条件 + 无导数搜索 ══════════ */
{
  const rng = S.mulberry32(2468);
  const H = 1e-5;
  ['separable', 'overlap', 'moons', 'xor'].forEach(kind => {
    const d = S.makeDataset(kind, 34, 1717, 0.18);
    const o = { kernel: 'rbf', C: 1, gamma: 1.2, tol: 1e-9 };
    const r = S.smo(d.X, d.y, o);
    let worstDv = 0, noWorse = 0, worstDeriv = 0, worstCurv = 0, tested = 0, interior = 0, corner = 0;
    let ci = 0, cj = 1;
    const atSub = x => {
      const v = r.alpha.slice();
      v[cj] = x;
      v[ci] = r.alpha[ci] + d.y[ci] * d.y[cj] * (r.alpha[cj] - x);
      return S.dualObjective(v, r.Q);
    };
    for (let t = 0; t < 300; t++) {
      const i = Math.floor(rng() * d.X.length), j = Math.floor(rng() * d.X.length);
      if (i === j) continue;
      const sp = S.subproblem2(r.K, d.y, r.alpha, r.b, r.f, r.C, i, j);
      if (!(sp.H > sp.L)) continue;
      ci = i; cj = j;
      const g = S.goldenMaxSubproblem(r.Q, d.y, r.alpha, i, j, sp.L, sp.H, 300);
      const va = atSub(sp.clamped);
      worstDv = Math.max(worstDv, Math.abs(va - g.val));
      noWorse = Math.max(noWorse, g.val - va);
      // 一阶最优性：内点 D'≈0；clamp 到下界 D'≤0；clamp 到上界 D'≥0
      const d1 = (atSub(sp.clamped + H) - atSub(sp.clamped - H)) / (2 * H);
      let violD;
      if (sp.clamped <= sp.L + 1e-12) { violD = Math.max(0, d1); corner++; }
      else if (sp.clamped >= sp.H - 1e-12) { violD = Math.max(0, -d1); corner++; }
      else { violD = Math.abs(d1); interior++; }
      worstDeriv = Math.max(worstDeriv, violD);
      // Δα 的定位精度上限 √(2ε/η)，故断言 Δα²·η 而非 Δα
      if (sp.eta > 1e-6) worstCurv = Math.max(worstCurv, (g.x - sp.clamped) * (g.x - sp.clamped) * sp.eta);
      tested++;
    }
    ok(tested > 60, `${kind} 子问题样本量足够`, String(tested));
    ok(interior >= 5, `${kind} 含内点样本（一阶条件的两侧分支都被覆盖）`, String(interior));
    ok(worstDv < 1e-13, `${kind} 子问题目标值一致（解析 vs 搜索）`, worstDv.toExponential(2));
    ok(noWorse < 1e-13, `${kind} 解析解不劣于无导数搜索`, noWorse.toExponential(2));
    ok(worstDeriv < 1e-7, `${kind} 解析解满足一阶最优性条件`, worstDeriv.toExponential(2));
    ok(worstCurv < 1e-12, `${kind} argmax 曲率界 Δα²η`, worstCurv.toExponential(2));
    info(`  ${kind}: tested=${tested} interior=${interior} corner=${corner} Δobj=${worstDv.toExponential(2)} ` +
      `Δα²η=${worstCurv.toExponential(2)} D'viol=${worstDeriv.toExponential(2)}`);
  });
}

/* ══════════ 5. SMO == 投影梯度上升（独立算法） ══════════ */
{
  ['separable', 'overlap', 'moons'].forEach((kind, ki) => {
    const d = S.makeDataset(kind, 28, 4242 + ki, 0.18);
    const o = { kernel: 'rbf', C: 1.5, gamma: 1, tol: 1e-10, maxIter: 30000 };
    const a = S.smo(d.X, d.y, o);
    const g = S.pgd(d.X, d.y, { kernel: 'rbf', C: 1.5, gamma: 1, iters: 40000 });
    const diff = Math.abs(a.dual - g.dual);
    ok(rel(a.dual, g.dual) < 1e-6, `${kind} SMO == PGA 对偶最优`, `${a.dual.toFixed(10)} vs ${g.dual.toFixed(10)}`);
    ok(g.kkt < 1e-4, `${kind} PGA 自身满足 KKT`, g.kkt.toExponential(2));
    ok(g.eqRes < 1e-9, `${kind} PGA 等式约束`, g.eqRes.toExponential(2));
    ok(Math.abs(a.w2 - g.w2) / Math.max(1e-12, a.w2) < 1e-5, `${kind} ‖w‖² 一致`, `${a.w2.toFixed(8)} vs ${g.w2.toFixed(8)}`);
    // PGA 的目标也不应优于 SMO（SMO 是更强的求解器）：允许 1e-9 的数值余量
    ok(g.dual <= a.dual + 1e-7, `${kind} PGA 未超过 SMO`, (g.dual - a.dual).toExponential(2));
  });
}

/* ══════════ 6. 投影算子本身 ══════════ */
{
  const rng = S.mulberry32(99);
  for (let t = 0; t < 60; t++) {
    const n = 4 + Math.floor(rng() * 8), C = 0.5 + rng() * 2, v = [], y = [];
    for (let i = 0; i < n; i++) { v.push(S.gauss(rng) * 3); y.push(rng() < 0.5 ? 1 : -1); }
    const a = S.projectBoxEq(v, y, C);
    let boxOK = true, s = 0;
    for (let i = 0; i < n; i++) { if (a[i] < -1e-9 || a[i] > C + 1e-9) boxOK = false; s += y[i] * a[i]; }
    ok(boxOK, `投影落在盒内 #${t}`, '');
    ok(Math.abs(s) < 1e-6, `投影满足 Σyα=0 #${t}`, s.toExponential(2));
    // 幂等性：投影两次 == 投影一次
    const a2 = S.projectBoxEq(a, y, C);
    let idem = 0;
    for (let i = 0; i < n; i++) idem = Math.max(idem, Math.abs(a[i] - a2[i]));
    ok(idem < 1e-7, `投影幂等 #${t}`, idem.toExponential(2));
    // 投影是到可行集的最近点：投影结果离 v 不比自己远
    let dA = 0, dZero = 0;
    for (let i = 0; i < n; i++) { const t2 = a[i] - v[i]; dA += t2 * t2; }
    const z = S.projectBoxEq(v.map(() => 0), y, C);
    for (let i = 0; i < n; i++) { const t2 = z[i] - v[i]; dZero += t2 * t2; }
    ok(dA <= dZero + 1e-7, `投影最优性 #${t}`, `${dA} vs ${dZero}`);
  }
}

/* ══════════ 7. 置换不变性 ══════════ */
{
  const rng = S.mulberry32(555);
  ['separable', 'overlap', 'moons'].forEach(kind => {
    const d = S.makeDataset(kind, 36, 8080, 0.15);
    [3, 9, 17].forEach(seed => {
      const o = { kernel: 'rbf', C: 2, gamma: 1, tol: 1e-9, maxIter: 30000 };
      const r0 = S.smo(d.X, d.y, o);
      const r1 = S.makeDataset(kind, 36, 8080, 0.15);
      const idx = r1.X.map((_, i) => i);
      const prng = S.mulberry32(seed);
      for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(prng() * (i + 1)); const t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
      const Xp = idx.map(i => r1.X[i]), yp = idx.map(i => r1.y[i]);
      const rp = S.smo(Xp, yp, o);
      ok(rel(r0.dual, rp.dual) < 1e-7, `${kind}/perm${seed} 对偶目标不变`, `${r0.dual} vs ${rp.dual}`);
      ok(Math.abs(r0.w2 - rp.w2) < 1e-6, `${kind}/perm${seed} ‖w‖² 不变`, `${r0.w2} vs ${rp.w2}`);
      ok(Math.abs(r0.b - rp.b) < 1e-5, `${kind}/perm${seed} b 不变`, `${r0.b} vs ${rp.b}`);
    });
  });
}

/* ══════════ 8. 2D 硬间隔**精确**最优性证书：SMO == O(n²) 候选方向枚举 ══════════ */
/* 证书只在**线性可分**时成立：此时「max_w min_i y_i(w·x+b)/‖w‖」可行，
   exactMargin2D 给出的就是精确最大间隔。不可分时该 max ≤ 0（问题不可行），
   SMO 的 2/‖w‖ 也不再是「最大间隔」——所以先用可行性把两类数据分开。
   注意：**不能**用方向角网格判可行性。近可分数据的可行方向锥可以极窄，
   20 万方向网格在 overlap/s6002 上只给 0.0435，而精确最优是 0.4227（低估一个数量级）。 */
{
  const cases = [
    { kind: 'separable', seed: 6000, noise: 0.0 },
    { kind: 'separable', seed: 6001, noise: 0.08 },
    { kind: 'overlap', seed: 6002, noise: 0.25 },
    { kind: 'overlap', seed: 6004, noise: 0.60 },
    { kind: 'moons', seed: 6003, noise: 0.10 },
    { kind: 'xor', seed: 6005, noise: 0.10 }
  ];
  let nFeasible = 0, nInfeasible = 0;
  cases.forEach(c => {
    const tag = `${c.kind}/s${c.seed}/nz${c.noise}`;
    const d = S.makeDataset(c.kind, 26, c.seed, c.noise);
    const r = S.smo(d.X, d.y, { kernel: 'linear', C: 1e5, tol: 1e-11, maxIter: 40000 });
    const ex = S.exactMargin2D(d.X, d.y);
    const grid = S.gridMarginBound(d.X, d.y, 200000);
    const feasible = ex.feasible;       // 几何判据：精确枚举的最大间隔 > 0
    const zeroErr = r.trainErr === 0;   // 经验判据：大 C 下没有错分

    // 两条完全独立的判据必须给出同一结论（不硬编码标签，从数据推导）
    ok(feasible === zeroErr, `${tag} 「最大间隔>0」⟺「训练误差=0」判据一致`,
      `grid=${grid.toFixed(4)} err=${r.trainErr}`);

    if (!feasible) {
      nInfeasible++;
      ok(ex.margin <= 0, `${tag} 不可分 ⇔ 精确最大间隔 ≤ 0`, `exact=${ex.margin.toFixed(6)}`);
      ok(grid <= ex.margin + 1e-9, `${tag} 不可分时网格下界同样不超过精确最优`,
        `${grid.toFixed(6)} vs ${ex.margin.toFixed(6)}`);
      // 不可分数据配 C=1e5 会让对偶严重病态（α∈[0,1e5]），SMO 4 万轮内不保证收敛，
      // 所以「不可分时 KKT 仍成立」这件事换一个良态 C 来验证。
      const rs = S.smo(d.X, d.y, { kernel: 'linear', C: 1, tol: 1e-9, maxIter: 40000 });
      ok(rs.kkt < 1e-6, `${tag} 不可分时（C=1）KKT 仍满足`, rs.kkt.toExponential(2));
      info(`  ${tag}: 不可分（精确最大间隔 ${ex.margin.toFixed(4)} ≤ 0），最大间隔证书不适用；` +
        `C=1 时 KKT=${rs.kkt.toExponential(2)}`);
      return;
    }
    nFeasible++;

    // 闭环：把 SMO 自己解出的方向（由 α 构造的 w）代进同一个 f(θ)，应恰好取到枚举出的最大值
    const wv = S.explicitW(r.alpha, d.y, d.X);
    const nw = Math.hypot(wv[0], wv[1]), cw = wv[0] / nw, sw = wv[1] / nw;
    let mP = Infinity, mN = -Infinity;
    for (let i = 0; i < d.X.length; i++) {
      const v = cw * d.X[i][0] + sw * d.X[i][1];
      if (d.y[i] > 0) { if (v < mP) mP = v; } else { if (v > mN) mN = v; }
    }
    ok(Math.abs((mP - mN) - ex.margin) < 1e-9, `${tag} SMO 解出的方向即为精确最优方向`,
      `f(ŵ)=${(mP - mN).toFixed(10)} vs exact=${ex.margin.toFixed(10)}`);

    // ★ 核心断言：SMO 的间隔 == 精确枚举的全局最优（两条完全独立的算法路径）
    ok(Math.abs(r.margin - ex.margin) < 1e-9, `${tag} SMO 间隔 == 精确枚举最优`,
      `SMO=${r.margin.toFixed(10)} exact=${ex.margin.toFixed(10)} Δ=${(r.margin - ex.margin).toExponential(2)}`);
    // 网格只是下界：任何网格方向都不超过精确最优（同时校验 gridMarginBound 与 exactMargin2D）
    ok(grid <= ex.margin + 1e-9, `${tag} 网格下界 ≤ 精确最优`,
      `${grid.toFixed(6)} vs ${ex.margin.toFixed(6)}`);
    // 加密：下界单调不减、且始终不超越精确最优
    const g1 = S.gridMarginBound(d.X, d.y, 5000);
    const g2 = S.gridMarginBound(d.X, d.y, 50000);
    const g3 = S.gridMarginBound(d.X, d.y, 500000);
    ok(g2 >= g1 - 1e-12 && g3 >= g2 - 1e-12, `${tag} 网格下界随加密单调不减`,
      `${g1.toFixed(6)} -> ${g2.toFixed(6)} -> ${g3.toFixed(6)}`);
    ok(g3 <= ex.margin + 1e-9, `${tag} 加密后仍不超越精确最优`,
      `${g3.toFixed(6)} vs ${ex.margin.toFixed(6)}`);

    // 对照：只扫过原点的方向（不优化 b）会系统性低估间隔，且加密不收敛
    let originOnly = -Infinity;
    for (let k2 = 0; k2 < 200000; k2++) {
      const th = Math.PI * k2 / 200000, cc = Math.cos(th), ss = Math.sin(th);
      let mn = Infinity;
      for (let i2 = 0; i2 < d.X.length; i2++) {
        const v = d.y[i2] * (cc * d.X[i2][0] + ss * d.X[i2][1]);
        if (v < mn) mn = v;
      }
      if (mn > originOnly) originOnly = mn;
    }
    const originMargin = 2 * originOnly;
    ok(originMargin <= ex.margin + 1e-9, `${tag}「过原点」网格 ≤ 精确最优`,
      `${originMargin.toFixed(6)} vs ${ex.margin.toFixed(6)}`);
    ok(ex.margin - originMargin > 1e-3, `${tag} 不优化 b 会系统性低估间隔`,
      `低估 ${(ex.margin - originMargin).toFixed(4)}`);
    info(`  ${tag}: SMO=${r.margin.toFixed(6)} 精确=${ex.margin.toFixed(6)} ` +
      `20万网格=${grid.toFixed(6)}(低估 ${(ex.margin - grid).toExponential(1)}) ` +
      `过原点=${originMargin.toFixed(6)}(低估 ${(ex.margin - originMargin).toFixed(4)})`);
  });
  ok(nFeasible >= 2, '可分样本覆盖到证书的适用分支', String(nFeasible));
  ok(nInfeasible >= 2, '不可分样本覆盖到证书的排除分支', String(nInfeasible));
}

/* ══════════ 9. 线性核：显式 w == 核展开 ══════════ */
{
  const d = S.makeDataset('overlap', 40, 3131, 0.2);
  const r = S.smo(d.X, d.y, { kernel: 'linear', C: 1, tol: 1e-10 });
  const w = S.explicitW(r.alpha, d.y, d.X);
  let worst = 0;
  for (let i = 0; i < d.X.length; i++) {
    const viaW = w[0] * d.X[i][0] + w[1] * d.X[i][1] + r.b;
    worst = Math.max(worst, Math.abs(viaW - r.f[i]));
  }
  ok(worst < 1e-10, '线性核：w=Σαᵢyᵢxᵢ 与核展开一致', worst.toExponential(2));
  let w2 = 0;
  for (let i = 0; i < w.length; i++) w2 += w[i] * w[i];
  ok(Math.abs(w2 - r.w2) < 1e-9, '‖w‖² 两条路径一致', `${w2} vs ${r.w2}`);
}

/* ══════════ 10. refineB 的两种分支 ══════════ */
{
  // 有自由 SV
  const d1 = S.makeDataset('moons', 30, 1212, 0.1);
  const r1 = S.smo(d1.X, d1.y, { kernel: 'rbf', C: 1, gamma: 1, tol: 1e-10 });
  ok(r1.nFree > 0, 'moons 存在自由 SV', String(r1.nFree));
  // 无自由 SV：C 很小 ⇒ α 全落在边界上
  const d2 = S.makeDataset('moons', 30, 1212, 0.1);
  const r2 = S.smo(d2.X, d2.y, { kernel: 'rbf', C: 0.02, gamma: 1, tol: 1e-10 });
  ok(isFinite(r2.b), 'C 极小（可能无自由 SV）b 仍有限', String(r2.b));
  ok(r2.kkt < 1e-4, 'C 极小 KKT 仍满足', r2.kkt.toExponential(2));
}

/* ══════════ 11. 自检面板本身可跑且全绿 ══════════ */
{
  const res = S.selfTest({ seed: 20260915 });
  ok(res.length === 8, 'selfTest 返回 8 条', String(res.length));
  res.forEach(r => ok(r.pass, `selfTest ${r.name.slice(0, 22)}`, r.detail));
  info('');
  info('── selfTest 面板 ──');
  res.forEach(r => info(`  ${r.pass ? 'OK  ' : 'FAIL'} ${r.name}`));
  res.forEach(r => info(`       ${r.detail}`));
}

info('');
info(`PASS ${pass} / ${pass + fail}`);
fs.writeFileSync(path.join(__dirname, '_smoke.log'), `PASS ${pass} / ${pass + fail}\n` + (fail ? 'FAIL\n' + fails.join('\n') : 'ALL GREEN') + '\n');
console.log(fail ? 'FAIL ' + fails.length : 'ALL GREEN');
if (fail) { console.log(fails.slice(0, 40).join('\n')); }
