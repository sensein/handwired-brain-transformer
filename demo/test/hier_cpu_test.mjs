// JS-CPU vs Python golden for hier1 programs (node, no browser): node demo/test/hier_cpu_test.mjs [demo-root]
// Continuous outputs: max |diff| relative to max |ref|. Binary (step) outputs: fraction of voxels whose 0.5-threshold differs
// (steep ramps amplify float32 vs float64 rounding at the threshold; a handful of boundary voxels may flip).
import fs from 'fs'; import path from 'path';
const root = process.argv[2] || path.resolve('demo');
const H = await import(path.join(root, 'hier.js'));
const gold = JSON.parse(fs.readFileSync(path.join(root, 'test/golden_hier.json'), 'utf8'));
const report = {}; let worst = 0, worstFlip = 0;
for (const [name, g] of Object.entries(gold)) {
  const prog = H.parseHier(fs.readFileSync(path.join(root, 'models', name + '.json'), 'utf8'));
  const t0 = Date.now(); const o = H.runHierCPU(prog, Float32Array.from(g.logI), g.dims, g.spacing); const dt = Date.now() - t0;
  for (const [k, ref] of Object.entries(g.out)) {
    const a = o[k];
    if (typeof ref === 'number') { report[`${name}.${k}`] = { diff: Math.abs(a - ref), ref }; worst = Math.max(worst, Math.abs(a - ref) / Math.max(1, Math.abs(ref))); continue; }
    let md = 0, mx = 0, flips = 0, binary = true;
    for (let i = 0; i < ref.length; i++) { md = Math.max(md, Math.abs(a[i] - ref[i])); mx = Math.max(mx, Math.abs(ref[i])); if ((a[i] > 0.5) !== (ref[i] > 0.5)) flips++; if (ref[i] > 1e-6 && ref[i] < 1 - 1e-6) binary = false; }
    report[`${name}.${k}`] = { maxdiff: +md.toExponential(2), refmax: +mx.toFixed(4), flipFrac: +(flips / ref.length).toExponential(2), binary };
    if (binary) worstFlip = Math.max(worstFlip, flips / ref.length); else worst = Math.max(worst, md / Math.max(mx, 1e-6));
  }
  report[`${name}.ms`] = dt;
}
console.log(JSON.stringify(report, null, 1));
console.log(`worst relative diff (continuous) ${worst.toExponential(2)}, worst flip fraction (binary) ${worstFlip.toExponential(2)}`);
process.exit(worst < 1e-2 && worstFlip < 5e-3 ? 0 : 1);
