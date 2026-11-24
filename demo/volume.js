// Efficient WebGL2 volume renderer: ray-marches an 8-bit 3D texture (adjusted log-intensity) with an optional label
// texture (model segmentation). Modes: MIP, composite (front-to-back with early termination), isosurface (gradient shading).
// Renders only on demand (interaction), orthographic camera with aspect-correct physical box.
const VS = `#version 300 es
in vec2 p; out vec2 uv; void main() { uv = p; gl_Position = vec4(p, 0., 1.); }`;
const FS = `#version 300 es
precision highp float; precision highp sampler3D;
in vec2 uv; out vec4 o;
uniform sampler3D uVol, uLab; uniform mat3 uRot; uniform vec3 uBox; uniform vec2 uAspect; uniform float uZoom;
uniform int uMode, uSteps; uniform bool uHasLab; uniform float uThr, uOp, uSegOp; uniform vec3 uPal[16];
vec3 lab(vec3 t, out float on) { float v = texture(uLab, t).r * 255.0; int k = int(v + 0.5); on = (uHasLab && k > 0) ? 1.0 : 0.0; return uPal[clamp(k, 0, 15)] / 255.0; }
void main() {
  vec3 ro = uRot * vec3(uv * uAspect / uZoom, -2.0), rd = uRot * vec3(0., 0., 1.);
  vec3 inv = 1.0 / rd, t0 = (-uBox - ro) * inv, t1 = (uBox - ro) * inv, tmin = min(t0, t1), tmax = max(t0, t1);
  float tn = max(max(tmin.x, tmin.y), tmin.z), tf = min(min(tmax.x, tmax.y), tmax.z);
  if (tf <= max(tn, 0.0)) { o = vec4(0., 0., 0., 1.); return; }
  tn = max(tn, 0.0); float dt = (tf - tn) / float(uSteps);
  vec4 acc = vec4(0.); float best = 0.0; vec3 bestCol = vec3(0.); bool hit = false; vec3 hp = vec3(0.);
  for (int i = 0; i < 1024; i++) {
    if (i >= uSteps) break;
    vec3 p = ro + rd * (tn + (float(i) + 0.5) * dt), tc = p / uBox * 0.5 + 0.5;
    float s = texture(uVol, tc).r, on; vec3 lc = lab(tc, on);
    if (uMode == 0) { if (s > best) { best = s; bestCol = mix(vec3(s), lc, on * uSegOp); } }
    else if (uMode == 1) {
      float a = clamp((s - uThr) / max(1.0 - uThr, 1e-3), 0.0, 1.0) * uOp * dt * 8.0; a = min(a, 1.0);
      vec3 c = mix(vec3(s), lc, on * uSegOp); if (on > 0.5) a = max(a, uSegOp * uOp * dt * 8.0);
      acc.rgb += (1.0 - acc.a) * a * c; acc.a += (1.0 - acc.a) * a; if (acc.a > 0.97) break;
    } else if (s > uThr) { hit = true; hp = tc; best = s; bestCol = lc; break; }
  }
  if (uMode == 0) o = vec4(bestCol * uOp, 1.);
  else if (uMode == 1) o = vec4(acc.rgb, 1.);
  else if (hit) {
    vec3 e = 1.0 / vec3(textureSize(uVol, 0)); float g0 = texture(uVol, hp + vec3(e.x, 0., 0.)).r - texture(uVol, hp - vec3(e.x, 0., 0.)).r,
      g1 = texture(uVol, hp + vec3(0., e.y, 0.)).r - texture(uVol, hp - vec3(0., e.y, 0.)).r, g2 = texture(uVol, hp + vec3(0., 0., e.z)).r - texture(uVol, hp - vec3(0., 0., e.z)).r;
    vec3 n = normalize(vec3(g0, g1, g2) + 1e-5), L = normalize(uRot * vec3(0.3, 0.4, -1.0)); float lit = 0.25 + 0.75 * abs(dot(n, L)), on; vec3 lc = lab(hp, on);
    o = vec4(mix(vec3(best), lc, on * uSegOp) * lit * uOp, 1.);
  } else o = vec4(0., 0., 0., 1.);
}`;

export class VolumeView {
  constructor(canvas) {
    this.cv = canvas; const gl = this.gl = canvas.getContext('webgl2', { antialias: false, preserveDrawingBuffer: true });
    if (!gl) throw new Error('WebGL2 unavailable');
    const sh = (t, src) => { const s = gl.createShader(t); gl.shaderSource(s, src); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
    const pr = this.pr = gl.createProgram(); gl.attachShader(pr, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(pr, sh(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(pr);
    if (!gl.getProgramParameter(pr, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(pr));
    gl.useProgram(pr);
    const vb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, vb); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(pr, 'p'); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    this.u = (n) => gl.getUniformLocation(pr, n);
    this.vol = gl.createTexture(); this.lab = gl.createTexture();
    this.par = { mode: 1, thr: 0.25, op: 1.0, segOp: 0.7, steps: 192, zoom: 1.6, az: 0.6, el: 0.45, seg: true };
    this.box = [0.5, 0.5, 0.5]; this.hasLab = false; this.has = false; this.pal = new Float32Array(48);
    this._pending = false;
  }
  _tex(tex, unit, data, dims, filter) {
    const gl = this.gl; gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_3D, tex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8, dims[2], dims[1], dims[0], 0, gl.RED, gl.UNSIGNED_BYTE, data);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, filter], [gl.TEXTURE_MAG_FILTER, filter], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_3D, k, v);
  }
  // data: Uint8Array (z,y,x) dims [Z,Y,X]; spacing [sz,sy,sx] in um; labels optional Uint8Array same dims; palette [[r,g,b],...]
  setVolume(data, dims, spacing, labels = null, palette = null) {
    const gl = this.gl; this._tex(this.vol, 0, data, dims, gl.LINEAR);
    this.hasLab = !!labels; this._tex(this.lab, 1, labels || new Uint8Array(1), labels ? dims : [1, 1, 1], gl.NEAREST);
    const ph = [dims[2] * spacing[2], dims[1] * spacing[1], dims[0] * spacing[0]], m = Math.max(...ph);   // x,y,z physical extents
    this.box = ph.map((v) => 0.5 * v / m); this.pal.fill(0);
    (palette || []).slice(0, 16).forEach((c, i) => this.pal.set(c, 3 * i));
    this.has = true; this.render();
  }
  setLabels(labels, dims, palette) { const gl = this.gl; this.hasLab = !!labels; this._tex(this.lab, 1, labels || new Uint8Array(1), labels ? dims : [1, 1, 1], gl.NEAREST); this.pal.fill(0); (palette || []).slice(0, 16).forEach((c, i) => this.pal.set(c, 3 * i)); this.render(); }
  set(p) { Object.assign(this.par, p); this.render(); }
  _rot() {   // rotation (camera-to-object): azimuth about y then elevation about x
    const { az, el } = this.par, ca = Math.cos(az), sa = Math.sin(az), ce = Math.cos(el), se = Math.sin(el);
    const Ry = [ca, 0, -sa, 0, 1, 0, sa, 0, ca], Rx = [1, 0, 0, 0, ce, se, 0, -se, ce], R = new Float32Array(9);   // column-major Ry*Rx
    for (let c = 0; c < 3; c++) for (let r = 0; r < 3; r++) { let s = 0; for (let k = 0; k < 3; k++) s += Ry[k * 3 + r] * Rx[c * 3 + k]; R[c * 3 + r] = s; }
    return R;
  }
  render() { if (this._pending) return; this._pending = true; requestAnimationFrame(() => { this._pending = false; this._draw(); }); }
  _draw() {
    const gl = this.gl, cv = this.cv, r = cv.getBoundingClientRect(), d = Math.min(devicePixelRatio || 1, 2);
    const W = Math.max(2, Math.round(r.width * d)), H = Math.max(2, Math.round(r.height * d)); if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
    gl.viewport(0, 0, W, H); if (!this.has) { gl.clearColor(0, 0, 0, 1); gl.clear(gl.COLOR_BUFFER_BIT); return; }
    const P = this.par, u = this.u;
    gl.useProgram(this.pr);
    gl.uniform1i(u('uVol'), 0); gl.uniform1i(u('uLab'), 1); gl.uniformMatrix3fv(u('uRot'), false, this._rot()); gl.uniform3fv(u('uBox'), this.box);
    gl.uniform2f(u('uAspect'), W >= H ? 1.2 * W / H : 1.2, W >= H ? 1.2 : 1.2 * H / W); gl.uniform1f(u('uZoom'), P.zoom);
    gl.uniform1i(u('uMode'), P.mode); gl.uniform1i(u('uSteps'), P.steps); gl.uniform1i(u('uHasLab'), this.hasLab && P.seg ? 1 : 0);
    gl.uniform1f(u('uThr'), P.thr); gl.uniform1f(u('uOp'), P.op); gl.uniform1f(u('uSegOp'), P.segOp); gl.uniform3fv(u('uPal'), this.pal);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
  attachMouse() {
    let drag = null; const cv = this.cv;
    cv.addEventListener('pointerdown', (e) => { cv.setPointerCapture(e.pointerId); drag = { x: e.clientX, y: e.clientY, az: this.par.az, el: this.par.el }; });
    cv.addEventListener('pointermove', (e) => { if (!drag) return; this.par.az = drag.az - (e.clientX - drag.x) * 0.01; this.par.el = Math.max(-1.55, Math.min(1.55, drag.el + (e.clientY - drag.y) * 0.01)); this.render(); });
    cv.addEventListener('pointerup', () => { drag = null; });
    cv.addEventListener('wheel', (e) => { e.preventDefault(); this.par.zoom = Math.max(0.3, Math.min(8, this.par.zoom * Math.exp(-e.deltaY * 0.0015))); this.render(); }, { passive: false });
  }
}
