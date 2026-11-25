// Minimal Blosc1 decoder (zstd / memcpy codecs, byte-shuffle) for zarr v2 chunks. No dependencies except a
// zstd decompressor function (fzstd.decompress) passed in.
export function bloscDecode(buf, zstd) {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const flags = buf[2], typesize = buf[3];
  const nbytes = dv.getUint32(4, true), blocksize = dv.getUint32(8, true);
  const out = new Uint8Array(nbytes);
  if (flags & 0x2) { out.set(buf.subarray(16, 16 + nbytes)); return out; }       // memcpyed
  const codec = flags >> 5;                                                        // 4 = zstd
  if (codec !== 4) throw new Error('blosc codec ' + codec + ' unsupported');
  const shuffle = (flags & 0x1) !== 0, dontSplit = (flags & 0x10) !== 0;
  if (flags & 0x4) throw new Error('bitshuffle unsupported');
  const nblocks = Math.ceil(nbytes / blocksize), leftover = nbytes % blocksize;
  for (let i = 0; i < nblocks; i++) {
    let pos = dv.getUint32(16 + 4 * i, true);
    const isLeft = i === nblocks - 1 && leftover !== 0;
    const bsize = isLeft ? leftover : blocksize;
    const nsplits = (!dontSplit && !isLeft && typesize <= 16 && bsize / typesize >= 128) ? typesize : 1;
    const neblock = Math.floor(bsize / nsplits);
    const tmp = new Uint8Array(bsize);
    let o = 0;
    for (let j = 0; j < nsplits; j++) {
      const cb = dv.getUint32(pos, true); pos += 4;
      const src = buf.subarray(pos, pos + cb); pos += cb;
      if (cb === neblock) tmp.set(src, o); else tmp.set(zstd(src, new Uint8Array(neblock)), o);
      o += neblock;
    }
    const dst = i * blocksize;
    if (shuffle && typesize > 1) {
      const n = Math.floor(bsize / typesize);
      for (let k = 0; k < n; k++) for (let b = 0; b < typesize; b++) out[dst + k * typesize + b] = tmp[b * n + k];
      for (let r = n * typesize; r < bsize; r++) out[dst + r] = tmp[r];
    } else out.set(tmp, dst);
  }
  return out;
}
