import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

// ═══════════════════════════════════════════════════════════════
// Config & palettes
// ═══════════════════════════════════════════════════════════════
const HP = { d: 768, heads: 12, layers: 3, seq: 128, vocab: 50000 };
const VIZ_HEADS = 4;
const EXPERTS = 4;
const TOY_DIM = 4;

const PALETTE_DEFAULT = {
  input: 0x34d399, embed: 0x22d3ee, pos: 0x38bdf8, attn: 0xa78bfa,
  cross: 0xc084fc, ffn: 0xfbbf24, residual: 0x94a3b8, norm: 0xfb7185,
  output: 0xf472b6, head: 0x6ea8ff, noise: 0xf97316, time: 0x2dd4bf,
  router: 0xe879f9, expert: 0xf59e0b, encoder: 0x60a5fa, loop: 0xfb923c,
  q: 0x60a5fa, k: 0x34d399, v: 0xfbbf24, score: 0xf472b6,
};

// Wong-inspired colorblind-safe set mapped to roles
const PALETTE_CB = {
  input: 0x009e73, embed: 0x56b4e9, pos: 0x0072b2, attn: 0xcc79a7,
  cross: 0xd55e00, ffn: 0xe69f00, residual: 0x999999, norm: 0xf0e442,
  output: 0xcc79a7, head: 0x56b4e9, noise: 0xd55e00, time: 0x009e73,
  router: 0xcc79a7, expert: 0xe69f00, encoder: 0x0072b2, loop: 0xd55e00,
  q: 0x0072b2, k: 0x009e73, v: 0xe69f00, score: 0xcc79a7,
};

let COLORS = { ...PALETTE_DEFAULT };
let colorblind = false;
let runMode = 'train'; // train | infer
let compareArchId = '';
let cutawayMode = false;
let cutawayBlock = 0;

const ARCH_ORDER = ['gpt', 'bert', 'seq2seq', 'diffusion', 'moe'];

function fmtNum(n) {
  if (!Number.isFinite(n)) return '—';
  if (Math.abs(n) >= 1e12) return (n / 1e12).toFixed(2) + 'T';
  if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(2) + 'B';
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(Math.round(n));
}

function fmtVec(v) {
  return '[' + v.map(x => (Math.round(x * 100) / 100).toFixed(2)).join(', ') + ']';
}

// ═══════════════════════════════════════════════════════════════
// Graph builders
// ═══════════════════════════════════════════════════════════════
function graphBuilder() {
  const nodes = [], edges = [];
  const add = (n) => { nodes.push(n); return n; };
  const link = (from, to, kind = 'forward') => edges.push({ from, to, kind });
  return { nodes, edges, add, link };
}

function addMHA(g, {
  prefix, y0, prevId, label, heads = VIZ_HEADS,
  attnColor = COLORS.attn, xBase = 0, block = 0,
  bidirectional = false, isCross = false, kvSource = null,
}) {
  const { add, link } = g;
  const mode = bidirectional ? 'bidirectional' : isCross ? 'cross-attention' : 'causal';
  add({
    id: `${prefix}_ln1`, name: `${label} · Pre-Norm (Attn)`, type: 'LayerNorm',
    color: COLORS.norm, shape: 'thin',
    desc: `LayerNorm before ${isCross ? 'cross-' : ''}attention (${mode}).`,
    stats: { dim: 'd_model', params: '2d' }, y: y0, block, xOffset: xBase, attnMode: mode,
  });
  link(prevId, `${prefix}_ln1`);

  const headIds = [];
  const nH = Math.min(heads, VIZ_HEADS);
  for (let h = 0; h < nH; h++) {
    const hid = `${prefix}_h${h}`;
    headIds.push(hid);
    add({
      id: hid, name: `${label} · Head ${h + 1}`,
      type: isCross ? 'Cross-Attn Head' : 'Attention Head',
      color: isCross ? COLORS.cross : attnColor, shape: 'head',
      desc: isCross
        ? 'Cross-attention: Q from decoder, K/V from encoder memory.'
        : `Self-attention head (${mode}).`,
      stats: { dim: 'd_head', params: '3·d·d_h' },
      y: y0 + 1, block, head: h,
      xOffset: xBase + (h - (nH - 1) / 2) * 1.35,
      attnMode: mode, isAttention: true,
    });
    link(`${prefix}_ln1`, hid, 'qkv');
    if (isCross && kvSource) link(kvSource, hid, 'memory');
  }

  add({
    id: `${prefix}_attn`,
    name: `${label} · ${isCross ? 'Cross-Attn' : 'Multi-Head'} Out`,
    type: isCross ? 'Cross-Attention' : 'Attention',
    color: isCross ? COLORS.cross : attnColor, shape: 'slab',
    desc: isCross ? 'Heads concat + W_O.' : `MHA output (${mode}).`,
    stats: { dim: 'd_model', params: 'd²' },
    y: y0 + 2, block, xOffset: xBase, attnMode: mode, isAttention: true,
  });
  headIds.forEach(hid => link(hid, `${prefix}_attn`, 'concat'));

  add({
    id: `${prefix}_res1`, name: `${label} · Residual + Attn`, type: 'Residual',
    color: COLORS.residual, shape: 'ring',
    desc: 'x ← x + Attention(Norm(x)).',
    stats: { dim: 'd_model', params: '0' }, y: y0 + 2.55, block, xOffset: xBase,
  });
  link(`${prefix}_attn`, `${prefix}_res1`);
  link(prevId, `${prefix}_res1`, 'skip');
  return { outId: `${prefix}_res1`, nextY: y0 + 3 };
}

function addFFN(g, { prefix, y0, prevId, label, block = 0, xBase = 0, moe = false }) {
  const { add, link } = g;
  add({
    id: `${prefix}_ln2`, name: `${label} · Pre-Norm (FFN)`, type: 'LayerNorm',
    color: COLORS.norm, shape: 'thin', desc: 'LayerNorm before FFN / experts.',
    stats: { dim: 'd', params: '2d' }, y: y0, block, xOffset: xBase,
  });
  link(prevId, `${prefix}_ln2`);

  if (moe) {
    add({
      id: `${prefix}_router`, name: `${label} · Router`, type: 'MoE Router',
      color: COLORS.router, shape: 'thin', desc: 'Gate → top-k experts.',
      stats: { dim: 'n_experts', params: 'd·E' }, y: y0 + 0.7, block, xOffset: xBase,
    });
    link(`${prefix}_ln2`, `${prefix}_router`, 'route');
    const expertIds = [];
    for (let e = 0; e < EXPERTS; e++) {
      const eid = `${prefix}_e${e}`;
      expertIds.push(eid);
      add({
        id: eid, name: `${label} · Expert ${e + 1}`, type: 'Expert FFN',
        color: COLORS.expert, shape: 'wide', desc: 'Sparse expert MLP.',
        stats: { dim: '4d', params: '~8d²' },
        y: y0 + 1.5, block, xOffset: xBase + (e - (EXPERTS - 1) / 2) * 1.5,
      });
      link(`${prefix}_router`, eid, 'top-k');
    }
    add({
      id: `${prefix}_ffn`, name: `${label} · Expert Combine`, type: 'MoE Combine',
      color: COLORS.ffn, shape: 'slab', desc: 'Weighted expert sum.',
      stats: { dim: 'd', params: '0' }, y: y0 + 2.3, block, xOffset: xBase,
    });
    expertIds.forEach(eid => link(eid, `${prefix}_ffn`, 'combine'));
  } else {
    add({
      id: `${prefix}_ffn`, name: `${label} · Feed-Forward (MLP)`, type: 'FFN / MLP',
      color: COLORS.ffn, shape: 'wide', desc: 'Expand → act → project.',
      stats: { dim: '4d', params: '~8d²' }, y: y0 + 0.7, block, xOffset: xBase,
    });
    link(`${prefix}_ln2`, `${prefix}_ffn`);
  }

  const resY = moe ? y0 + 2.9 : y0 + 1.3;
  add({
    id: `${prefix}_res2`, name: `${label} · Residual + FFN`, type: 'Residual',
    color: COLORS.residual, shape: 'ring', desc: 'x ← x + FFN(Norm(x)).',
    stats: { dim: 'd', params: '0' }, y: resY, block, xOffset: xBase,
  });
  link(`${prefix}_ffn`, `${prefix}_res2`);
  link(prevId, `${prefix}_res2`, 'skip');
  return { outId: `${prefix}_res2`, nextY: resY + 0.7 };
}

function addTransformerBlock(g, opts) {
  const mha = addMHA(g, opts);
  return addFFN(g, { ...opts, y0: mha.nextY, prevId: mha.outId });
}

/** Expanded single-block attention cutaway */
function buildCutaway(blockIndex = 0) {
  const g = graphBuilder();
  const { add, link } = g;
  const b = blockIndex + 1;
  add({
    id: 'x_in', name: `Block ${b} input x`, type: 'Input', color: COLORS.input, shape: 'slab',
    desc: 'Hidden state entering this transformer block.',
    stats: { dim: 'seq × d', params: '0' }, y: 0,
  });
  add({
    id: 'ln', name: 'LayerNorm', type: 'LayerNorm', color: COLORS.norm, shape: 'thin',
    desc: 'Pre-LN stabilizes scale before projections.',
    stats: { dim: 'd', params: '2d' }, y: 1,
  });
  link('x_in', 'ln');

  const projs = [
    { id: 'q', name: 'Q = x W_Q', color: COLORS.q, desc: 'Query projection — “what am I looking for?”' },
    { id: 'k', name: 'K = x W_K', color: COLORS.k, desc: 'Key projection — “what do I contain?” (cached in inference)' },
    { id: 'v', name: 'V = x W_V', color: COLORS.v, desc: 'Value projection — “what do I pass forward?” (cached in inference)' },
  ];
  projs.forEach((p, i) => {
    add({
      id: p.id, name: p.name, type: 'Projection', color: p.color, shape: 'slab',
      desc: p.desc + (runMode === 'infer' && p.id !== 'q' ? ' · KV cache hit on decode steps.' : ''),
      stats: { dim: 'd_head', params: 'd·d_h' },
      y: 2.2, xOffset: (i - 1) * 2.2, isAttention: true, attnMode: 'causal',
    });
    link('ln', p.id, 'qkv');
  });

  add({
    id: 'scores', name: 'Scores = QKᵀ / √d', type: 'Attention', color: COLORS.score, shape: 'wide',
    desc: 'Pairwise similarity. Causal mask zeros future positions in GPT.',
    stats: { dim: 'seq × seq', params: '0' }, y: 3.5, isAttention: true, attnMode: 'causal',
  });
  link('q', 'scores'); link('k', 'scores');

  add({
    id: 'soft', name: 'Softmax (masked)', type: 'Attention', color: COLORS.attn, shape: 'slab',
    desc: 'Normalize scores into attention weights per query row.',
    stats: { dim: 'seq × seq', params: '0' }, y: 4.5, isAttention: true, attnMode: 'causal',
  });
  link('scores', 'soft');

  add({
    id: 'ctx', name: 'Context = weights · V', type: 'Attention', color: COLORS.v, shape: 'slab',
    desc: 'Weighted mix of values — the head output.',
    stats: { dim: 'seq × d_h', params: '0' }, y: 5.5, isAttention: true, attnMode: 'causal',
  });
  link('soft', 'ctx');
  link('v', 'ctx', 'forward');

  add({
    id: 'o_proj', name: 'Output proj W_O', type: 'Attention', color: COLORS.attn, shape: 'slab',
    desc: 'Concat heads (implied) then linear W_O back to d_model.',
    stats: { dim: 'd', params: 'd²' }, y: 6.5, isAttention: true, attnMode: 'causal',
  });
  link('ctx', 'o_proj');

  add({
    id: 'res', name: 'Residual x + attn', type: 'Residual', color: COLORS.residual, shape: 'ring',
    desc: 'Add back the block input — critical for deep stacks.',
    stats: { dim: 'd', params: '0' }, y: 7.5,
  });
  link('o_proj', 'res');
  link('x_in', 'res', 'skip');

  add({
    id: 'out', name: 'To FFN half…', type: 'Output', color: COLORS.output, shape: 'slab',
    desc: 'Next: second norm + FFN + residual (not expanded here).',
    stats: { dim: 'd', params: '0' }, y: 8.5,
  });
  link('res', 'out');
  return g;
}

function buildGPT() {
  const g = graphBuilder();
  const { add, link } = g;
  const L = HP.layers;
  add({ id: 'tokens', name: 'Input Tokens', type: 'Input', color: COLORS.input, shape: 'tokens',
    desc: runMode === 'infer'
      ? 'At inference: stream tokens one-by-one (with KV cache).'
      : 'At training: full sequence in parallel (teacher forcing).',
    stats: { dim: 'seq', params: '0' }, y: 0 });
  add({ id: 'embed', name: 'Token Embedding', type: 'Embedding', color: COLORS.embed, shape: 'slab',
    desc: 'Token ID → d_model.', stats: { dim: 'V×d', params: 'V·d' }, y: 1 });
  link('tokens', 'embed');
  add({ id: 'pos', name: 'Positional Encoding', type: 'Position', color: COLORS.pos, shape: 'slab',
    desc: 'RoPE / ALiBi / learned.', stats: { dim: 'seq×d', params: 'pos' }, y: 2 });
  link('embed', 'pos');
  let prev = 'pos', y = 3;
  for (let b = 0; b < L; b++) {
    const r = addTransformerBlock(g, {
      prefix: `b${b}`, y0: y, prevId: prev, label: `Block ${b + 1}`, block: b,
    });
    prev = r.outId; y = r.nextY + 0.3;
  }
  add({ id: 'final_norm', name: 'Final LayerNorm', type: 'LayerNorm', color: COLORS.norm, shape: 'thin',
    desc: 'Norm before unembedding.', stats: { dim: 'd', params: '2d' }, y });
  link(prev, 'final_norm');
  add({ id: 'lm_head', name: 'LM Head', type: 'Output', color: COLORS.output, shape: 'slab',
    desc: 'd → V.', stats: { dim: 'd×V', params: 'd·V' }, y: y + 1 });
  link('final_norm', 'lm_head');
  add({ id: 'logits', name: 'Softmax → Next Token', type: 'Output', color: COLORS.head, shape: 'tokens',
    desc: runMode === 'infer' ? 'Sample token, append, loop (decode).' : 'Teacher-forced next-token loss over positions.',
    stats: { dim: 'V', params: '0' }, y: y + 2 });
  link('lm_head', 'logits');
  link('logits', 'tokens', 'loop');
  return g;
}

function buildBERT() {
  const g = graphBuilder();
  const { add, link } = g;
  const L = HP.layers;
  add({ id: 'tokens', name: 'Input + [CLS]/[SEP]', type: 'Input', color: COLORS.input, shape: 'tokens',
    desc: 'Full sequence (bidirectional).', stats: { dim: 'seq', params: '0' }, y: 0 });
  add({ id: 'embed', name: 'Token + Segment Embed', type: 'Embedding', color: COLORS.embed, shape: 'slab',
    desc: 'Token + segment embeddings.', stats: { dim: 'V×d', params: 'emb' }, y: 1 });
  link('tokens', 'embed');
  add({ id: 'pos', name: 'Absolute Positions', type: 'Position', color: COLORS.pos, shape: 'slab',
    desc: 'Learned absolute positions.', stats: { dim: 'pos×d', params: 'pos' }, y: 2 });
  link('embed', 'pos');
  let prev = 'pos', y = 3;
  for (let b = 0; b < L; b++) {
    const r = addTransformerBlock(g, {
      prefix: `b${b}`, y0: y, prevId: prev, label: `Enc ${b + 1}`, block: b,
      bidirectional: true, attnColor: COLORS.encoder,
    });
    prev = r.outId; y = r.nextY + 0.3;
  }
  add({ id: 'pool', name: '[CLS] Pooler', type: 'Pooler', color: COLORS.embed, shape: 'slab',
    desc: 'Sequence vector.', stats: { dim: 'd', params: 'd²' }, y });
  link(prev, 'pool');
  add({ id: 'mlm', name: 'MLM Head', type: 'Output', color: COLORS.output, shape: 'wide',
    desc: runMode === 'train' ? 'Predict masked tokens (pretrain).' : 'Usually unused at pure classification inference.',
    stats: { dim: 'V', params: 'd·V' }, y: y + 1.2, xOffset: -2 });
  link(prev, 'mlm');
  add({ id: 'cls_head', name: 'Classification Head', type: 'Output', color: COLORS.head, shape: 'slab',
    desc: 'Downstream labels.', stats: { dim: 'C', params: 'd·C' }, y: y + 1.2, xOffset: 2 });
  link('pool', 'cls_head');
  return g;
}

function buildSeq2Seq() {
  const g = graphBuilder();
  const { add, link } = g;
  const L = HP.layers;
  const nEnc = Math.max(1, Math.ceil(L / 2));
  const nDec = Math.max(1, Math.floor(L / 2) || 1);
  const xEnc = -4.5, xDec = 4.5;
  add({ id: 'src_tokens', name: 'Source Tokens', type: 'Input', color: COLORS.input, shape: 'tokens',
    desc: 'Source sequence.', stats: { dim: 'src', params: '0' }, y: 0, xOffset: xEnc });
  add({ id: 'src_embed', name: 'Source Embedding', type: 'Embedding', color: COLORS.embed, shape: 'slab',
    desc: 'Embed source.', stats: { dim: 'V×d', params: 'emb' }, y: 1, xOffset: xEnc });
  link('src_tokens', 'src_embed');
  let encPrev = 'src_embed', yEnc = 2;
  for (let b = 0; b < nEnc; b++) {
    const r = addTransformerBlock(g, {
      prefix: `enc${b}`, y0: yEnc, prevId: encPrev, label: `Encoder ${b + 1}`, block: b,
      bidirectional: true, attnColor: COLORS.encoder, xBase: xEnc,
    });
    encPrev = r.outId; yEnc = r.nextY + 0.25;
  }
  add({ id: 'memory', name: 'Encoder Memory', type: 'Memory', color: COLORS.encoder, shape: 'slab',
    desc: 'K/V for cross-attention' + (runMode === 'infer' ? ' (computed once, reused).' : '.'),
    stats: { dim: 'src×d', params: '0' }, y: yEnc, xOffset: xEnc });
  link(encPrev, 'memory');
  add({ id: 'tgt_tokens', name: 'Target Tokens', type: 'Input', color: COLORS.input, shape: 'tokens',
    desc: runMode === 'train' ? 'Full target (teacher forcing).' : 'Generated prefix so far.',
    stats: { dim: 'tgt', params: '0' }, y: 0, xOffset: xDec });
  add({ id: 'tgt_embed', name: 'Target Embedding', type: 'Embedding', color: COLORS.embed, shape: 'slab',
    desc: 'Embed decoder tokens.', stats: { dim: 'V×d', params: 'emb' }, y: 1, xOffset: xDec });
  link('tgt_tokens', 'tgt_embed');
  let decPrev = 'tgt_embed', yDec = 2;
  for (let b = 0; b < nDec; b++) {
    const self = addMHA(g, {
      prefix: `dec${b}s`, y0: yDec, prevId: decPrev, label: `Dec ${b + 1} Self`, block: b, xBase: xDec,
    });
    const cross = addMHA(g, {
      prefix: `dec${b}c`, y0: self.nextY, prevId: self.outId, label: `Dec ${b + 1} Cross`,
      block: b, isCross: true, kvSource: 'memory', xBase: xDec,
    });
    const ffn = addFFN(g, {
      prefix: `dec${b}`, y0: cross.nextY, prevId: cross.outId, label: `Dec ${b + 1}`, block: b, xBase: xDec,
    });
    decPrev = ffn.outId; yDec = ffn.nextY + 0.25;
  }
  const yOut = Math.max(yEnc, yDec) + 0.5;
  add({ id: 'lm_head', name: 'LM Head', type: 'Output', color: COLORS.output, shape: 'slab',
    desc: 'Decoder → vocab.', stats: { dim: 'd×V', params: 'd·V' }, y: yOut, xOffset: xDec });
  link(decPrev, 'lm_head');
  add({ id: 'logits', name: 'Next Target Token', type: 'Output', color: COLORS.head, shape: 'tokens',
    desc: 'Next target token.', stats: { dim: 'V', params: '0' }, y: yOut + 1.2, xOffset: xDec });
  link('lm_head', 'logits');
  link('logits', 'tgt_tokens', 'loop');
  return g;
}

function buildDiffusion() {
  const g = graphBuilder();
  const { add, link } = g;
  const L = HP.layers;
  add({ id: 'clean', name: 'Clean / Target Text', type: 'Input', color: COLORS.input, shape: 'tokens',
    desc: runMode === 'train' ? 'Real text to corrupt.' : 'Goal of reverse process.',
    stats: { dim: 'seq', params: '0' }, y: 0 });
  add({ id: 'noise_sched', name: 'Noise Schedule', type: 'Diffusion', color: COLORS.noise, shape: 'wide',
    desc: runMode === 'train' ? 'Forward q(x_t|x_0) — training path.' : 'Schedule still defines reverse steps.',
    stats: { dim: 't', params: 'sched' }, y: 1.1 });
  link('clean', 'noise_sched');
  add({ id: 'noisy', name: 'Noisy State x_t', type: 'Latent', color: COLORS.noise, shape: 'tokens',
    desc: runMode === 'infer' ? 'Often start from pure noise / full mask.' : 'Sampled noisy training input.',
    stats: { dim: 'seq×d', params: '0' }, y: 2.2 });
  link('noise_sched', 'noisy');
  add({ id: 'time_emb', name: 'Timestep Embedding', type: 'Conditioning', color: COLORS.time, shape: 'thin',
    desc: 'Condition on t.', stats: { dim: 'd', params: 'd' }, y: 2.2, xOffset: 4.2 });
  link('noise_sched', 'time_emb', 't');
  add({ id: 'embed', name: 'Token / Latent Embed', type: 'Embedding', color: COLORS.embed, shape: 'slab',
    desc: 'Embed noisy state.', stats: { dim: 'd', params: 'emb' }, y: 3.3 });
  link('noisy', 'embed');
  link('time_emb', 'embed', 'condition');
  add({ id: 'pos', name: 'Positional Encoding', type: 'Position', color: COLORS.pos, shape: 'slab',
    desc: 'Positions for denoiser.', stats: { dim: 'seq×d', params: 'pos' }, y: 4.2 });
  link('embed', 'pos');
  let prev = 'pos', y = 5.1;
  for (let b = 0; b < L; b++) {
    const r = addTransformerBlock(g, {
      prefix: `d${b}`, y0: y, prevId: prev, label: `Denoiser ${b + 1}`, block: b, bidirectional: true,
    });
    link('time_emb', `d${b}_ln1`, 'condition');
    prev = r.outId; y = r.nextY + 0.25;
  }
  add({ id: 'pred_head', name: 'Predict Clean / Noise', type: 'Output', color: COLORS.output, shape: 'slab',
    desc: 'ε-pred / x-pred / discrete denoise head.', stats: { dim: 'seq×V', params: 'head' }, y });
  link(prev, 'pred_head');
  add({ id: 'sampler', name: 'Sampler Step t→t−1', type: 'Diffusion', color: COLORS.loop, shape: 'wide',
    desc: runMode === 'infer' ? 'Iterative reverse — main inference loop.' : 'One reverse step (also used in some trainers).',
    stats: { dim: '1 step', params: '0' }, y: y + 1.1 });
  link('pred_head', 'sampler');
  link('time_emb', 'sampler', 't');
  add({ id: 'x_prev', name: 'Less-Noisy x_{t−1}', type: 'Latent', color: COLORS.time, shape: 'tokens',
    desc: 'Refined state.', stats: { dim: 'seq', params: '0' }, y: y + 2.2 });
  link('sampler', 'x_prev');
  link('x_prev', 'noisy', 'loop');
  add({ id: 'final_text', name: 'Decoded Text (t=0)', type: 'Output', color: COLORS.head, shape: 'tokens',
    desc: 'Clean text after reverse process.', stats: { dim: 'seq', params: '0' }, y: y + 3.3, xOffset: 3.5 });
  link('x_prev', 'final_text');
  return g;
}

function buildMoE() {
  const g = graphBuilder();
  const { add, link } = g;
  const L = HP.layers;
  add({ id: 'tokens', name: 'Input Tokens', type: 'Input', color: COLORS.input, shape: 'tokens',
    desc: 'AR token stream.', stats: { dim: 'seq', params: '0' }, y: 0 });
  add({ id: 'embed', name: 'Token Embedding', type: 'Embedding', color: COLORS.embed, shape: 'slab',
    desc: 'Embed.', stats: { dim: 'V×d', params: 'emb' }, y: 1 });
  link('tokens', 'embed');
  add({ id: 'pos', name: 'Positional Encoding', type: 'Position', color: COLORS.pos, shape: 'slab',
    desc: 'Positions.', stats: { dim: 'seq×d', params: 'pos' }, y: 2 });
  link('embed', 'pos');
  let prev = 'pos', y = 3;
  for (let b = 0; b < L; b++) {
    const mha = addMHA(g, { prefix: `b${b}`, y0: y, prevId: prev, label: `Block ${b + 1}`, block: b });
    const ffn = addFFN(g, {
      prefix: `b${b}`, y0: mha.nextY, prevId: mha.outId, label: `Block ${b + 1}`, block: b, moe: true,
    });
    prev = ffn.outId; y = ffn.nextY + 0.3;
  }
  add({ id: 'final_norm', name: 'Final LayerNorm', type: 'LayerNorm', color: COLORS.norm, shape: 'thin',
    desc: 'Final norm.', stats: { dim: 'd', params: '2d' }, y });
  link(prev, 'final_norm');
  add({ id: 'lm_head', name: 'LM Head', type: 'Output', color: COLORS.output, shape: 'slab',
    desc: 'Unembed.', stats: { dim: 'd×V', params: 'd·V' }, y: y + 1 });
  link('final_norm', 'lm_head');
  add({ id: 'logits', name: 'Next Token', type: 'Output', color: COLORS.head, shape: 'tokens',
    desc: 'Sample next token.', stats: { dim: 'V', params: '0' }, y: y + 2 });
  link('lm_head', 'logits');
  link('logits', 'tokens', 'loop');
  return g;
}

const ARCHITECTURES = {
  gpt: {
    id: 'gpt', name: 'GPT · Decoder-only', short: 'GPT',
    tags: ['causal', 'AR', 'next-token'],
    blurb: 'Causal transformer: next-token prediction.',
    camera: { pos: [14, 16, 22], target: [0, 10, 0] },
    causal: true, build: buildGPT,
    modeBlurb: {
      train: 'Train: full sequence in parallel, cross-entropy on every position (teacher forcing). Loop edge is conceptual.',
      infer: 'Infer: decode one token at a time; K/V from past tokens are cached — orange loop is the real runtime path.',
    },
    tour: [
      { id: 'tokens', title: 'Tokens in', body: 'Text becomes token IDs — the only discrete input.' },
      { id: 'embed', title: 'Embed', body: 'Each ID maps to a d_model vector (see hyperparams).' },
      { id: 'pos', title: 'Position', body: 'Order signal so attention is not bag-of-tokens.' },
      { pick: 'b0_attn', title: 'Causal attention', body: 'Each position attends only to the past. Try Cutaway for Q/K/V.' },
      { id: 'lm_head', title: 'LM head', body: 'Project to vocabulary logits.' },
      { id: 'logits', title: 'Loop', body: 'Infer mode: sample → append → again. Toggle Train/Infer to see emphasis change.' },
    ],
  },
  bert: {
    id: 'bert', name: 'BERT · Encoder-only', short: 'BERT',
    tags: ['bidirectional', 'MLM'],
    blurb: 'Bidirectional encoder for understanding tasks.',
    camera: { pos: [12, 14, 18], target: [0, 8, 0] },
    causal: false, build: buildBERT,
    modeBlurb: {
      train: 'Train: masked LM and/or NSP-style objectives over full bidirectional context.',
      infer: 'Infer: usually a single forward pass into a task head — no token-by-token loop.',
    },
    tour: [
      { id: 'tokens', title: 'Special tokens', body: '[CLS] pools; [SEP] bounds segments.' },
      { pick: 'b0_attn', title: 'Bidirectional attn', body: 'Every token sees the full sequence.' },
      { id: 'mlm', title: 'MLM', body: 'Pretraining predicts masks.' },
      { id: 'cls_head', title: 'Task head', body: 'Fine-tune classifier on pooled vector.' },
    ],
  },
  seq2seq: {
    id: 'seq2seq', name: 'Encoder–Decoder', short: 'Enc–Dec',
    tags: ['T5/BART', 'cross-attn'],
    blurb: 'Encoder memory + causal decoder with cross-attention.',
    camera: { pos: [18, 14, 20], target: [0, 9, 0] },
    causal: true, build: buildSeq2Seq,
    modeBlurb: {
      train: 'Train: teacher-force full target; encoder runs on source; loss on decoder positions.',
      infer: 'Infer: encode source once, then decode token-by-token with cross-attn to fixed memory.',
    },
    tour: [
      { id: 'src_tokens', title: 'Encoder', body: 'Source is encoded bidirectionally.' },
      { id: 'memory', title: 'Memory', body: 'Final encoder states feed every cross-attn layer.' },
      { id: 'tgt_tokens', title: 'Decoder', body: 'Target is causal.' },
      { pick: 'dec0c_attn', title: 'Cross-attn', body: 'Decoder queries look up encoder K/V.' },
    ],
  },
  diffusion: {
    id: 'diffusion', name: 'Diffusion LLM', short: 'Diffusion',
    tags: ['denoising', 'iterative'],
    blurb: 'Iterative denoising instead of pure left-to-right AR.',
    camera: { pos: [14, 15, 20], target: [0, 10, 0] },
    causal: false, build: buildDiffusion,
    modeBlurb: {
      train: 'Train: sample t, corrupt text, predict clean/noise — emphasis on forward noise path.',
      infer: 'Infer: start from noise, reverse-sample many steps — emphasis on sampler loop.',
    },
    tour: [
      { id: 'noise_sched', title: 'Noise', body: 'Forward corruption over timesteps.' },
      { id: 'time_emb', title: 'Time', body: 'Network is conditioned on t.' },
      { pick: 'd0_attn', title: 'Denoiser', body: 'Usually bidirectional transformer.' },
      { id: 'sampler', title: 'Reverse', body: 'Step t→t−1 until clean text.' },
    ],
  },
  moe: {
    id: 'moe', name: 'MoE · Sparse GPT', short: 'MoE',
    tags: ['router', 'experts'],
    blurb: 'GPT-like with routed expert FFNs.',
    camera: { pos: [16, 15, 22], target: [0, 10, 0] },
    causal: true, build: buildMoE,
    modeBlurb: {
      train: 'Train: router + experts with load-balancing losses; still teacher-forced AR.',
      infer: 'Infer: same AR loop; only top-k experts run per token (sparse compute).',
    },
    tour: [
      { pick: 'b0_attn', title: 'Dense attention', body: 'Attention is still full; sparsity is in FFN.' },
      { pick: 'b0_router', title: 'Router', body: 'Chooses top-k experts per token.' },
      { pick: 'b0_e0', title: 'Expert', body: 'Only selected experts compute.' },
      { id: 'logits', title: 'Output', body: 'Unembedding matches dense GPT.' },
    ],
  },
};

// ═══════════════════════════════════════════════════════════════
// Scene
// ═══════════════════════════════════════════════════════════════
const wrap = document.getElementById('canvas-wrap');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x070b14);
scene.fog = new THREE.FogExp2(0x070b14, 0.012);

const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.1, 500);
camera.position.set(14, 16, 22);

const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
wrap.appendChild(renderer.domElement);

const labelRenderer = new CSS2DRenderer();
labelRenderer.setSize(innerWidth, innerHeight);
Object.assign(labelRenderer.domElement.style, { position: 'absolute', top: '0', pointerEvents: 'none' });
wrap.appendChild(labelRenderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.minDistance = 4;
controls.maxDistance = 100;
controls.target.set(0, 10, 0);

scene.add(new THREE.AmbientLight(0x8899bb, 0.55));
const keyL = new THREE.DirectionalLight(0xffffff, 1.1);
keyL.position.set(10, 25, 15);
scene.add(keyL);
const fillL = new THREE.DirectionalLight(0x6688ff, 0.35);
fillL.position.set(-12, 8, -10);
scene.add(fillL);

const grid = new THREE.GridHelper(80, 50, 0x1a2744, 0x121a2c);
grid.position.y = -0.5;
scene.add(grid);

{
  const g = new THREE.BufferGeometry();
  const pos = new Float32Array(700 * 3);
  for (let i = 0; i < 700; i++) {
    pos[i * 3] = (Math.random() - 0.5) * 140;
    pos[i * 3 + 1] = Math.random() * 90 - 5;
    pos[i * 3 + 2] = (Math.random() - 0.5) * 140;
  }
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  scene.add(new THREE.Points(g, new THREE.PointsMaterial({
    color: 0x88aadd, size: 0.06, transparent: true, opacity: 0.45, sizeAttenuation: true,
  })));
}

const Y_SCALE = 1.3;
const EXPLODE_GAP = 0.5;
let explodeAmount = 0, arrowsEnabled = true, currentExplode = 0;
const clock = new THREE.Clock();

const primaryGroup = new THREE.Group();
const compareGroup = new THREE.Group();
scene.add(primaryGroup);
scene.add(compareGroup);

const tokenMat = new THREE.MeshStandardMaterial({
  color: 0x34d399, emissive: 0x14532d, metalness: 0.3, roughness: 0.35,
});
const tokenMesh = new THREE.Mesh(new THREE.SphereGeometry(0.28, 20, 16), tokenMat);
tokenMesh.visible = false;
scene.add(tokenMesh);

// Graph state (primary)
let nodes = [], edges = [], nodeById = {};
const nodeMeshes = {};
let edgeLines = [];
// Compare ghost
let cmpNodes = [], cmpEdges = [];
const cmpMeshes = {};
let cmpEdgeLines = [];

let focusedId = null;
let focusedEdge = null;
let currentArchId = 'gpt';

// Toy vector
let toyVec = [0.5, -0.2, 0.8, 0.1];
let toyOut = toyVec.slice();

// ═══════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════
function makeMaterial(color, opts = {}) {
  return new THREE.MeshStandardMaterial({
    color, metalness: 0.25, roughness: 0.45, transparent: true,
    opacity: opts.opacity ?? 0.92,
    emissive: new THREE.Color(color).multiplyScalar(0.15),
    side: THREE.DoubleSide, ...opts,
  });
}

function shortName(node) {
  if (node.type === 'Attention Head' || node.type === 'Cross-Attn Head') return `H${(node.head ?? 0) + 1}`;
  if (node.type === 'Expert FFN') return `E${(node.name.match(/Expert (\d+)/) || [])[1] || '?'}`;
  if (node.name.includes('Pre-Norm (Attn)')) return 'LN₁';
  if (node.name.includes('Pre-Norm (FFN)')) return 'LN₂';
  if (node.name.includes('Multi-Head') || node.name.includes('Cross-Attn Out')) return node.name.includes('Cross') ? 'XAttn' : 'MHA';
  if (node.name.includes('Feed-Forward')) return 'FFN';
  if (node.name.includes('Residual')) return '+';
  if (node.name.includes('Router')) return 'Router';
  if (node.name.includes('Q =')) return 'Q';
  if (node.name.includes('K =')) return 'K';
  if (node.name.includes('V =')) return 'V';
  if (node.name.includes('Scores')) return 'QKᵀ';
  if (node.name.includes('Softmax')) return 'σ';
  if (node.name.includes('Context')) return '·V';
  if (node.name.includes('W_O')) return 'W_O';
  return node.name.length > 18 ? node.name.slice(0, 16) + '…' : node.name;
}

function disposeObject(obj) {
  obj.traverse(child => {
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      (Array.isArray(child.material) ? child.material : [child.material]).forEach(m => m.dispose?.());
    }
    if (child.isCSS2DObject && child.element?.parentNode) child.element.parentNode.removeChild(child.element);
  });
}

function clearGroup(group, meshMap, edgeArr) {
  while (group.children.length) {
    const c = group.children[0];
    group.remove(c);
    disposeObject(c);
  }
  Object.keys(meshMap).forEach(k => delete meshMap[k]);
  edgeArr.length = 0;
}

function createNodeMesh(node, group, meshMap, { ghost = false, idPrefix = '' } = {}) {
  const g = new THREE.Group();
  const color = node.color;
  const w = 3.2, d = 2.4;
  const baseOp = ghost ? 0.28 : 0.92;

  switch (node.shape) {
    case 'tokens': {
      const wrapG = new THREE.Group();
      const box = new THREE.BoxGeometry(0.45, 0.45, 0.45);
      const mat = makeMaterial(color, { opacity: baseOp });
      for (let i = 0; i < 6; i++) {
        const m = new THREE.Mesh(box, mat);
        m.position.x = (i - 2.5) * 0.55;
        wrapG.add(m);
      }
      g.add(wrapG); g.userData.primaryMat = mat; break;
    }
    case 'head': {
      const mat = makeMaterial(color, { opacity: baseOp });
      g.add(new THREE.Mesh(new THREE.SphereGeometry(0.38, 20, 16), mat));
      const shellMat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: ghost ? 0.05 : 0.12, depthWrite: false });
      shellMat.userData.baseOp = shellMat.opacity;
      g.add(new THREE.Mesh(new THREE.SphereGeometry(0.48, 12, 10), shellMat));
      g.userData.primaryMat = mat; break;
    }
    case 'thin': {
      const mat = makeMaterial(color, { opacity: ghost ? 0.25 : 0.85 });
      g.add(new THREE.Mesh(new THREE.BoxGeometry(w * 0.95, 0.12, d * 0.95), mat));
      g.userData.primaryMat = mat; break;
    }
    case 'wide': {
      const geo = new THREE.BoxGeometry(w * 1.15, 0.55, d * 0.7);
      const mat = makeMaterial(color, { opacity: baseOp });
      g.add(new THREE.Mesh(geo, mat));
      if (!ghost) g.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.2 })));
      g.userData.primaryMat = mat; break;
    }
    case 'ring': {
      const mat = makeMaterial(color, { opacity: ghost ? 0.25 : 0.9 });
      const torus = new THREE.Mesh(new THREE.TorusGeometry(0.85, 0.08, 10, 40), mat);
      torus.rotation.x = Math.PI / 2;
      g.add(torus);
      g.add(new THREE.Mesh(new THREE.CylinderGeometry(0.55, 0.55, 0.06, 24), makeMaterial(color, { opacity: ghost ? 0.12 : 0.35 })));
      g.userData.primaryMat = mat; break;
    }
    default: {
      const geo = new THREE.BoxGeometry(w, 0.35, d);
      const mat = makeMaterial(color, { opacity: baseOp });
      g.add(new THREE.Mesh(geo, mat));
      if (!ghost) g.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.18 })));
      g.userData.primaryMat = mat; break;
    }
  }

  if (!ghost) {
    const hit = new THREE.Mesh(
      node.shape === 'head' ? new THREE.SphereGeometry(0.7, 8, 8) : new THREE.BoxGeometry(3.6, 0.7, 2.8),
      new THREE.MeshBasicMaterial({ visible: false })
    );
    g.add(hit);
    g.userData.hit = hit;
    g.userData.nodeId = node.id;
  }

  const div = document.createElement('div');
  div.textContent = (ghost ? '▸ ' : '') + shortName(node);
  div.style.cssText = `color:${ghost ? '#7a8aaa' : '#c8d4ec'};font-size:${ghost ? 10 : 11}px;font-family:system-ui,sans-serif;padding:2px 6px;background:rgba(8,12,22,0.75);border:1px solid rgba(255,255,255,0.08);border-radius:4px;white-space:nowrap;user-select:none;opacity:${ghost ? 0.55 : 0.85};`;
  const label = new CSS2DObject(div);
  label.position.set(0, node.shape === 'head' ? 0.65 : 0.45, 0);
  g.add(label);
  g.userData.labelEl = div;

  const baseY = node.y * Y_SCALE;
  const baseX = (node.xOffset || 0);
  g.position.set(baseX, baseY, 0);
  group.add(g);
  const id = idPrefix + node.id;
  meshMap[id] = { group: g, mat: g.userData.primaryMat, baseY, baseX, node, ghost };
}

function edgeStyle(kind) {
  if (kind === 'skip') return { color: 0x64748b, opacity: 0.25 };
  if (kind === 'loop') return { color: 0xfb923c, opacity: 0.45 };
  if (kind === 'memory' || kind === 'condition' || kind === 't') return { color: 0x2dd4bf, opacity: 0.4 };
  if (kind === 'qkv' || kind === 'concat') return { color: 0xa78bfa, opacity: 0.4 };
  if (kind === 'route' || kind === 'top-k' || kind === 'combine') return { color: 0xe879f9, opacity: 0.45 };
  return { color: 0x6ea8ff, opacity: 0.4 };
}

function makeArrowGeometry() {
  const geo = new THREE.ConeGeometry(0.07, 0.16, 8);
  geo.translate(0, -0.08, 0);
  return geo;
}

function createEdgesFor(edgeList, meshMap, group, outArr, { ghost = false, idPrefix = '' } = {}) {
  edgeList.forEach((e, idx) => {
    const fromId = idPrefix + e.from, toId = idPrefix + e.to;
    if (!meshMap[fromId] || !meshMap[toId]) return;
    const { color, opacity } = edgeStyle(e.kind);
    const op = ghost ? opacity * 0.35 : opacity;
    const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: op, depthWrite: false });
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]), mat);
    group.add(line);

    const arrowMat = new THREE.MeshBasicMaterial({
      color, transparent: true, opacity: ghost ? 0.2 : Math.min(1, op + 0.45), depthWrite: false,
    });
    const arrow = new THREE.Mesh(makeArrowGeometry(), arrowMat);
    arrow.visible = arrowsEnabled && !ghost;
    group.add(arrow);
    const arrow2Mat = arrowMat.clone();
    const arrow2 = new THREE.Mesh(makeArrowGeometry(), arrow2Mat);
    const wantSecond = e.kind === 'skip' || e.kind === 'loop' || e.kind === 'memory';
    arrow2.visible = arrowsEnabled && wantSecond && !ghost;
    arrow2.scale.setScalar(0.85);
    group.add(arrow2);

    let pickTube = null, tubeMesh = null, tubeMat = null;
    if (!ghost) {
      const pickMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.001, depthWrite: false });
      pickTube = new THREE.Mesh(new THREE.BufferGeometry(), pickMat);
      pickTube.userData.edgeIndex = idx;
      pickTube.userData.isEdgePick = true;
      group.add(pickTube);
      tubeMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0, depthWrite: false });
      tubeMesh = new THREE.Mesh(new THREE.BufferGeometry(), tubeMat);
      tubeMesh.visible = false;
      group.add(tubeMesh);
    }

    outArr.push({
      line, mat, arrow, arrowMat, arrow2, arrow2Mat, wantSecond,
      tubeMesh, tubeMat, pickTube,
      from: e.from, to: e.to, kind: e.kind,
      fromKey: fromId, toKey: toId,
      baseOpacity: op, baseColor: color, ghost,
    });
  });
}

function placeArrowOnCurve(mesh, curve, u) {
  const pos = curve.getPoint(u);
  let tangent = curve.getTangent(u);
  if (tangent.lengthSq() < 1e-8) tangent = new THREE.Vector3(0, 1, 0);
  else tangent.normalize();
  mesh.position.copy(pos);
  const up = new THREE.Vector3(0, 1, 0);
  if (Math.abs(up.dot(tangent)) > 0.999) { tangent.x += 0.001; tangent.normalize(); }
  mesh.quaternion.setFromUnitVectors(up, tangent);
}

function updateEdgeGeometryList(list, meshMap) {
  list.forEach(ed => {
    const a = meshMap[ed.fromKey], b = meshMap[ed.toKey];
    if (!a || !b) return;
    const p0 = a.group.position.clone(); p0.y += 0.2;
    const p3 = b.group.position.clone(); p3.y -= 0.15;
    let p1, p2;
    if (ed.kind === 'skip') {
      const side = (a.baseX + b.baseX) / 2 >= 0 ? 3.6 : -3.6;
      p1 = new THREE.Vector3(a.baseX + side, p0.y + 0.15, 0);
      p2 = new THREE.Vector3(b.baseX + side, p3.y - 0.15, 0);
    } else if (ed.kind === 'loop') {
      p1 = new THREE.Vector3(5.5, p0.y, 1.5);
      p2 = new THREE.Vector3(5.5, p3.y, 1.5);
    } else if (ed.kind === 'memory') {
      p1 = new THREE.Vector3(p0.x + (p3.x - p0.x) * 0.35, p0.y + 0.4, -1.2);
      p2 = new THREE.Vector3(p0.x + (p3.x - p0.x) * 0.65, p3.y - 0.2, -1.2);
    } else {
      const midY = (p0.y + p3.y) / 2;
      const bulge = Math.abs(p3.x - p0.x) * 0.12 + 0.35;
      p1 = new THREE.Vector3(p0.x, midY, -bulge);
      p2 = new THREE.Vector3(p3.x, midY, -bulge);
    }
    const curve = new THREE.CubicBezierCurve3(p0, p1, p2, p3);
    ed.line.geometry.setFromPoints(curve.getPoints(24));
    ed.curve = curve;
    if (!ed.ghost) {
      placeArrowOnCurve(ed.arrow, curve, 0.88);
      if (ed.wantSecond) placeArrowOnCurve(ed.arrow2, curve, 0.45);
      const key = `${p0.x.toFixed(2)},${p0.y.toFixed(2)},${p3.x.toFixed(2)},${p3.y.toFixed(2)}`;
      if (ed._pickKey !== key && ed.pickTube) {
        ed._pickKey = key;
        if (ed.pickTube.geometry) ed.pickTube.geometry.dispose();
        ed.pickTube.geometry = new THREE.TubeGeometry(curve, 20, 0.12, 5, false);
      }
      const edgeFocused = focusedEdge && ed.from === focusedEdge.from && ed.to === focusedEdge.to && ed.kind === focusedEdge.kind;
      if (ed.highlighted || edgeFocused) {
        if (ed._tubeKey !== key && ed.tubeMesh) {
          ed._tubeKey = key;
          if (ed.tubeMesh.geometry) ed.tubeMesh.geometry.dispose();
          ed.tubeMesh.geometry = new THREE.TubeGeometry(curve, 32, 0.05, 6, false);
        }
        if (ed.tubeMesh) {
          ed.tubeMesh.visible = true;
          ed.tubeMat.opacity = edgeFocused ? 0.55 : 0.35;
          ed.tubeMat.color.setHex(edgeFocused ? 0xfbbf24 : 0xa5c8ff);
        }
      } else if (ed.tubeMesh) {
        ed.tubeMesh.visible = false;
        ed.tubeMat.opacity = 0;
        ed._tubeKey = null;
      }
    }
  });
}

// ═══════════════════════════════════════════════════════════════
// Mode emphasis
// ═══════════════════════════════════════════════════════════════
function modeEdgeBoost(ed) {
  // Returns multiplier for opacity when not focused
  if (currentArchId === 'gpt' || currentArchId === 'moe' || currentArchId === 'seq2seq') {
    if (runMode === 'infer' && ed.kind === 'loop') return 1.8;
    if (runMode === 'train' && ed.kind === 'loop') return 0.35;
  }
  if (currentArchId === 'diffusion') {
    if (runMode === 'train' && (ed.from === 'clean' || ed.to === 'noise_sched' || ed.from === 'noise_sched')) return 1.5;
    if (runMode === 'infer' && (ed.kind === 'loop' || ed.from === 'sampler' || ed.to === 'sampler')) return 1.6;
    if (runMode === 'train' && ed.kind === 'loop') return 0.4;
  }
  if (currentArchId === 'bert') {
    if (runMode === 'infer' && ed.to === 'mlm') return 0.35;
    if (runMode === 'train' && ed.to === 'mlm') return 1.4;
  }
  return 1;
}

function updateModeBlurb() {
  const arch = ARCHITECTURES[currentArchId];
  const el = document.getElementById('mode-blurb');
  el.textContent = arch.modeBlurb?.[runMode] || '';
}

// ═══════════════════════════════════════════════════════════════
// Estimates & toy math
// ═══════════════════════════════════════════════════════════════
function estimateModel() {
  const { d, heads, layers, seq, vocab } = HP;
  const emb = vocab * d;
  let params = emb + layers * (4 * d * d + 8 * d * d) + layers * 4 * d;
  if (currentArchId === 'moe') params = emb + layers * (4 * d * d + EXPERTS * 8 * d * d + d * EXPERTS);
  if (currentArchId === 'seq2seq') params = emb + layers * (4 * d * d + 4 * d * d + 8 * d * d) * 1.2;
  const flops = 2 * params + layers * 2 * seq * d * d;
  return { params, flops, dHead: d / heads };
}

function updateParamUI() {
  document.getElementById('val-d').textContent = HP.d;
  document.getElementById('val-h').textContent = HP.heads;
  document.getElementById('val-l').textContent = HP.layers;
  document.getElementById('val-s').textContent = HP.seq;
  document.getElementById('val-v').textContent = HP.vocab >= 1000 ? (HP.vocab / 1000).toFixed(0) + 'k' : HP.vocab;
  const est = estimateModel();
  document.getElementById('est-params').textContent = fmtNum(est.params);
  document.getElementById('est-flops').textContent = fmtNum(est.flops);
}

function liveStats(node) {
  const { d, heads, seq, vocab } = HP;
  const dHead = Math.max(1, Math.round(d / heads));
  const t = node.type;
  if (t === 'Embedding') return { dim: `${seq}×${d}`, params: fmtNum(vocab * d) };
  if (t === 'LayerNorm') return { dim: `${d}`, params: fmtNum(2 * d) };
  if (t.includes('Attention') || t === 'Attention Head' || t === 'Cross-Attn Head' || t === 'Projection')
    return { dim: t.includes('Head') || t === 'Projection' ? `${dHead}` : `${seq}×${d}`, params: fmtNum(4 * d * d / (t.includes('Head') ? heads : 1)) };
  if (t === 'FFN / MLP' || t === 'Expert FFN') return { dim: `${4 * d}`, params: fmtNum(8 * d * d) };
  if (t === 'MoE Router') return { dim: `${EXPERTS}`, params: fmtNum(d * EXPERTS) };
  if (t === 'Output') return { dim: node.id.includes('logit') || node.id === 'final_text' ? `${vocab}` : `${d}×${vocab}`, params: fmtNum(d * vocab) };
  return { dim: node.stats?.dim || '—', params: node.stats?.params || '—' };
}

function applyToyAtNode(node, vec) {
  // Deterministic toy transforms by layer type (not real weights)
  const out = vec.slice();
  const t = node.type;
  const seed = (node.id || '').split('').reduce((a, c) => a + c.charCodeAt(0), 0);
  const s = (i) => Math.sin((seed + i * 17) * 0.1) * 0.5 + 0.5;
  if (t === 'Embedding' || t === 'Input' || t === 'Position') {
    for (let i = 0; i < TOY_DIM; i++) out[i] = vec[i] * 0.9 + s(i) * 0.2;
  } else if (t === 'LayerNorm') {
    const mean = out.reduce((a, b) => a + b, 0) / TOY_DIM;
    const varr = out.reduce((a, b) => a + (b - mean) ** 2, 0) / TOY_DIM + 1e-5;
    const std = Math.sqrt(varr);
    for (let i = 0; i < TOY_DIM; i++) out[i] = (out[i] - mean) / std;
  } else if (t.includes('Attention') || t === 'Attention Head' || t === 'Cross-Attn Head' || t === 'Projection') {
    // mix like a tiny attention
    const mix = out.reduce((a, b) => a + b, 0) / TOY_DIM;
    for (let i = 0; i < TOY_DIM; i++) out[i] = out[i] * 0.6 + mix * 0.4 + s(i) * 0.05;
  } else if (t === 'FFN / MLP' || t === 'Expert FFN' || t === 'MoE Combine') {
    for (let i = 0; i < TOY_DIM; i++) {
      const h = Math.max(0, out[i] * 1.5 + s(i)); // relu-ish
      out[i] = h * 0.7 - 0.1;
    }
  } else if (t === 'Residual') {
    // identity-ish residual: keep close
    for (let i = 0; i < TOY_DIM; i++) out[i] = vec[i] * 0.85 + out[i] * 0.15;
  } else if (t === 'Output' || t === 'Pooler') {
    for (let i = 0; i < TOY_DIM; i++) out[i] = Math.tanh(out[i] * 1.2 + s(i) * 0.1);
  } else if (t === 'MoE Router') {
    // softmax-ish gates into first dims
    const ex = out.map(Math.exp);
    const sum = ex.reduce((a, b) => a + b, 0);
    for (let i = 0; i < TOY_DIM; i++) out[i] = ex[i] / sum;
  } else if (t === 'Diffusion' || t === 'Latent' || t === 'Conditioning') {
    const noise = runMode === 'train' ? 0.3 : 0.1;
    for (let i = 0; i < TOY_DIM; i++) out[i] = out[i] * (1 - noise) + (s(i) - 0.5) * noise;
  }
  return out;
}

function updateToyDisplay() {
  document.getElementById('toy-out').textContent = 'Output: ' + fmtVec(toyOut);
}

function initToyInputs() {
  const host = document.getElementById('toy-inputs');
  host.innerHTML = '';
  for (let i = 0; i < TOY_DIM; i++) {
    const inp = document.createElement('input');
    inp.type = 'number';
    inp.step = '0.1';
    inp.value = toyVec[i];
    inp.addEventListener('change', () => {
      toyVec[i] = Number(inp.value) || 0;
      toyOut = toyVec.slice();
      updateToyDisplay();
      syncURL();
    });
    host.appendChild(inp);
  }
  updateToyDisplay();
}

// ═══════════════════════════════════════════════════════════════
// Focus / info
// ═══════════════════════════════════════════════════════════════
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

function kindLabel(kind) {
  const k = kind;
  if (k === 'skip') return 'residual skip';
  if (k === 'loop') return 'generation / reverse loop';
  if (k === 'memory') return 'encoder memory (K/V)';
  if (k === 'condition' || k === 't') return 'conditioning';
  if (k === 'qkv') return 'Q/K/V';
  if (k === 'concat') return 'concat heads';
  if (k === 'route') return 'routing';
  if (k === 'top-k') return 'top-k expert';
  if (k === 'combine') return 'expert mix';
  return 'forward';
}

function getNeighbors(id) {
  const incoming = edges.filter(e => e.to === id).map(e => e.from);
  const outgoing = edges.filter(e => e.from === id).map(e => e.to);
  return { incoming, outgoing, all: new Set([...incoming, ...outgoing, id]) };
}

function setFocus(id) {
  focusedId = id;
  focusedEdge = null;
  if (id && nodeById[id]) {
    toyOut = applyToyAtNode(nodeById[id], toyVec);
    updateToyDisplay();
  }
  updateFocusVisuals();
  updateInfoPanel();
  updateLayerButtons();
  syncURL();
}

function setEdgeFocus(edge) {
  focusedEdge = edge;
  focusedId = null;
  updateFocusVisuals();
  updateInfoPanel();
  updateLayerButtons();
  syncURL();
}

function clearFocus() {
  focusedId = null;
  focusedEdge = null;
  updateFocusVisuals();
  updateInfoPanel();
  updateLayerButtons();
  syncURL();
}

function updateFocusVisuals() {
  const neigh = focusedId ? getNeighbors(focusedId) : null;
  const edgeEnds = focusedEdge ? new Set([focusedEdge.from, focusedEdge.to]) : null;

  Object.values(nodeMeshes).forEach(({ group, node }) => {
    if (!group.userData.nodeId) return;
    const isFocused = focusedId === node.id || (edgeEnds && edgeEnds.has(node.id));
    const isNeighbor = neigh && neigh.all.has(node.id);
    const dimmed = (focusedId || focusedEdge) && !isFocused && !isNeighbor;

    group.traverse(obj => {
      const m = obj.material;
      if (!m || m.visible === false) return;
      if (obj.isLineSegments || m.isLineBasicMaterial) {
        m.opacity = dimmed ? 0.05 : (isFocused ? 0.55 : 0.18);
        return;
      }
      if (m.isMeshBasicMaterial) {
        if (focusedId || focusedEdge) m.opacity = isFocused ? 0.28 : (isNeighbor ? 0.12 : 0.02);
        else m.opacity = m.userData.baseOp ?? 0.12;
        return;
      }
      if (m.userData.baseOp === undefined) m.userData.baseOp = m.opacity;
      if (focusedId || focusedEdge) {
        if (isFocused) {
          m.opacity = 1;
          if (m.emissive) m.emissive.copy(new THREE.Color(node.color).multiplyScalar(0.5));
        } else if (isNeighbor) {
          m.opacity = 0.9;
          if (m.emissive) m.emissive.copy(new THREE.Color(node.color).multiplyScalar(0.25));
        } else {
          m.opacity = 0.1;
          if (m.emissive) m.emissive.setHex(0);
        }
      } else {
        m.opacity = m.userData.baseOp;
        if (m.emissive) m.emissive.copy(new THREE.Color(node.color).multiplyScalar(0.15));
      }
    });

    if (group.userData.labelEl) {
      const el = group.userData.labelEl;
      if (focusedId || focusedEdge) {
        el.style.opacity = (isFocused || isNeighbor) ? '1' : '0.12';
        el.style.fontWeight = isFocused ? '700' : '400';
      } else {
        el.style.opacity = '0.85';
        el.style.fontWeight = '400';
      }
    }
    group.userData.targetScale = isFocused ? 1.12 : 1;
  });

  edgeLines.forEach(ed => {
    const related = focusedId && (ed.from === focusedId || ed.to === focusedId);
    const between = focusedId && neigh && neigh.all.has(ed.from) && neigh.all.has(ed.to);
    const isEdgeFocus = focusedEdge && ed.from === focusedEdge.from && ed.to === focusedEdge.to && ed.kind === focusedEdge.kind;
    const boost = modeEdgeBoost(ed);

    const setArrow = (opacity, colorHex, scale = 1) => {
      const show = arrowsEnabled && opacity > 0.05;
      ed.arrow.visible = show;
      ed.arrow2.visible = show && ed.wantSecond;
      ed.arrowMat.opacity = opacity;
      ed.arrow2Mat.opacity = opacity * 0.9;
      ed.arrowMat.color.setHex(colorHex);
      ed.arrow2Mat.color.setHex(colorHex);
      ed.arrow.scale.setScalar(scale);
      ed.arrow2.scale.setScalar(scale * 0.85);
    };

    if (isEdgeFocus) {
      ed.mat.opacity = 1;
      ed.mat.color.setHex(0xfbbf24);
      ed.highlighted = true;
      setArrow(1, 0xfbbf24, 1.15);
    } else if (!focusedId && !focusedEdge) {
      ed.mat.opacity = Math.min(1, ed.baseOpacity * boost);
      ed.mat.color.setHex(boost > 1.2 ? 0xfbbf24 : ed.baseColor);
      ed.highlighted = boost > 1.4;
      setArrow(Math.min(1, ed.baseOpacity * boost + 0.3), boost > 1.2 ? 0xfbbf24 : ed.baseColor, boost > 1.2 ? 1.1 : 1);
    } else if (related) {
      ed.mat.opacity = 0.95;
      ed.mat.color.setHex(0xffffff);
      ed.highlighted = true;
      setArrow(1, 0xffffff, 1.12);
    } else if (between) {
      ed.mat.opacity = 0.35;
      ed.mat.color.setHex(ed.baseColor);
      ed.highlighted = false;
      setArrow(0.5, ed.baseColor, 1);
    } else {
      ed.mat.opacity = 0.04;
      ed.highlighted = false;
      setArrow(0.03, ed.baseColor, 0.7);
    }
  });
}

function drawAttentionMask(canvas, mode) {
  const n = 10;
  const ctx = canvas.getContext('2d');
  const s = canvas.width / n;
  for (let q = 0; q < n; q++) {
    for (let k = 0; k < n; k++) {
      let on = true;
      if (mode === 'causal') on = k <= q;
      const v = on ? (0.35 + 0.55 * (1 - Math.abs(q - k) / n)) : 0.06;
      ctx.fillStyle = on ? `rgb(${30 + v * 40},${Math.floor(40 + v * 180)},${Math.floor(80 + v * 160)})` : 'rgb(18,22,32)';
      ctx.fillRect(k * s, q * s, s - 1, s - 1);
    }
  }
}

function maskCaption(mode) {
  if (mode === 'causal') return 'Causal: q attends only to k ≤ q.';
  if (mode === 'cross-attention') return 'Cross-attn: queries see full encoder source.';
  return 'Bidirectional: full attention matrix.';
}

function updateInfoPanel() {
  const empty = document.getElementById('info-empty');
  const content = document.getElementById('info-content');

  if (focusedEdge) {
    empty.style.display = 'none';
    content.style.display = 'block';
    const a = nodeById[focusedEdge.from], b = nodeById[focusedEdge.to];
    content.innerHTML = `
      <div class="layer-type">Connection · ${kindLabel(focusedEdge.kind)}</div>
      <h2>${a?.name || focusedEdge.from} → ${b?.name || focusedEdge.to}</h2>
      <p>Data flows along this wire. Kind: <strong>${kindLabel(focusedEdge.kind)}</strong>.
      ${runMode === 'infer' && focusedEdge.kind === 'loop' ? ' In inference this loop is the decode/reverse cycle.' : ''}
      ${runMode === 'train' && focusedEdge.kind === 'loop' ? ' In training this loop is often conceptual (teacher forcing).' : ''}</p>
      <div class="stats">
        <div class="stat"><div class="val">${a?.name || '—'}</div><div class="lbl">Source</div></div>
        <div class="stat"><div class="val">${b?.name || '—'}</div><div class="lbl">Target</div></div>
      </div>
      <div class="section-label">Actions</div>
      <ul>
        <li class="clickable" data-goto="${focusedEdge.from}"><span class="arrow">◎</span><span>Focus source</span></li>
        <li class="clickable" data-goto="${focusedEdge.to}"><span class="arrow">◎</span><span>Focus target</span></li>
      </ul>`;
    content.querySelectorAll('[data-goto]').forEach(el => {
      el.addEventListener('click', () => { setFocus(el.dataset.goto); flyToNode(el.dataset.goto); });
    });
    return;
  }

  if (!focusedId) {
    empty.style.display = 'block';
    content.style.display = 'none';
    const arch = ARCHITECTURES[currentArchId];
    const tags = arch.tags.map(t => `<span class="tag">${t}</span>`).join('');
    empty.innerHTML = `
      <div class="arch-overview">
        <h3>${arch.name}${cutawayMode ? ' · Cutaway' : ''}</h3>
        <p>${cutawayMode ? 'Attention internals: Q, K, V, scores, softmax, context, W_O, residual.' : arch.blurb}</p>
        <div class="tags">${tags}<span class="tag">${runMode}</span>${compareArchId ? `<span class="tag">vs ${ARCHITECTURES[compareArchId]?.short}</span>` : ''}</div>
      </div>
      <p class="hint-pick">Play a pass · click layer/wire · <strong>Cutaway</strong> expands attention · compare stacks side by side · edit toy vector.</p>`;
    return;
  }

  empty.style.display = 'none';
  content.style.display = 'block';
  const node = nodeById[focusedId];
  if (!node) return;
  const { incoming, outgoing } = getNeighbors(focusedId);
  const hex = '#' + new THREE.Color(node.color).getHexString();
  const ls = liveStats(node);

  const edgeLi = (id, dir) => {
    const n = nodeById[id];
    const kinds = edges.filter(e => dir === 'in' ? (e.from === id && e.to === focusedId) : (e.from === focusedId && e.to === id));
    const kind = kinds[0]?.kind || 'forward';
    return `<li class="clickable" data-edge-from="${dir === 'in' ? id : focusedId}" data-edge-to="${dir === 'in' ? focusedId : id}" data-edge-kind="${kind}">
      <span class="arrow">${dir === 'in' ? '←' : '→'}</span>
      <span><strong>${n?.name || id}</strong><br><span style="color:var(--muted);font-size:0.72rem">${kindLabel(kind)}</span></span>
    </li>`;
  };

  let maskHtml = '';
  if (node.isAttention || (node.type && node.type.includes('Attention')) || node.type === 'Projection') {
    const mode = node.attnMode || (ARCHITECTURES[currentArchId]?.causal ? 'causal' : 'bidirectional');
    maskHtml = `<div class="mask-wrap"><div class="mask-title">Attention mask</div>
      <canvas id="attn-mask" width="200" height="200"></canvas>
      <div class="mask-caption">${maskCaption(mode)}</div></div>`;
  }

  const canCut = node.block !== undefined && !cutawayMode;
  content.innerHTML = `
    <div class="layer-type">${ARCHITECTURES[currentArchId]?.short || 'Cutaway'} · ${node.type}${node.block !== undefined ? ` · block ${node.block + 1}` : ''} · ${runMode}</div>
    <h2><span class="color-dot" style="background:${hex};color:${hex}"></span>${node.name}</h2>
    <p>${node.desc}</p>
    <div class="stats">
      <div class="stat"><div class="val">${ls.dim}</div><div class="lbl">Shape (live)</div></div>
      <div class="stat"><div class="val">${ls.params}</div><div class="lbl">Params order</div></div>
    </div>
    <div class="section-label">Toy vector at this layer</div>
    <p style="font-family:monospace;font-size:0.78rem;color:#a7f3d0">${fmtVec(toyOut)}</p>
    ${maskHtml}
    ${canCut ? `<button class="ctrl" id="info-cutaway" style="width:100%;margin-top:8px">Open cutaway for block ${node.block + 1}</button>` : ''}
    <div class="section-label">Incoming (${incoming.length})</div>
    <ul>${incoming.map(id => edgeLi(id, 'in')).join('') || '<li style="color:var(--muted)">None</li>'}</ul>
    <div class="section-label">Outgoing (${outgoing.length})</div>
    <ul>${outgoing.map(id => edgeLi(id, 'out')).join('') || '<li style="color:var(--muted)">None</li>'}</ul>`;

  content.querySelectorAll('[data-edge-from]').forEach(el => {
    el.addEventListener('click', () => setEdgeFocus({
      from: el.dataset.edgeFrom, to: el.dataset.edgeTo, kind: el.dataset.edgeKind,
    }));
  });
  const cutBtn = document.getElementById('info-cutaway');
  if (cutBtn) cutBtn.addEventListener('click', () => enterCutaway(node.block));
  const canvas = document.getElementById('attn-mask');
  if (canvas) {
    const mode = node.attnMode || (ARCHITECTURES[currentArchId]?.causal ? 'causal' : 'bidirectional');
    drawAttentionMask(canvas, mode);
  }
}

function buildLayerButtons() {
  const host = document.getElementById('layer-buttons');
  host.innerHTML = '';
  const list = nodes.filter(n => n.shape !== 'head' && n.type !== 'Expert FFN');
  list.forEach((node, i) => {
    const btn = document.createElement('button');
    btn.className = 'layer-btn';
    btn.dataset.id = node.id;
    btn.dataset.search = (node.name + ' ' + node.type + ' ' + node.id).toLowerCase();
    const hex = '#' + new THREE.Color(node.color).getHexString();
    btn.innerHTML = `<span class="swatch" style="background:${hex}"></span><span class="name">${node.name.replace(/Block (\d+) · /, 'B$1 · ')}</span><span class="idx">${String(i + 1).padStart(2, '0')}</span>`;
    btn.addEventListener('click', () => {
      stopPass();
      if (focusedId === node.id) clearFocus();
      else { setFocus(node.id); flyToNode(node.id); }
    });
    host.appendChild(btn);
  });
  applySearchFilter();
}

function updateLayerButtons() {
  const passId = pass.active ? pass.order[pass.index] : null;
  document.querySelectorAll('.layer-btn').forEach(btn => {
    btn.classList.toggle('focused', btn.dataset.id === focusedId);
    btn.classList.toggle('pass-current', btn.dataset.id === passId);
  });
}

function applySearchFilter() {
  const q = (document.getElementById('layer-search').value || '').trim().toLowerCase();
  document.querySelectorAll('.layer-btn').forEach(btn => {
    btn.classList.toggle('hidden-search', q && !btn.dataset.search.includes(q));
  });
}

function flyToNode(id) {
  const nm = nodeMeshes[id];
  if (!nm) return;
  const target = nm.group.position.clone();
  // account for primary group offset
  target.add(primaryGroup.position);
  const startTarget = controls.target.clone();
  const startPos = camera.position.clone();
  const dir = startPos.clone().sub(startTarget).normalize();
  const endPos = target.clone().add(dir.multiplyScalar(14));
  let t = 0;
  (function step() {
    t = Math.min(1, t + 0.03);
    const e = 1 - Math.pow(1 - t, 3);
    camera.position.lerpVectors(startPos, endPos, e);
    controls.target.lerpVectors(startTarget, target, e);
    if (t < 1) requestAnimationFrame(step);
  })();
}

function flyCamera(pos, target) {
  const startPos = camera.position.clone(), startT = controls.target.clone();
  const endPos = new THREE.Vector3(...pos), endT = new THREE.Vector3(...target);
  let t = 0;
  (function step() {
    t = Math.min(1, t + 0.025);
    const e = 1 - Math.pow(1 - t, 3);
    camera.position.lerpVectors(startPos, endPos, e);
    controls.target.lerpVectors(startT, endT, e);
    if (t < 1) requestAnimationFrame(step);
  })();
}

function cameraPreset(name) {
  const midY = 10;
  if (name === 'top') flyCamera([0, 40, 0.01], [0, 0, 0]);
  else if (name === 'side') flyCamera([32, midY, 0], [0, midY, 0]);
  else if (name === 'front') flyCamera([0, midY, 28], [0, midY, 0]);
  else {
    const arch = ARCHITECTURES[currentArchId];
    flyCamera(arch.camera.pos, arch.camera.target);
  }
}

// ═══════════════════════════════════════════════════════════════
// Pass player
// ═══════════════════════════════════════════════════════════════
const pass = { active: false, playing: false, order: [], index: -1, timer: null, anim: null };

function buildPassOrder() {
  const mainEdges = edges.filter(e => e.kind !== 'loop' && e.kind !== 'skip');
  const indeg = {};
  nodes.forEach(n => { indeg[n.id] = 0; });
  mainEdges.forEach(e => { if (indeg[e.to] !== undefined) indeg[e.to]++; });
  const adj = {};
  nodes.forEach(n => { adj[n.id] = []; });
  mainEdges.forEach(e => adj[e.from]?.push(e.to));
  const yOf = id => nodeById[id]?.y ?? 0;
  const q = nodes.filter(n => indeg[n.id] === 0).map(n => n.id).sort((a, b) => yOf(a) - yOf(b));
  const order = [], seen = new Set();
  while (q.length) {
    const id = q.shift();
    if (seen.has(id)) continue;
    seen.add(id);
    order.push(id);
    for (const to of adj[id] || []) {
      indeg[to]--;
      if (indeg[to] <= 0) q.push(to);
    }
    q.sort((a, b) => yOf(a) - yOf(b));
  }
  nodes.forEach(n => { if (!seen.has(n.id)) order.push(n.id); });
  return order.filter(id => {
    const n = nodeById[id];
    return n && n.shape !== 'head' && n.type !== 'Expert FFN';
  });
}

function updatePassUI() {
  const label = document.getElementById('pass-label');
  const name = document.getElementById('pass-step-name');
  const prog = document.getElementById('pass-progress');
  const playBtn = document.getElementById('pass-play');
  if (!pass.active || pass.index < 0) {
    label.textContent = 'Pass idle';
    name.textContent = '—';
    prog.style.width = '0%';
    playBtn.textContent = '▶ Play';
    playBtn.classList.remove('active');
    return;
  }
  const n = pass.order.length;
  label.textContent = `Step ${pass.index + 1} / ${n}`;
  name.textContent = nodeById[pass.order[pass.index]]?.name || '';
  prog.style.width = `${((pass.index + 1) / n) * 100}%`;
  playBtn.textContent = pass.playing ? '⏸ Pause' : '▶ Play';
  playBtn.classList.toggle('active', pass.playing);
  updateLayerButtons();
}

function goPassStep(i, animateEdge = true) {
  if (!pass.order.length) pass.order = buildPassOrder();
  if (!pass.order.length) return;
  pass.active = true;
  pass.index = Math.max(0, Math.min(i, pass.order.length - 1));
  const id = pass.order[pass.index];
  // cumulative toy from start
  let v = toyVec.slice();
  for (let s = 0; s <= pass.index; s++) {
    const node = nodeById[pass.order[s]];
    if (node) v = applyToyAtNode(node, v);
  }
  toyOut = v;
  updateToyDisplay();
  setFocus(id);
  flyToNode(id);
  if (animateEdge && pass.index > 0) {
    const prev = pass.order[pass.index - 1];
    const ed = edgeLines.find(e => e.from === prev && e.to === id && e.kind !== 'skip')
      || edgeLines.find(e => e.from === prev && e.to === id);
    if (ed?.curve) {
      pass.anim = { edge: ed, t0: performance.now(), dur: 450 };
      tokenMesh.visible = true;
    }
  } else if (nodeMeshes[id]) {
    tokenMesh.visible = true;
    tokenMesh.position.copy(nodeMeshes[id].group.position).add(primaryGroup.position);
  }
  updatePassUI();
}

function passNext() {
  if (!pass.active) { goPassStep(0, false); return; }
  if (pass.index >= pass.order.length - 1) { pass.playing = false; updatePassUI(); return; }
  goPassStep(pass.index + 1, true);
}
function passPrev() {
  if (!pass.active) return;
  goPassStep(pass.index - 1, false);
}
function stopPass() {
  pass.active = false;
  pass.playing = false;
  pass.index = -1;
  if (pass.timer) { clearInterval(pass.timer); pass.timer = null; }
  pass.anim = null;
  tokenMesh.visible = false;
  updatePassUI();
}
function togglePlay() {
  if (!pass.active) { pass.order = buildPassOrder(); goPassStep(0, false); }
  pass.playing = !pass.playing;
  if (pass.playing) {
    if (pass.timer) clearInterval(pass.timer);
    pass.timer = setInterval(() => {
      if (!pass.playing) return;
      if (pass.index >= pass.order.length - 1) {
        pass.playing = false; updatePassUI();
        clearInterval(pass.timer); pass.timer = null;
        return;
      }
      passNext();
    }, 900);
  } else if (pass.timer) { clearInterval(pass.timer); pass.timer = null; }
  updatePassUI();
}

// ═══════════════════════════════════════════════════════════════
// Tour
// ═══════════════════════════════════════════════════════════════
const tour = { active: false, index: 0 };

function showTourStep() {
  const steps = ARCHITECTURES[currentArchId]?.tour || [];
  const card = document.getElementById('tour-card');
  if (!tour.active || !steps.length) { card.classList.remove('visible'); return; }
  const step = steps[tour.index];
  card.classList.add('visible');
  document.getElementById('tour-kicker').textContent = `Tour · ${tour.index + 1} / ${steps.length}`;
  document.getElementById('tour-title').textContent = step.title;
  document.getElementById('tour-body').textContent = step.body;
  const target = step.id || step.pick;
  if (target && nodeById[target]) { setFocus(target); flyToNode(target); }
  document.getElementById('tour-next').textContent = tour.index >= steps.length - 1 ? 'Done' : 'Next';
  document.getElementById('tour-back').disabled = tour.index === 0;
}
function startTour() {
  if (cutawayMode) exitCutaway();
  stopPass();
  tour.active = true;
  tour.index = 0;
  document.getElementById('btn-tour').classList.add('active');
  showTourStep();
}
function endTour() {
  tour.active = false;
  document.getElementById('tour-card').classList.remove('visible');
  document.getElementById('btn-tour').classList.remove('active');
}

// ═══════════════════════════════════════════════════════════════
// Cutaway / compare / load
// ═══════════════════════════════════════════════════════════════
function enterCutaway(block = 0) {
  cutawayMode = true;
  cutawayBlock = block;
  document.getElementById('btn-cutaway').classList.add('active');
  document.getElementById('cutaway-badge').classList.remove('hidden');
  document.getElementById('compare-badge').classList.add('hidden');
  loadGraph({ cutaway: true });
  syncURL();
}
function exitCutaway() {
  cutawayMode = false;
  document.getElementById('btn-cutaway').classList.remove('active');
  document.getElementById('cutaway-badge').classList.add('hidden');
  loadGraph();
  syncURL();
}

function loadGraph({ cutaway = cutawayMode, keepCamera = false } = {}) {
  stopPass();
  focusedId = null;
  focusedEdge = null;
  clearGroup(primaryGroup, nodeMeshes, edgeLines);
  clearGroup(compareGroup, cmpMeshes, cmpEdgeLines);

  let g;
  if (cutaway) g = buildCutaway(cutawayBlock);
  else g = ARCHITECTURES[currentArchId].build();

  nodes = g.nodes;
  edges = g.edges;
  nodeById = Object.fromEntries(nodes.map(n => [n.id, n]));

  // layout offset for compare
  const comparing = !cutaway && compareArchId && compareArchId !== currentArchId;
  primaryGroup.position.set(comparing ? -7 : 0, 0, 0);
  compareGroup.position.set(comparing ? 7 : 0, 0, 0);
  compareGroup.visible = comparing;

  nodes.forEach(n => createNodeMesh(n, primaryGroup, nodeMeshes));
  createEdgesFor(edges, nodeMeshes, primaryGroup, edgeLines);

  if (comparing) {
    const cg = ARCHITECTURES[compareArchId].build();
    cmpNodes = cg.nodes;
    cmpEdges = cg.edges;
    cmpNodes.forEach(n => createNodeMesh(n, compareGroup, cmpMeshes, { ghost: true, idPrefix: 'c:' }));
    createEdgesFor(cmpEdges, cmpMeshes, compareGroup, cmpEdgeLines, { ghost: true, idPrefix: 'c:' });
    document.getElementById('compare-badge').textContent =
      `${ARCHITECTURES[currentArchId].short} (left)  vs  ${ARCHITECTURES[compareArchId].short} (right · ghost)`;
    document.getElementById('compare-badge').classList.remove('hidden');
  } else {
    document.getElementById('compare-badge').classList.add('hidden');
  }

  updateEdgeGeometryList(edgeLines, nodeMeshes);
  updateEdgeGeometryList(cmpEdgeLines, cmpMeshes);
  updateFocusVisuals();
  buildLayerButtons();
  updateInfoPanel();
  updateParamUI();
  updateModeBlurb();

  if (!keepCamera) {
    if (comparing) flyCamera([0, 16, 28], [0, 10, 0]);
    else if (cutaway) flyCamera([10, 12, 16], [0, 5, 0]);
    else {
      const arch = ARCHITECTURES[currentArchId];
      flyCamera(arch.camera.pos, arch.camera.target);
    }
  }
}

function loadArchitecture(archId, opts = {}) {
  if (!ARCHITECTURES[archId]) return;
  if (cutawayMode) {
    cutawayMode = false;
    document.getElementById('btn-cutaway').classList.remove('active');
    document.getElementById('cutaway-badge').classList.add('hidden');
  }
  endTour();
  currentArchId = archId;
  const sel = document.getElementById('arch-select');
  if (sel.value !== archId) sel.value = archId;
  loadGraph(opts);
  syncURL();
}

// ═══════════════════════════════════════════════════════════════
// URL state
// ═══════════════════════════════════════════════════════════════
function syncURL() {
  const p = new URLSearchParams();
  p.set('arch', currentArchId);
  p.set('mode', runMode);
  if (compareArchId) p.set('compare', compareArchId);
  if (cutawayMode) { p.set('cutaway', '1'); p.set('block', String(cutawayBlock)); }
  if (focusedId) p.set('focus', focusedId);
  if (focusedEdge) p.set('edge', `${focusedEdge.from}>${focusedEdge.to}:${focusedEdge.kind}`);
  p.set('d', String(HP.d));
  p.set('h', String(HP.heads));
  p.set('l', String(HP.layers));
  p.set('s', String(HP.seq));
  p.set('v', String(HP.vocab));
  if (colorblind) p.set('cb', '1');
  p.set('toy', toyVec.map(x => x.toFixed(2)).join(','));
  const url = `${location.pathname}?${p.toString()}`;
  history.replaceState(null, '', url);
}

function readURL() {
  const p = new URLSearchParams(location.search);
  if (p.get('arch') && ARCHITECTURES[p.get('arch')]) currentArchId = p.get('arch');
  if (p.get('mode') === 'infer' || p.get('mode') === 'train') runMode = p.get('mode');
  if (p.get('compare') && ARCHITECTURES[p.get('compare')]) compareArchId = p.get('compare');
  if (p.get('cb') === '1') {
    colorblind = true;
    COLORS = { ...PALETTE_CB };
    document.getElementById('btn-cb').classList.add('active');
  }
  if (p.get('d')) HP.d = Number(p.get('d')) || HP.d;
  if (p.get('h')) HP.heads = Number(p.get('h')) || HP.heads;
  if (p.get('l')) HP.layers = Number(p.get('l')) || HP.layers;
  if (p.get('s')) HP.seq = Number(p.get('s')) || HP.seq;
  if (p.get('v')) HP.vocab = Number(p.get('v')) || HP.vocab;
  if (p.get('toy')) {
    const parts = p.get('toy').split(',').map(Number);
    if (parts.length === TOY_DIM && parts.every(Number.isFinite)) toyVec = parts;
  }
  // sync sliders
  document.getElementById('p-d').value = HP.d;
  document.getElementById('p-h').value = HP.heads;
  document.getElementById('p-l').value = HP.layers;
  document.getElementById('p-s').value = HP.seq;
  document.getElementById('p-v').value = HP.vocab;
  document.getElementById('arch-select').value = currentArchId;
  document.getElementById('compare-select').value = compareArchId || '';
  document.querySelectorAll('.seg-btn').forEach(b => b.classList.toggle('active', b.dataset.mode === runMode));

  const wantCut = p.get('cutaway') === '1';
  cutawayBlock = Number(p.get('block') || 0) || 0;
  loadGraph({ cutaway: wantCut, keepCamera: false });
  cutawayMode = wantCut;
  document.getElementById('btn-cutaway').classList.toggle('active', wantCut);
  document.getElementById('cutaway-badge').classList.toggle('hidden', !wantCut);

  if (p.get('focus') && nodeById[p.get('focus')]) {
    setFocus(p.get('focus'));
  } else if (p.get('edge')) {
    const m = p.get('edge').match(/^(.+)>(.+):(.+)$/);
    if (m) setEdgeFocus({ from: m[1], to: m[2], kind: m[3] });
  }
  updateModeBlurb();
  updateParamUI();
}

// ═══════════════════════════════════════════════════════════════
// Export PNG
// ═══════════════════════════════════════════════════════════════
function exportPNG() {
  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);
  // composite webgl only (labels are separate DOM) — still useful
  const a = document.createElement('a');
  a.download = `llm-${currentArchId}-${runMode}.png`;
  a.href = renderer.domElement.toDataURL('image/png');
  a.click();
}

// ═══════════════════════════════════════════════════════════════
// UI wiring
// ═══════════════════════════════════════════════════════════════
const archSelect = document.getElementById('arch-select');
const compareSelect = document.getElementById('compare-select');
ARCH_ORDER.forEach(id => {
  const o = document.createElement('option');
  o.value = id; o.textContent = ARCHITECTURES[id].name;
  archSelect.appendChild(o);
  const o2 = document.createElement('option');
  o2.value = id; o2.textContent = ARCHITECTURES[id].name;
  compareSelect.appendChild(o2);
});
archSelect.addEventListener('change', () => loadArchitecture(archSelect.value));
compareSelect.addEventListener('change', () => {
  compareArchId = compareSelect.value;
  if (cutawayMode) exitCutaway();
  else { loadGraph({ keepCamera: false }); syncURL(); }
});

document.querySelectorAll('.seg-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    runMode = btn.dataset.mode;
    document.querySelectorAll('.seg-btn').forEach(b => b.classList.toggle('active', b === btn));
    // rebuild so descriptions update
    loadGraph({ keepCamera: true });
    syncURL();
  });
});

document.getElementById('layer-search').addEventListener('input', applySearchFilter);
document.getElementById('layer-search').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    const first = [...document.querySelectorAll('.layer-btn:not(.hidden-search)')][0];
    if (first) { setFocus(first.dataset.id); flyToNode(first.dataset.id); }
  }
});

document.querySelectorAll('[data-cam]').forEach(btn => {
  btn.addEventListener('click', () => cameraPreset(btn.dataset.cam));
});

const bindSlider = (id, key, rebuild = false) => {
  document.getElementById(id).addEventListener('input', (e) => {
    HP[key] = Number(e.target.value);
    updateParamUI();
    if (rebuild) loadGraph({ keepCamera: true });
    else if (focusedId) updateInfoPanel();
    syncURL();
  });
};
bindSlider('p-d', 'd');
bindSlider('p-h', 'heads');
bindSlider('p-l', 'layers', true);
bindSlider('p-s', 'seq');
bindSlider('p-v', 'vocab');

document.getElementById('toy-reset').addEventListener('click', () => {
  toyVec = [0.5, -0.2, 0.8, 0.1];
  toyOut = toyVec.slice();
  initToyInputs();
  syncURL();
});

document.getElementById('pass-play').addEventListener('click', togglePlay);
document.getElementById('pass-next').addEventListener('click', () => {
  pass.playing = false; if (pass.timer) { clearInterval(pass.timer); pass.timer = null; }
  passNext(); updatePassUI();
});
document.getElementById('pass-prev').addEventListener('click', () => {
  pass.playing = false; if (pass.timer) { clearInterval(pass.timer); pass.timer = null; }
  passPrev();
});
document.getElementById('pass-stop').addEventListener('click', () => { stopPass(); clearFocus(); });

document.getElementById('btn-tour').addEventListener('click', () => tour.active ? endTour() : startTour());
document.getElementById('tour-skip').addEventListener('click', endTour);
document.getElementById('tour-back').addEventListener('click', () => { if (tour.index > 0) { tour.index--; showTourStep(); } });
document.getElementById('tour-next').addEventListener('click', () => {
  const steps = ARCHITECTURES[currentArchId]?.tour || [];
  if (tour.index >= steps.length - 1) endTour();
  else { tour.index++; showTourStep(); }
});

document.getElementById('btn-cutaway').addEventListener('click', () => {
  if (cutawayMode) exitCutaway();
  else {
    let b = 0;
    if (focusedId && nodeById[focusedId]?.block !== undefined) b = nodeById[focusedId].block;
    enterCutaway(b);
  }
});

document.getElementById('btn-explode').addEventListener('click', (e) => {
  explodeAmount = explodeAmount > 0.5 ? 0 : 1;
  e.currentTarget.classList.toggle('active', explodeAmount > 0.5);
});
document.getElementById('btn-arrows').addEventListener('click', (e) => {
  arrowsEnabled = !arrowsEnabled;
  e.currentTarget.classList.toggle('active', arrowsEnabled);
  updateFocusVisuals();
});
document.getElementById('btn-cb').addEventListener('click', (e) => {
  colorblind = !colorblind;
  COLORS = colorblind ? { ...PALETTE_CB } : { ...PALETTE_DEFAULT };
  e.currentTarget.classList.toggle('active', colorblind);
  loadGraph({ keepCamera: true });
  syncURL();
});
document.getElementById('btn-export').addEventListener('click', exportPNG);
document.getElementById('btn-reset').addEventListener('click', () => {
  stopPass(); endTour();
  if (cutawayMode) exitCutaway();
  else {
    clearFocus();
    explodeAmount = 0;
    document.getElementById('btn-explode').classList.remove('active');
    cameraPreset('default');
  }
});

// Mobile tabs
document.querySelectorAll('.panel-tabs .tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.panel-tabs .tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    const panel = tab.dataset.panel;
    const list = document.getElementById('layer-list');
    const info = document.getElementById('info-panel');
    if (panel === 'info') {
      list.classList.add('mobile-hide');
      info.classList.add('mobile-show');
    } else {
      list.classList.remove('mobile-hide');
      info.classList.remove('mobile-show');
    }
  });
});

// Picking
function onPointerDown(event) {
  if (event.button !== 0) return;
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);

  const edgeHits = [];
  edgeLines.forEach(ed => {
    if (!ed.pickTube?.geometry?.attributes?.position) return;
    const hits = raycaster.intersectObject(ed.pickTube, false);
    if (hits.length) edgeHits.push({ dist: hits[0].distance, ed });
  });
  edgeHits.sort((a, b) => a.dist - b.dist);

  const nodeHits = [];
  Object.values(nodeMeshes).forEach(({ group }) => {
    const hit = group.userData.hit;
    if (!hit) return;
    const hits = raycaster.intersectObject(hit, false);
    if (hits.length) nodeHits.push({ dist: hits[0].distance, id: group.userData.nodeId });
  });
  nodeHits.sort((a, b) => a.dist - b.dist);

  const bestEdge = edgeHits[0], bestNode = nodeHits[0];
  if (bestEdge && (!bestNode || bestEdge.dist < bestNode.dist - 0.15)) {
    stopPass();
    const ed = bestEdge.ed;
    if (focusedEdge && focusedEdge.from === ed.from && focusedEdge.to === ed.to && focusedEdge.kind === ed.kind) clearFocus();
    else setEdgeFocus({ from: ed.from, to: ed.to, kind: ed.kind });
    return;
  }
  if (bestNode) {
    stopPass();
    if (focusedId === bestNode.id) clearFocus();
    else setFocus(bestNode.id);
  }
}

function onPointerMove(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  let over = false;
  for (const { group } of Object.values(nodeMeshes)) {
    if (group.userData.hit && raycaster.intersectObject(group.userData.hit, false).length) { over = true; break; }
  }
  if (!over) {
    for (const ed of edgeLines) {
      if (ed.pickTube && raycaster.intersectObject(ed.pickTube, false).length) { over = true; break; }
    }
  }
  renderer.domElement.style.cursor = over ? 'pointer' : 'grab';
}

renderer.domElement.addEventListener('pointerdown', onPointerDown);
renderer.domElement.addEventListener('pointermove', onPointerMove);

function applyExplodeLayout() {
  Object.values(nodeMeshes).forEach(({ group, baseY, baseX, node }) => {
    group.position.y = baseY + node.y * EXPLODE_GAP * currentExplode;
    group.position.x = baseX;
    if (node.shape === 'head') group.rotation.y = clock.elapsedTime * 0.4 + (node.head || 0);
    const ts = group.userData.targetScale ?? 1;
    group.scale.setScalar(group.scale.x + (ts - group.scale.x) * 0.12);
  });
  Object.values(cmpMeshes).forEach(({ group, baseY, baseX, node }) => {
    group.position.y = baseY + node.y * EXPLODE_GAP * currentExplode;
    group.position.x = baseX;
  });
  updateEdgeGeometryList(edgeLines, nodeMeshes);
  updateEdgeGeometryList(cmpEdgeLines, cmpMeshes);
}

function animate() {
  requestAnimationFrame(animate);
  currentExplode += (explodeAmount - currentExplode) * 0.06;
  applyExplodeLayout();
  if (pass.anim?.edge?.curve) {
    const u = Math.min(1, (performance.now() - pass.anim.t0) / pass.anim.dur);
    const p = pass.anim.edge.curve.getPoint(u); p.add(primaryGroup.position);
    tokenMesh.position.copy(p);
    tokenMesh.visible = true;
    if (u >= 1) {
      const id = pass.order[pass.index];
      if (nodeMeshes[id]) tokenMesh.position.copy(nodeMeshes[id].group.position).add(primaryGroup.position);
      pass.anim = null;
    }
  }
  controls.update();
  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);
}

window.addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
  labelRenderer.setSize(innerWidth, innerHeight);
});

window.addEventListener('keydown', (e) => {
  if (e.target.matches('input, select, textarea')) return;
  if (e.key === 'Escape') {
    stopPass(); endTour();
    if (cutawayMode) exitCutaway();
    else clearFocus();
  }
  if (e.key === ' ') { e.preventDefault(); togglePlay(); }
  if (e.key === 'ArrowRight') { pass.playing = false; passNext(); }
  if (e.key === 'ArrowLeft') { pass.playing = false; passPrev(); }
  if (e.key === '/' && !e.ctrlKey && !e.metaKey) {
    e.preventDefault();
    document.getElementById('layer-search').focus();
  }
  if (e.key === 'c' || e.key === 'C') {
    if (cutawayMode) exitCutaway();
    else {
      let b = 0;
      if (focusedId && nodeById[focusedId]?.block !== undefined) b = nodeById[focusedId].block;
      enterCutaway(b);
    }
  }
  if (e.key === 't' || e.key === 'T') {
    runMode = 'train';
    document.querySelectorAll('.seg-btn').forEach(b => b.classList.toggle('active', b.dataset.mode === 'train'));
    loadGraph({ keepCamera: true }); syncURL();
  }
  if (e.key === 'i' || e.key === 'I') {
    runMode = 'infer';
    document.querySelectorAll('.seg-btn').forEach(b => b.classList.toggle('active', b.dataset.mode === 'infer'));
    loadGraph({ keepCamera: true }); syncURL();
  }
  const idx = parseInt(e.key, 10);
  if (idx >= 1 && idx <= ARCH_ORDER.length) loadArchitecture(ARCH_ORDER[idx - 1]);
});

// Boot
initToyInputs();
readURL();
updatePassUI();
animate();
