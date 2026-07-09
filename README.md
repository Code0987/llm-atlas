# LLM Atlas

3D architecture explorer for how LLMs are layered — GPT, BERT, Enc–Dec, diffusion, MoE, and more.

**Live site:** https://code0987.github.io/llm-atlas/  
**Repository:** https://github.com/Code0987/llm-atlas


3D architecture explorer for how LLMs are layered — GPT, BERT, Enc–Dec, diffusion, MoE, and more.

## Run

```bash
cd "llm-3d"
python3 -m http.server 8765
```

Open http://localhost:8765

## Files

| File | Role |
|------|------|
| `index.html` | Shell + UI |
| `styles.css` | Layout / theme |
| `explorer.js` | Three.js app, architectures, interactions |

## Features

### Core
- Model families: **GPT**, **BERT**, **Enc–Dec**, **Diffusion**, **MoE**
- Click **layer** or **wire** to focus connections
- Direction **arrows**, **Explode**, orbit / zoom / pan

### Interaction suite
| Control | What it does |
|---------|----------------|
| **▶ Play** | Forward-pass walk + token along wires |
| **Hyperparameters** | d_model, heads, layers, seq, vocab + size estimates |
| **Numeric toy** | dim-4 vector; transforms as you focus / step |
| **Train / Infer** | Emphasizes training vs runtime paths (loops, MLM, noise, KV story) |
| **Compare with** | Side-by-side ghost stack of another family |
| **Cutaway** | Expand one block into Q, K, V, scores, softmax, ·V, W_O, residual |
| **Tour** | Guided walkthrough for the active family |
| **Search** | Filter layers; Enter jumps to first hit |
| **Camera** | Default / Top / Side / Front |
| **CB** | Colorblind-safe palette |
| **PNG** | Export WebGL snapshot |
| **URL state** | Share `?arch=&mode=&compare=&focus=&…` |

### Keyboard
- `Space` play/pause · `←` `→` step · `Esc` clear/exit cutaway  
- `/` focus search · `C` cutaway · `T`/`I` train/infer  
- `1`–`5` switch model family  

## Notes
- Diagram is **schematic**, not a real model run.
- Param / FLOP numbers are **order-of-magnitude**.
- 3D draws a capped head/expert count; sliders mainly drive estimates + labels.
