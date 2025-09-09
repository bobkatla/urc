
---

## How to incorporate URC into a generative model (and what to test first)

Below is a **minimal integration recipe** that’s model-agnostic (Diffusion / GAN / VAE). The key is to compute **projected features** \(y\), **condition embeddings** \(e\), and **labels** for per-class thresholds.

### 0) Decide φ and e

- **φ (features y)**:  
  - If you already compute rich features (e.g., a perceptual encoder), **reuse them** as \(y\) (set `phi=None` in config).  
  - Otherwise, use `PhiImage` once to fit PCA stats, and call `urc.project_phi(x)` to project both real and generated.
- **e (conditions)**:
  - Class-conditional: one-hot or learned embedding. For real data, **detach**: `e_real = g(c_real).detach()`. For generated, **no detach** so grads can flow if needed (URC won’t backprop through e by default, but avoids blocking).

### 1) One-time setup

- Pick **dimensions**: `d_y` must match your \(y\) feature dim; `d_e` matches your conditioning vector size.
- Set **MDN stability**:
  - `var_floor=1e-3` (prevents vanishing variance).
  - `lr=5e-4~1e-3`.
- Choose **quantiles**:
  - `num_classes = #classes`.
  - `window_size` ≳ 10× your batch size per class.
  - `warmup_min` so τ only activates after a few dozen samples per class (16–128).
  - `alpha=0.05~0.2` initially.

### 2) In your training step

1) Compute `y_real`, `y_gen`, `e_real`, `e_gen`, and `labels` (LongTensor).  
2) Call:
   ```python
   out = urc.step(
       y_real=y_real, e_real=e_real, labels_real=labels,
       y_gen=y_gen,   e_gen=e_gen,   labels_gen=labels,
       # optionally: y_gen_neg, e_gen_neg, labels_neg  (for separation)
   )
   total_loss = base_loss + urc_weight * out["loss"]
