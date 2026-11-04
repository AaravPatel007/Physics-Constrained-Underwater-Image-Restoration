# Sunlight from depth: what changed and why

Branch `sunlight_phy_model`, started from `main` @ `721a609` (3 Nov 2026). Nothing outside `sunlight_phy/` was changed.

## 1. The idea in one paragraph

The baseline BetaNet predicts 12 numbers per photo, and three of them are the sunlight colour **E** (R, G, B) that
lights the scene. The professor's suggestion: don't let the network guess E freely. Let it predict **one physical
number, the depth below the water surface d_s**. Then compute E from the textbook law for how sunlight fades on its
way down:

```
E_c(d_s) = E0_c · exp(−K_d,c · d_s)        c = R, G, B
```

So the network goes from 12 outputs to **10**: 3 β_D + 3 β_B + 3 B∞ + 1 d_s. Everything else stays the same.

## 2. Two distances that must not be mixed up

| Symbol | Meaning | Where it is used |
|---|---|---|
| **z** | camera → object distance along the line of sight (range map, per pixel) | direct-signal loss `exp(−β_D z)` and backscatter `1 − exp(−β_B z)`. **Unchanged.** |
| **d_s** | depth of the scene below the sea surface (one number per photo) | only the sunlight colour E. **New output.** |

## 3. Two attenuation coefficients that must not be mixed up

| Symbol | Name | What it describes |
|---|---|---|
| **β_D** | (wideband) beam attenuation of the direct signal | light travelling camera ← object along a narrow path |
| **K_d** | diffuse attenuation of downwelling irradiance | sunlight from the whole sky going down through the water column |

They are different quantities. K_d is usually smaller than the beam attenuation in turbid water (forward-scattered light
still counts as downwelling light). In clear, absorption-dominated water the two are close.

## 4. How the existing synthetic data already makes E (we checked the code)

`src/physics.py`, `sample_water`, lines 121–128:

```python
d_v = U(0.5, 12.0)                          # depth below the surface [m]   (never saved)
E   = exp(−β_D · d_v · U(0.6, 1.0))         # the U(0.6, 1.0) factor = κ    (never saved)
E   = E / max_c(E)                          # brightest channel = 1
E   = max(E, 0.03)                          # floor
```

So the generator already uses the proposed law, with:

- **E0 = (1, 1, 1)**: white surface light, implicit. That matches the clean images, which are NYU indoor photos
  treated as "the scene under white light".
- **K_d,c = κ · β_D,c**, with **κ ~ U(0.6, 1.0)** drawn per image (one κ for all three channels). This ratio is a
  modelling shortcut of the generator. We found no citation for it in the code or the docs.
- **d_s = d_v ~ U(0.5, 12) m.**

Then E is normalised (brightest channel = 1) and floored at 0.03. E is therefore a **colour** (2 degrees of freedom),
not an absolute brightness. Overall brightness lives in the exposure jitter `k ~ U(0.7, 1.3)` in `train.make_batch`.

Caveat for later work: the β_D sampler itself uses UWCNN's Jerlov table (`JERLOV_N`). Jerlov water types are
defined from measurements of downwelling-irradiance attenuation, so those numbers are probably closer to K_d than to
a beam attenuation. We have **not** verified this against a primary source (e.g. Solonenko & Mobley 2015).

## 5. Decision: option A (chosen by the user, 3 Nov)

**K_d,c = κ₀ · β̂_D,c with the fixed κ₀ = 0.8**, the midpoint of the generator's κ range. This is the generator's own
relation, so the training data, the splits and the baseline stay valid. The only change is the E head. The cost is
that E inherits any error in β̂_D, and the true κ of each water (0.6–1.0) is unknown to the network (see §8).

Rejected alternative (B): K_d from a literature table, independent of β_D. That needs new synthetic data (so a
retrained baseline) and a water-type input to pick the right K_d row.

## 6. The new model (`model_sun.py`, class `SunDepthBetaNet`)

```
photo + range z ─► ResNet-18 (4-channel stem, unchanged) ─► 512 ─► 256 ─► 10 numbers
   o[0:3]  → β_D = exp(clamp(o) − 1.5)          (identical to BetaNet)
   o[3:6]  → β_B = exp(clamp(o) − 1.5)          (identical)
   o[6:9]  → B∞  = sigmoid(o − 1)               (identical)
   o[9]    → d_s = 15 m · sigmoid(o)            (new)
   K_d = 0.8 · β_D (detached)                   (new, physics)
   E   = exp(−K_d d_s) / max_c(…),  ≥ 0.03      (new, physics; sun_physics.sunlight_from_depth)
```

- **Depth range (0, 15) m.** The sigmoid keeps it positive and bounded. 15 m covers every depth label (0.5–12 m) and
  every depth that could reproduce a training E under the fixed κ₀: at most 12 m · (1.0 / 0.8) = 15 m.
- **Same E convention as the baseline.** Brightest channel = 1, floor 0.03. So `physics.restore`, the image loss,
  `safe_restore` (Day 1b) and the evaluation code use the computed E with **no changes**.
- **Numerical stability.** The exponent is shifted by its per-image maximum before `exp()`, so deep or murky water
  cannot underflow. Everything is computed in float32 (the network runs in bf16 autocast, the outputs are cast up).
- **E0** is fixed to white, as in the generator. Because E is max-normalised, a different E0 would only change the
  colour, not the scale.
- **β_D is detached inside K_d.** β_D then receives exactly the gradients it gets in the baseline (its own coefficient
  loss + water removal in the image loss), and the sunlight term cannot pull on it. The sunlight path trains only
  the depth unit. This keeps the experiment "only the E component changed". It can be switched off with
  `--no-detach-kd` (not run).
- **Initialisation.** ImageNet ResNet-18, like the baseline; trained from scratch. We did not start from v1's weights:
  that would give the new model 60 extra epochs and confound the comparison.

## 7. Training (`train_sun.py`)

**Same as `baseline_v1_60ep_bigdata`.**
- Data: `data/nyu_train_big.npz`, 7 140 training rooms, the last 288 frames used for validation.
- Optimiser: 60 epochs, batch 24, AdamW (lr 5e-4, weight decay 1e-4), one-cycle schedule, bf16, gradient clip 1.0.
- Water: very murky water kept at 40 % (`--murky-keep 0.4`), new random water every batch.
- Augmentation: flips, range scale 0.7–2, exposure 0.7–1.3, sensor noise 0.002–0.01.
- Checkpoint rule: `val_coef + 10 · val_img_l1`.

**Identical data stream.** `train.make_batch` is used unchanged. Only its call to `sample_water` is pointed at
`sun_physics.sample_water_sun`, a line-by-line copy of the original that also returns d_v and κ. `verify_rng.py`
checks that the waters, images, flips, exposure and noise are bit-identical to the baseline's, batch for batch. So
the new model sees exactly the same 17 820 training batches as the baseline retrain.

**Loss:**

```
L = coef_loss(p, w)                    unchanged function: log-L1 β_D, β_B, 3·L1 B∞, log-L1 E
                                       its E term now compares the PHYSICS E with the true E (sunlight consistency)
  + 1.0 · img_loss(restore(I, z, p))   unchanged (transmission-weighted L1 of the restored image)
  + 1.0 · |log d̂_s − log d_s|          new: depth supervision against the generator's true d_v
```

- The depth term has the same form and weight as the other coefficient terms (log-L1, weight 1). No tuning was done.
- The E term is the baseline's own E supervision, kept on purpose. With the depth term it pulls d̂_s towards the
  depth that best explains the sunlight colour.

## 8. What the network can and cannot know (identifiability)

With the generator's own law, E_c = exp(−(β_D,c − min β_D) · κ · d_s): the colour depends on d_s only through
**τ = κ · d_s**. Even with a perfect reading of E and β_D, d_s is only known to lie in [τ, τ / 0.6].

- `eval_sun.depth_floor` computes the best estimate any method could make from a perfect τ: the posterior median,
  i.e. the geometric mean of the two ends. Its error is reported next to the network's as **"best possible from E"**.
- When red sits at the 0.03 floor (deep or murky water), red says nothing more about depth.
- On real photos only the product K_d · d_s is observable. Metres need an anchor such as a dive computer.

## 9. Files

| File | What it does |
|---|---|
| `sun_physics.py` | `sunlight_from_depth` (physics layer), `sample_water_sun` (generator copy returning d_s, κ, K_d) |
| `model_sun.py` | `SunDepthBetaNet` (10 outputs, E from depth) |
| `train_sun.py` | training, v1 recipe + depth loss |
| `verify_rng.py` | proves the depth labels belong to exactly the baseline's waters |
| `eval_sun.py` | synthetic test (654 rooms + 10 Jerlov types): images, coefficients, sunlight, depth, latency, attribution |
| `eval_real_sun.py` | UIEB (90 val) + Sea-thru charts, same protocol as `src/eval_real.py` |
| `figs_sun.py` | all figures |
| `data_setup/` | downloads and data preparation (parallel curl; the original `hf_hub_download` stalled on this network) |
