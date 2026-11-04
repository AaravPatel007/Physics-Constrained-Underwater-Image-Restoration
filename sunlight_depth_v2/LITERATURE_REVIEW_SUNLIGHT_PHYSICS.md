# Literature review: the physics behind E = E0 · exp(−K_d · d_s)

This review supports the sunlight model of SunDepthBetaNet v2 and the v2 synthetic water (`water_v2.py`). It uses
a small number of sources, mostly primary. For each claim it says what was read and how:

- **read**: the text was extracted from the PDF and the passage checked;
- **abstract / secondary**: only the abstract or another paper's description was available.

Where a relation is our own derivation from cited equations, it is labelled **derived**.

Every statement is tagged with one of three levels:

- **[physics]**: an established physical law or definition;
- **[empirical]**: a published fit or approximation with stated limits;
- **[project]**: a modelling choice made by this project.

---

## 1. Summary table

| # | Paper | Equation / assumption supported | Key finding | Relevance to our model |
|---|---|---|---|---|
| 1 | Gordon (1989), *L&O* | Definition of K_d and the exponential decay of downwelling irradiance with depth | K_d = −(1/E_d) dE_d/dz (p. 1389). Once corrected for the light-field geometry, K_d obeys the Lambert–Beer law within 5–10 % in Case 1 water (abstract). | The equation E = E0 exp(−K_d d_s), and its accuracy limits. |
| 2 | Lee, Du & Arnone (2005), *JGR* | K_d from the water's absorption and backscatter, and its sun-angle dependence | K_d = (a + b_b)/μ₀ (Eq. 4b). Gordon (1989) found K_d(0) ≈ 1.04 (a + b_b)/μ₀ (§1). Their model, Eq. 11: K_d = (1 + 0.005 θ_a) a + 4.18 (1 − 0.52 e^{−10.8a}) b_b. | Why K_d differs from beam attenuation, and why it grows at low sun (our 1/μ_w factor). |
| 3 | Akkaynak & Treibitz (2018), *CVPR* | K_d ≠ beam attenuation; the light at depth falls off exponentially; D65 at the surface | "Since both β and K_d are attenuation coefficients, which do we use …? The answer is both, and their difference is important … Generally, β is 2–5 times greater than K_d" (§2). Jerlov's types are defined "based on the diffuse downwelling attenuation coefficients K_d" (Fig. 1c). | The core justification for predicting K_d separately from β_D, and for E0 = D65 as the reference white. |
| 4 | Akkaynak et al. (2017), *CVPR* | Origin of the Jerlov types; RGB ("wideband") coefficients depend on camera, range, reflectance and illuminant | Jerlov "measured the downwelling irradiance at various depths" (§3.1). The wideband coefficient (Eq. 9) "depends on S_c, z, Δz, ρ and E, as opposed to β(λ) that is a property of the water" (§4). | Why the RGB K_d of the sunlight path need not equal the RGB β_D of the line of sight, even in the same water (our independent per-channel spread). |
| 5 | Li, Anwar & Porikli (2020), *Pattern Recognition* (UWCNN) | The Jerlov table used by this project | Table I gives N_λ for the 10 Jerlov types at 600 / 525 / 475 nm (R/G/B, §1), citing Jerlov, *Marine Optics* (1976). Eq. 2: T_λ = N_λ^{d}. | The project's `JERLOV_N` is this table. −ln N_λ is the per-metre loss of Jerlov's downwelling irradiance, i.e. a K_d. |
| 6 | Solonenko & Mobley (2015), *Applied Optics* | Absorption and scattering (IOPs) consistent with each Jerlov K_d type | a(λ) and b(λ) for all Jerlov types, 300–700 nm, fitted to reproduce Jerlov's K_d (abstract). | The route to a physically separate beam attenuation c = a + b (future work, §7). |
| 7 | Judd, MacAdam, Wyszecki et al. (1964), *JOSA* | Daylight colour is close to a one-parameter family (the daylight locus) | 622 measured daylight spectra are reproduced by a mean plus two characteristic vectors, with chromaticities on y = 2.870x − 3.000x² − 0.275. Adopted by the CIE. | E0 is drawn on the CIE daylight locus: one physical degree of freedom (CCT). |
| 8 | CIE 15:2004, *Colorimetry* (3rd ed.) | Daylight locus x_D(T) for 4000–25000 K | x_D = 0.244063 + 0.09911·10³/T + 2.9678·10⁶/T² − 4.6070·10⁹/T³ (4000–7000 K); a second polynomial for 7000–25000 K. | The exact formulas in `water_v2.daylight_xy`. |
| 9 | Hernández-Andrés et al. (2001), *JOSA A* | Measured spread of daylight colour temperature | 2600 daylight spectra (Granada) span CCT ≈ 3758–34573 K; the most frequent is ≈ 175–180 mired (≈ 5600 K). | The CCT range drawn for E0 (4000–10000 K). |
| 10 | Gregg & Carder (1990), *L&O* | Surface irradiance as direct sun + diffuse sky, depending on solar zenith and atmosphere | A spectral model (350–700 nm) of the direct and diffuse irradiance at and just below the sea surface, from solar zenith, aerosol, water vapour and ozone. | The more physical, not implemented, alternative for E0 (§7). |

---

## 2. Downwelling irradiance attenuation: E_d(λ, d) = E_d(λ, 0) · exp(−K_d(λ) d)

### Where it comes from **[physics]**

Gordon (1989, p. 1389) defines the diffuse attenuation coefficient of downwelling irradiance E_d as

  K_d = −(1/E_d) · dE_d/dz .

If K_d does not change with depth, integrating from the surface to depth d gives the exponential law
E_d(d) = E_d(0) exp(−K_d d). Lee et al. (2005, Eq. 2) write the layer-averaged form, K_d(z₁→z₂) = ln[E_d(z₁)/E_d(z₂)] / (z₂ − z₁).

Akkaynak & Treibitz (2018) use the same picture for imaging: "Ambient light attenuates exponentially with depth d
reaching an object" (Fig. 2 caption). The light at the object depth is written E(d, λ) in their Eqs. 4–10.

### What K_d is

- **An apparent optical property (AOP).** It describes how all downward-travelling photons (direct sun and diffuse
  sky light, scattered in all directions) are lost with depth. It is not the loss of one collimated beam.
- **Partly a property of the light field.** Gordon (1989, abstract) stresses that K_d depends on the geometric
  structure of the light field, i.e. on the sun angle and the sky. Akkaynak & Treibitz (2018, §2) note that its
  magnitude "only weakly depends on ambient light", which is why it is used to classify water bodies.
- **Tied to the water's inherent optical properties (IOPs)** **[empirical]**. Lee et al. (2005):
  - Eq. 4b: K_d = (a + b_b)/μ₀, with a the absorption, b_b the backscattering and μ₀ the mean cosine of the
    downwelling light just below the surface;
  - Gordon (1989) found empirically K_d(0) ≈ 1.04 (a + b_b)/μ₀;
  - Lee et al.'s own semi-analytical model, Eq. 11, with θ_a the above-surface solar zenith angle in degrees:
    K_d = (1 + 0.005 θ_a) a + 4.18 (1 − 0.52 e^{−10.8 a}) b_b.

### When the exponential form is valid

- **Case 1 (open-ocean) water.** Gordon (1989, abstract): geometry-corrected K_d "satisfy the Lambert-Beer law with a
  maximum error of 5–10 % depending on wavelength". Lee et al. (2005) note the simple (a + b_b)/μ₀ forms
  underestimate the contribution of backscatter and need refining. They report 2 % average and 12 % maximum error
  for their own model.
- **K_d roughly constant with depth.** This requires a homogeneous water column, no strong stratification, and
  depths below the first few metres where the light field is still changing. Otherwise K_d must be read as a
  depth-averaged value (Lee et al. 2005, Eq. 2).
- **Moderate solar zenith angles.** The 1/μ₀ dependence is an approximation that degrades for a low sun. Our
  generator therefore limits the sun to 0–60° from the zenith **[project]**.
- **Elastic scattering only.** Inelastic effects (e.g. Raman scattering, fluorescence) are ignored; they matter
  mostly deep down and in the red.

### How K_d varies with wavelength and water

- **Clear ocean water (Jerlov I–III)** absorbs red strongly. K_d is largest in the red and smallest in the blue,
  hence the blue cast. This is visible in the project's table: type I has K_d = 0.217 / 0.040 / 0.018 m⁻¹ at
  600 / 525 / 475 nm.
- **Coastal water (1C–9C)** has dissolved organic matter and particles that absorb blue as well. Blue K_d rises
  until it matches or exceeds red: 9C has 0.598 / 0.777 / 1.238 m⁻¹. See also Akkaynak et al. (2017, §3.1): "short
  wavelengths [attenuate] just as strongly as long ones".

---

## 3. K_d (diffuse) vs β_D (beam / direct-signal) attenuation: not interchangeable

**[physics]**

- **Beam attenuation** c(λ) = a(λ) + b(λ) is the loss of a collimated beam (Akkaynak & Treibitz 2018, Table 1:
  "beam attenuation coefficient: a(λ)+b(λ)"). Any photon scattered out of the beam, even by a tiny angle, is lost.
  In the revised image formation model, the direct-signal coefficient β_D of an RGB camera is a wideband version of
  this beam loss along the line of sight. It also depends on range, reflectance, illuminant and sensor
  (Akkaynak et al. 2017, Eq. 9 and §4).
- **Diffuse attenuation** K_d counts all downward photons. A photon scattered forward is still travelling downward
  and still counts, so mostly absorption and *back*-scattering remove light: K_d ≈ 1.04 (a + b_b)/μ₀
  (Gordon 1989, via Lee et al. 2005).
- **Size of the difference.** Akkaynak & Treibitz (2018, §2): "Generally, β is 2–5 times greater than K_d". They cite
  Lee et al. (2015, *Remote Sensing of Environment* 169:139–149) for this; we did not read that paper.

**Derived** (our algebra from the two equations above, not a quoted result):

  K_d / c ≈ 1.04 (a + b_b) / [μ₀ (a + b)] = 1.04 (1 − ω + ω B̃) / μ₀,

with ω = b/c the single-scattering albedo and B̃ = b_b/b the backscattering fraction. The ratio is close to (or
above) 1 where absorption dominates: red light, clear water, ω small. It is much smaller than 1 where scattering
dominates: blue and green light in particle-rich coastal water, ω large. **So no single constant can link K_d to
β_D**, because the ratio changes with wavelength, water type and sun angle.

**Consequence for this project [project]:** the project's line-of-sight β_D was built from the Jerlov table
(`src/physics.py`: β = −ln N_λ with N_λ from UWCNN Table I). By its origin that table is a K_d table: Jerlov's types
are defined by K_d, and Jerlov measured downwelling irradiance (refs 3, 4). Akkaynak et al. (2017, §3.1) made the same
identification, calling the irradiance-derived coefficient "the total attenuation coefficient".

- **What we changed:** v2 now uses the table for K_d, its proper meaning.
- **What we kept:** β_D unchanged, so all three models and all earlier experiments share the same line-of-sight
  water.
- **Why it matters:** with the 2–5× factor above, real beam attenuation would be larger than the project's β_D.
  This is a limitation of the project's water model, flagged in the summary. Fixing it needs per-type c(λ) from
  Solonenko & Mobley (2015) and would change every earlier experiment.

---

## 4. Jerlov water types as a source of K_d

**What the types are** **[physics]**:

- Jerlov's classification is based on measured downwelling irradiance, i.e. K_d. Sources:
  - Akkaynak et al. 2017, §3.1: "They measured the downwelling irradiance at various depths … Jerlov categorized
    the world's oceans into five oceanic and five coastal classes";
  - Akkaynak & Treibitz 2018, Fig. 1c: defined "based on the diffuse downwelling attenuation coefficients K_d he
    measured globally".
- The original tables are in Jerlov, *Marine Optics* (Elsevier Oceanography Series 14, 1976); we did not have
  access to the book.
- The RGB values used here come from UWCNN Table I, which cites Jerlov (1976) and samples it at 600, 525 and 475 nm.

**Physically justified use** **[project]**: in v2, K_d,c = (Jerlov table interpolated at the water's position) ×
(the water's turbidity factor) × (per-channel spread, σ = 0.15) / μ_w(θ_s).

- **Table and turbidity:** the same water body as the line of sight.
- **Per-channel spread:** the RGB-effective coefficient of the sunlight path differs from that of the line of sight
  (ref 4, Eq. 9).
- **1/μ_w:** the sun-angle dependence of K_d, with μ_w the cosine of the refracted solar zenith angle
  (refs 1, 2). Lee et al.'s Eq. 11 gives a similar size of effect: ≈ +30 % at θ_a = 60°, against 1/μ_w(60°) = 1.31.

**Limits:**

- The table values are single wavelengths, not band integrals.
- Treating the table as the overhead-sun value is our assumption.
- Jerlov's classes "[do] not [cover] all water bodies", e.g. very murky lagoons (ref 4, §3.1).
- **Later work:** Solonenko & Mobley (2015) supply IOPs (a, b) consistent with each Jerlov K_d. This makes a
  self-consistent c(λ) and K_d(λ) per type possible.

---

## 5. Surface irradiance E0

### What varies at the surface

- **Solar zenith angle and atmosphere** **[physics/empirical]**. Global irradiance at the sea surface is direct sun
  plus diffuse sky light. Both depend on the solar zenith angle, aerosols, water vapour and ozone: Gregg & Carder
  (1990) model them separately at 350–700 nm, at and just below the sea surface.
  - A low sun gives a longer atmospheric path, a redder direct beam and relatively more blue sky light.
  - Cloud changes the balance again.
- **Colour of daylight** **[empirical]**. Judd et al. (1964) showed that the spectral shape of daylight is captured
  by a mean plus two characteristic vectors, with chromaticities on a quadratic daylight locus. The CIE adopted this
  as the D-series illuminants (CIE 15:2004, formulas for 4000–25000 K). Measured daylight spreads widely:
  Hernández-Andrés et al. (2001) found CCT ≈ 3758–34573 K over 2600 Granada spectra, most often near 5600 K. That
  range is as reported by a later paper (Peyvandi et al. 2016, *JOSA A*); we could not open the 2001 paper.
- **The surface itself.** Fresnel transmission into the water is nearly the same for all colours, so it changes
  brightness more than colour. Brightness is irrelevant here because E is max-normalised.

### How E0 is represented in our generator **[project]**

1. **Draw a colour temperature:** CCT uniform in 1/CCT (mired) between 4000 K and 10000 K.
   - 4000 K is the lower validity limit of the CIE locus formula.
   - 10000 K covers overcast and blue-sky-dominated light, in the representative range of Judd et al. (1964).
   - Very blue skies above 10000 K are excluded, a choice.
2. **Convert to linear sRGB:** CIE daylight locus (x_D, y_D) → XYZ → linear sRGB (IEC 61966-2-1 matrix).
3. **White-balance:** divide by the colour at 6504 K (D65), so that D65 → [1, 1, 1]. Akkaynak & Treibitz (2018, §3.2)
   also "assume CIE D65 at the surface", and it is the old generator's implicit white.
4. **Normalise** to max 1.

E0 therefore has one physical degree of freedom. The network still outputs three numbers, compared only up to
scale, so two degrees of freedom in practice.

**Limitations:**

- No coupling between CCT and the sun angle used for K_d. In nature a low sun is both warmer and gives a larger K_d.
- The camera is assumed to have sRGB primaries.
- The RGB product E0 ⊙ exp(−K_d d_s) is a wideband approximation of a spectral product (ref 4).

---

## 6. The previous assumption K_d = 0.8 β_D (SunDepthBetaNet v1)

**No published source supports it.**

- **Where it came from [project]:** the old generator (`src/physics.py`, line 126) computes the sunlight as
  E = exp(−β_D · d_v · κ) with κ ~ U(0.6, 1.0). The comment cites no source, and none was found in the repository.
  v1 fixed κ₀ = 0.8, the midpoint of that range.
- **Why it cannot be a general law:** published physics points the other way.
  - K_d and beam attenuation differ by a factor that depends on the single-scattering albedo, the backscatter
    fraction and the sun angle (§3, derived from refs 1, 2).
  - Beam attenuation is typically 2–5× K_d (ref 3), so K_d/β ≈ 0.2–0.5 if β is a true beam attenuation, not 0.8.
  - In this project the ratio means something else again, because the project's β_D itself came from a K_d table
    (§3).
- **Our measurement of its cost:** in the v1 experiment, even with all true coefficients and the true depth, the
  K_d = 0.8 β_D rule cost 0.47 dB PSNR on the old generator's own data (`sunlight_phy/docs/02_results.md`, §3).
  The v2 model removes the rule.

---

## 7. What would make the formulation more physical (not done here)

1. **Separate c(λ) and K_d(λ) per Jerlov type.** Take a(λ) and b(λ) from Solonenko & Mobley (2015), compute c = a + b
   and K_d from Lee et al. (2005, Eq. 11), and integrate both over camera sensitivities (ref 4, Eq. 9). This would
   replace the project's β_D table, so every earlier result would change.
2. **Couple E0 and the sun angle** with a sky model such as Gregg & Carder (1990). This gives the direct and diffuse
   parts as functions of solar zenith and atmosphere, instead of an independent CCT.

---

## 8. Presentation talking points

**Q: Where does E = E0 exp(−K_d d_s) come from?**
It is the definition of the diffuse attenuation coefficient of downwelling irradiance, K_d = −(1/E_d) dE_d/dz
(Gordon 1989, p. 1389), integrated with K_d constant over depth. It is the same "ambient light attenuates
exponentially with depth" used by the revised underwater image formation model (Akkaynak & Treibitz 2018).

**Q: When is that valid?**
- Fairly homogeneous water with K_d roughly constant with depth, elastic scattering, and a moderate sun angle.
- In open-ocean water, geometry-corrected K_d follows Lambert–Beer within 5–10 % (Gordon 1989).
- Our generator keeps the sun within 60° of the zenith.

**Q: Why not use β_D for the sunlight, as v1 did?**
They are different quantities. β_D (beam / direct-signal attenuation) loses any photon scattered out of the beam.
K_d loses only absorbed or back-scattered photons, K_d ≈ 1.04 (a + b_b)/μ₀ (Gordon 1989; Lee et al. 2005).
Akkaynak & Treibitz (2018): "both, and their difference is important … β is 2–5 times greater than K_d." Their
ratio depends on wavelength, water type and sun angle, so a fixed 0.8 cannot hold.

**Q: Where did 0.8 come from?**
From our own synthetic generator, which used κ ~ U(0.6, 1.0) with no citation. v1 took the midpoint. There is no
literature behind it, and it is one of the things v2 removes.

**Q: Where do your K_d values come from?**
- From the Jerlov water-type tables, which are tables of K_d: Jerlov measured downwelling irradiance (Akkaynak et
  al. 2017, §3.1; Akkaynak & Treibitz 2018, Fig. 1c), used at 600/525/475 nm as in UWCNN.
- Three variations on top:
  - the water's turbidity;
  - a per-channel spread, because RGB coefficients depend on the light and path (Akkaynak et al. 2017, Eq. 9);
  - the sun angle, through 1/μ_w (Gordon 1989; Lee et al. 2005, Eq. 11 gives a similar ≈ 30 % at 60°).

**Q: But your β_D comes from the same Jerlov table — isn't that the same thing?**
Be candid here:
- The project's β_D was inherited from UWCNN, which used Jerlov's (K_d) table as a beam attenuation, a common
  shortcut in underwater computer vision (Akkaynak et al. 2017 did the same in §3.1).
- We kept β_D unchanged so that all three models and every earlier experiment share the same line-of-sight water.
- In v2 the two are no longer tied: K_d/β_D varies from about 0.73 to 1.67 (5–95 %) per channel and image.
- Properly separating them needs c = a + b per water type (Solonenko & Mobley 2015). That is listed as future work.

**Q: How do you model the surface light E0?**
- With the CIE daylight locus (Judd et al. 1964; CIE 15:2004): measured daylight colour is essentially a
  one-parameter family indexed by colour temperature.
- We sample 4000–10000 K, which covers low-sun through overcast light; measured daylight spans about 3800 K to
  well above 10000 K (Hernández-Andrés et al. 2001).
- We white-balance at D65, the same reference Akkaynak & Treibitz (2018) use for surface light.
- Brightness cannot be recovered and is not used, because E is normalised. The network predicts E0's colour only.

**Q: Can the network actually separate E0, K_d and d_s?**
- **Not from the sunlight colour alone.** E has two colour degrees of freedom, and K_d and d_s enter only as their
  product.
- **Separation comes from other cues and priors:**
  - K_d is tied to the water the network also sees along the line of sight (same water body);
  - E0 lies on the one-dimensional daylight locus;
  - the depth is supervised directly.
- The results show how well that works: the K_d, E0 and depth errors, and the PSNR gained when each is replaced by
  its true value (see `SUNLIGHT_DEPTH_V2_SUMMARY.md`).

**Q: Is the comparison fair?**
- All three models were trained on the same v2 water, with the same images, splits, seeds, augmentation, recipe and
  checkpoint-selection rule. All are tested on the same scenes.
- The older checkpoints, trained on the old water, are shown only as a labelled reference.

---

## 9. References (full)

1. H. R. Gordon, "Can the Lambert-Beer law be applied to the diffuse attenuation coefficient of ocean water?",
   *Limnology and Oceanography* 34(8):1389–1409, 1989. DOI 10.4319/lo.1989.34.8.1389.
   *Read:* p. 1389 (definition of K_d, abstract). Copy: https://misclab.umeoce.maine.edu/ftp/classes/OO2015/Readings/Gordon_L&O1989.pdf
2. Z.-P. Lee, K.-P. Du, R. Arnone, "A model for the diffuse attenuation coefficient of downwelling irradiance",
   *Journal of Geophysical Research* 110, C02016, 2005. DOI 10.1029/2004JC002275.
   *Read:* §1 (Gordon 1989 approximation), Eqs. 2, 4a, 4b, 11, 12.
3. D. Akkaynak, T. Treibitz, "A Revised Underwater Image Formation Model", *CVPR* 2018.
   https://openaccess.thecvf.com/content_cvpr_2018/html/Akkaynak_A_Revised_Underwater_CVPR_2018_paper.html
   *Read:* Fig. 1 caption, §2, Table 1, Eqs. 1–10, §3.2.
4. D. Akkaynak, T. Treibitz, T. Shlesinger, R. Tamir, Y. Loya, D. Iluz, "What is the space of attenuation coefficients
   in underwater computer vision?", *CVPR* 2017, pp. 568–577. DOI 10.1109/CVPR.2017.68.
   *Read:* §3.1, §4, Eqs. 6 and 9.
5. C. Li, S. Anwar, F. Porikli, "Underwater scene prior inspired deep underwater image and video enhancement",
   *Pattern Recognition* 98:107038, 2020. arXiv:1807.03528 (https://arxiv.org/abs/1807.03528).
   *Read:* Eq. 2, Table I, §1 (wavelengths), reference [3].
6. M. G. Solonenko, C. D. Mobley, "Inherent optical properties of Jerlov water types", *Applied Optics*
   54(17):5392–5401, 2015. DOI 10.1364/AO.54.005392. *Abstract only.*
7. D. B. Judd, D. L. MacAdam, G. Wyszecki, H. W. Budde, H. R. Condit, S. T. Henderson, J. L. Simonds, "Spectral
   distribution of typical daylight as a function of correlated color temperature", *JOSA* 54(8):1031–1040, 1964.
   DOI 10.1364/JOSA.54.001031. *Secondary* (DOI record and the CIE D-series description).
8. CIE 15:2004, *Colorimetry*, 3rd edition, Commission Internationale de l'Éclairage, 2004 (daylight locus).
   *Secondary* (formulas as reproduced in standard references; the coefficients match `water_v2.daylight_xy`).
9. J. Hernández-Andrés, J. Romero, J. L. Nieves, R. L. Lee Jr., "Color and spectral analysis of daylight in southern
   Europe", *JOSA A* 18(6):1325–1335, 2001. DOI 10.1364/JOSAA.18.001325.
   *Secondary* (CCT range as reported by Peyvandi et al., *JOSA A* 2016).
10. W. W. Gregg, K. L. Carder, "A simple spectral solar irradiance model for cloudless maritime atmospheres",
    *Limnology and Oceanography* 35(8):1657–1675, 1990. DOI 10.4319/lo.1990.35.8.1657. *Abstract / secondary.*
11. IEC 61966-2-1:1999, *Default RGB colour space — sRGB* (XYZ → linear sRGB matrix).

Cited only through ref 3, not read: Z. Lee et al., "Secchi disk depth: A new theory and mechanistic model for
underwater visibility", *Remote Sensing of Environment* 169:139–149, 2015 (the "β is 2–5× K_d" statement); and
N. G. Jerlov, *Marine Optics*, Elsevier Oceanography Series 14, 1976 (the original tables).
