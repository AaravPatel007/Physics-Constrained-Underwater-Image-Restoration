# Likely questions at the review, and short answers

**Q: Why not just train an image-to-image network (like UWCNN / the friend's BridgeNet)?**
We trained one (the UNet baseline). It wins on synthetic data (25.1 vs 20.0 dB) and on our two
chart frames, because it learns the training distribution directly. But on real water it paints
the warm colours of the indoor NYU rooms it was trained on (pink open water in D5), it has the
worst colour error on UIEB (ΔE 15.7 vs 13.9), and it gives no coefficients, which is what we were
asked for. BetaNet's image follows from a fixed equation, so every pixel can be explained and the
12 numbers describe the water. The obvious next step is a hybrid: physics first, then a small
learned refiner on the residual.

**Q: Why not train only on image quality and let the coefficients emerge?**
We tried it (image-loss-only ablation). Image PSNR is slightly better (20.7 vs 20.0 dB), but the β_D
error rises from 19 % to 33 % (fig14). Different wrong combinations of β_D, β_B, B∞ and E give
the same restored image, so the coefficients are not identifiable from the image alone. If you
want physically meaningful coefficients, you must supervise them.

**Q: You said "two coefficients" but output 12 numbers. Why?**
β_D (attenuation) and β_B (backscatter) are the two coefficients the professor named, but each
one is different for red, green and blue, so that is 6 numbers. To invert the model you also need
B∞ (the colour of the water far away) and E (the colour of the sunlight reaching the scene after
travelling down from the surface). That is 3 + 3 more. Without E, our v1 model blamed all the
missing red on the camera-to-object distance and produced magenta images.

**Q: Why does the network need depth? Can't it work from the photo alone?**
The attenuation is e^{−β·z}, so the same colour cast can come from murky water up close or clear
water far away. Without z, β is ambiguous. Our RGB-only ablation measures how much worse it gets.

**Q: Where does the depth come from on a real robot?**
Either a stereo camera or SfM (like the Sea-thru dataset), or monocular Depth Anything V2. We
measured DAv2 on real Sea-thru frames: the *shape* is excellent (AbsRel 0.07 after alignment) but
the *metric scale* is wrong (AbsRel 0.76 as-is). A single range reading (altimeter, DVL, laser
scaler, or known object size) fixes most of that (0.12). AUVs carry a DVL/altimeter anyway.

**Q: How do you know the coefficients are right on real images, where there is no ground truth?**
Three checks. (1) Colour charts: grey patches must come out grey (angular error, as in the
Sea-thru paper). (2) Consistency: frames from the same dive must give similar β's, and different
sites must differ. (3) Synthetic test rooms never seen in training, where the true β is known.

**Q: Isn't NYU-v2 (indoor rooms) a strange training set for underwater?**
Yes. It is the standard recipe (UWCNN) because it has real metric depth, but the scene content
is indoors. The physics is the same whatever the scene, and that is what the network must learn,
yet scene priors leak in. We have since fine-tuned on 552 real Sea-thru dive frames without labels
(next questions). Still to do: outdoor/natural RGB-D and extra realism (forward-scatter blur, marine snow).

**Q: How can you fine-tune on real photos when nobody knows their true coefficients?**
We use physics checks instead of labels, the same cues classical Sea-thru uses. The darkest pixels at
each distance must sit on the predicted haze curve. After removing the water, near and far parts of the
scene must have the same average colour. The restored image should be grey on average. Every step also
contains a synthetic batch with known coefficients, so the numbers keep their physical units. Without
that anchor, synthetic β_D error went from 19 % to 70 % while the real-image losses looked just as good.

**Q: Did it actually help on real photos?**
On held-out frames, yes. Without the final gray-world step, the grey-patch error on the chart frames went
from 16–17° to about 10°: the network now gets the sunlight colour right by itself. With gray-world it is
9.4 / 8.2°, level with gray-world alone (8.9 / 8.1°). Colour drift with distance on the never-seen dive
D3 nearly halved (0.47 → 0.27). The synthetic test did not get worse (PSNR 20.0 → 20.4 dB). Caveats: only
2 chart frames, and the outputs are flatter in contrast than before.

**Q: What went wrong during fine-tuning?**
Our first version added a "frames from the same dive share the same water" loss. Every dive then got the
same coefficients: one average water, so a turbid canyon and a clear reef read alike. The images still
looked fine. Only the per-dive coefficient check caught it, which is the same lesson as the image-only
ablation: good images do not imply right coefficients. Dropping that loss, or adding a small "teacher"
term that keeps the coefficients near the original model's, fixed it (fig18, fig20). Fine-tuning on UIEB
also made UIEB worse. UIEB has no depth, and the monocular depth has the wrong scale, which corrupts the
distance-based losses.

**Q: How did you choose the final model? Didn't you just pick the best chart number?**
No. The rule was written before looking: synthetic accuracy must not drop, the dives must still be told
apart, then pick the lowest physics losses on held-out frames. The chart frames and UIEB are only reported.
We changed the rule once, openly: its first spread measure let the collapsed model pass, because the
spread within each dive collapsed too. That change is documented in `src/select_ft.py`.

**Q: How is this different from Sea-thru?**
Sea-thru solves an optimisation per image, using the darkest pixels at each range and an
illuminant map; it needs a good depth map and is slow (offline). Ours is a trained predictor, one
forward pass (3.2 ms on a laptop GPU vs 0.38 s at the same size for the classical pipeline), and
it can be fine-tuned on data.

**Q: How is it different from DeepSeeColor (ICRA 2023)?**
DeepSeeColor also uses CNNs and the Sea-thru model, but its network weights *are* the parameters
of one scene: it re-optimises for each new water/stream. Ours is trained once on many waters and
predicts the parameters for a new image directly (amortised inference). The two are complementary:
ours could initialise theirs.

**Q: Your network is worse than gray-world on [some metric]. Why bother?**
Gray-world is a strong baseline for overall colour balance but it ignores distance: near and far
objects get the same correction, so far objects stay hazy and blue. The chart test measures only
the colour of one object at one distance, which is the best case for gray-world. The physics
model removes haze that grows with distance and tells you about the water. We show both honestly.

**Q: Why ResNet-18?**
Small (11 M parameters), ImageNet-pretrained, trains in about 10 minutes on a 4 GB laptop GPU, and
the task needs global statistics (12 numbers per image) rather than pixel-level detail.

**Q: Did you use the friend's repo?**
Yes: its Sea-thru implementation is our classical baseline, and its "watering" engine (blur, marine
snow) is planned for extra realism. We did not use its trained network, because it is
image-to-image (no coefficients), it was trained on only 3 fixed water types, and its own README
reports that it does not beat Sea-thru on real photos.

**Q: What are UCIQE / UIQM and why don't you rely on them?**
No-reference "underwater quality" scores built from colourfulness, contrast and sharpness. They
reward over-saturated images, so a method can score high while being physically wrong. We report
them only as secondary numbers.
