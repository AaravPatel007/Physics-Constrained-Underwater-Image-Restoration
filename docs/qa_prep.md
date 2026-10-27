# Likely questions at the review, and short answers

**Q: Why not just train an image-to-image network (like UWCNN / the friend's BridgeNet)?**
It would work on synthetic data, and we train one (the UNet baseline) to compare against. But a
black box gives no coefficients, can hallucinate colours, and gives no way to check whether the
output is physically sensible. Our network outputs 12 interpretable numbers and the image follows
from a fixed equation, so every restored pixel can be explained. The coefficients are also
useful on their own: they describe the water.

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
yet scene priors leak in. Next steps: outdoor/natural RGB-D, extra realism (forward-scatter blur,
marine snow), and self-supervised fine-tuning on real underwater photos.

**Q: How is this different from Sea-thru?**
Sea-thru solves an optimisation per image, using the darkest pixels at each range and an
illuminant map; it needs a good depth map and is slow (offline). Ours is a trained predictor, one
forward pass (~2 ms on a laptop GPU vs ~0.4–1.2 s per small frame for the classical pipeline), and
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
