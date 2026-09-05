# Mathematics of the Two-Hand Flower Bloom

This document derives every formula used in `utils/rose.py`, `utils/violet.py`,
`utils/common.py`, `utils/hand_gestures.py`, and `utils/stem_leaf.py`.

Both flowers are generated as a parametric surface

$$
(r, \theta) \;\longmapsto\; \big(X(r,\theta), \; Y(r,\theta), \; Z(r,\theta)\big), \qquad r \in [0,1]
$$

where $Z$ is treated as the flower's **bloom / opening axis**.

---

## 1. Shared petal-shaping term

Both flowers use the same "rose curve" petal envelope. For polar angle $\theta$
and a petal count $p$:

$$
x(\theta) \;=\; 1 \;-\; \frac{1}{2}\left[\frac{5}{4}\left(1 - \frac{(p\,\theta) \bmod 2\pi}{\pi}\right)^{2} - \frac{1}{4}\right]^{2}
$$

This is a periodic function of $\theta$ that produces $p$ lobes ("petals") per
$2\pi$, and multiplies every radius so the surface pinches inward between
petals and bulges outward at petal centers.

---

## 2. Flower 1 — Rose (right hand)

**Domain**

$$
r \in [0,1], \qquad \theta \in [-2,\; 20\pi]
$$

The extra range beyond a single $2\pi$ loop makes $\theta$ sweep around
$10$ times, which is what gives the rose its layered, spiralling look.

**Parameters**

$$
p = 3.6, \qquad A = 1.995653, \qquad B = 1.27689, \qquad C = 8
$$

**Bloom angle $\varphi$ — theta-dependent**

At full bloom, $\varphi$ decays exponentially along the spiral, which is
what makes the rose unfurl progressively rather than open all at once:

$$
\varphi_{\text{open}}(\theta) = \frac{\pi}{2}\, e^{-\theta / (C\pi)}
$$

For the hand-driven animation, this is interpolated from a near-flat,
unopened bud state up to $\varphi_{\text{open}}$ using the bloom amount
$b \in [0,1]$ (from §4):

$$
\varphi_{\text{closed}}(\theta) = 0.05\, e^{-\theta/(C\pi)}
$$
$$
\varphi(\theta; b) \;=\; \varphi_{\text{closed}}(\theta) + b\big[\varphi_{\text{open}}(\theta) - \varphi_{\text{closed}}(\theta)\big]
$$

**Petal curl term**

$$
y(r,\theta;b) \;=\; A\, r^{2}\,(B r - 1)^{2}\, \sin\varphi(\theta;b)
$$

**Combined radius**

$$
R_2(r,\theta;b) \;=\; x(\theta)\Big[\,r\sin\varphi(\theta;b) + y(r,\theta;b)\cos\varphi(\theta;b)\,\Big]
$$

**Final coordinates**

$$
X = R_2 \sin\theta, \qquad Y = R_2 \cos\theta, \qquad
Z = x(\theta)\Big[\,r\cos\varphi(\theta;b) - y(r,\theta;b)\sin\varphi(\theta;b)\,\Big]
$$

---

## 3. Flower 2 — Violet (left hand)

**Domain**

$$
r \in [0,1], \qquad \theta \in [0,\; 2\pi]
$$

A single loop — **no spiral term** — which is the key structural
difference from the rose.

**Parameters**

$$
p = 5, \qquad A_2 = 1.95653, \qquad B_2 = 1.27689
$$

**Bloom angle $\varphi$ — constant**

Unlike the rose, $\varphi$ at full bloom does **not** depend on $\theta$ at
all — it's a single fixed value. This is exactly what removes the
spiralling/unfurling effect: every point on the flower opens by the same
amount simultaneously.

$$
\varphi_{\text{open}} = \frac{\pi}{2}\, e^{-\frac{2\pi}{8\pi}} = \frac{\pi}{2}\, e^{-1/4} \;\approx\; 1.223\ \text{rad} \;\approx\; 70.1^{\circ}
$$

Interpolated the same way as the rose, but with plain scalars instead of
$\theta$-dependent arrays:

$$
\varphi_{\text{closed}} = 0.05, \qquad \varphi(b) = \varphi_{\text{closed}} + b\big(\varphi_{\text{open}} - \varphi_{\text{closed}}\big)
$$

**Petal curl term, combined radius, and final coordinates** take the same
form as the rose (§2), substituting this flower's $x(\theta)$, $A_2$, $B_2$,
and constant $\varphi(b)$:

$$
y(r;b) = A_2\, r^{2} (B_2 r - 1)^{2} \sin\varphi(b)
$$
$$
R_2(r,\theta;b) = x(\theta)\big[r\sin\varphi(b) + y(r;b)\cos\varphi(b)\big]
$$
$$
X = R_2\sin\theta,\quad Y = R_2\cos\theta,\quad Z = x(\theta)\big[r\cos\varphi(b) - y(r;b)\sin\varphi(b)\big]
$$

---

## 4. Bloom amount from hand openness

Using MediaPipe hand landmarks (wrist $p_0$; fingertips $p_8, p_{12}, p_{16}, p_{20}$;
index/pinky knuckles $p_5, p_{17}$ for a palm-width scale reference):

$$
\text{openness} \;=\; \frac{\dfrac{1}{4}\displaystyle\sum_{t \in \{8,12,16,20\}} \lVert p_t - p_0 \rVert}{\lVert p_5 - p_{17} \rVert}
$$

Linearly mapped to a bloom amount and clamped to $[0,1]$:

$$
b_{\text{target}} \;=\; \operatorname{clip}\!\left(\frac{\text{openness} - 1.20}{2.65 - 1.20},\; 0,\; 1\right)
$$

**Temporal smoothing** (exponential moving average, per frame $k$):

$$
b_{\text{smooth}}^{(k)} \;=\; 0.82\, b_{\text{smooth}}^{(k-1)} \;+\; 0.18\, b_{\text{target}}^{(k)}
$$

## 5. Pinch detection ("plant the stem")

Using thumb tip $p_4$ and index tip $p_8$, normalized the same way:

$$
d_{\text{pinch}} \;=\; \frac{\lVert p_4 - p_8 \rVert}{\lVert p_5 - p_{17} \rVert}
$$

$$
\text{pinching} \iff d_{\text{pinch}} < 0.30
$$

Once true for a given hand, that flower's stem is planted permanently for
the session (a one-way switch) — see `main.py`.

---

## 6. Fixed camera transform (Y-up)

Both flowers' `create_XXX(bloom)` return $Z$ as the bloom axis. Before any
camera rotation, the point is remapped so that axis becomes world $+Y$
("up"), with the old $Y$ used as depth:

$$
\mathbf{p}_{\text{remap}} = (X,\; Z,\; Y)
$$

Azimuth $\alpha$ spins around the up axis ($Y$); elevation $\epsilon$ tilts
around $X$:

$$
R_y(\alpha) = \begin{pmatrix} \cos\alpha & 0 & \sin\alpha \\ 0 & 1 & 0 \\ -\sin\alpha & 0 & \cos\alpha \end{pmatrix}
\qquad
R_x(\epsilon) = \begin{pmatrix} 1 & 0 & 0 \\ 0 & \cos\epsilon & -\sin\epsilon \\ 0 & \sin\epsilon & \cos\epsilon \end{pmatrix}
$$

$$
\mathbf{p}_{\text{view}} = R_x(\epsilon)\, R_y(\alpha)\, \mathbf{p}_{\text{remap}}
$$

| Flower | Elevation $\epsilon$ | Azimuth $\alpha$ |
|---|---|---|
| Rose | $18^{\circ}$ | $-25^{\circ}$ |
| Violet | $70^{\circ}$ | $-12.7^{\circ}$ (close to the MATLAB reference's $81.2^\circ,\,-12.7^\circ$) |

This matrix is constant — never driven by the hand — so neither flower
rotates.

**Depth flip** (so the bloom opens toward the camera rather than away):

$$
z_{\text{view}} \;\leftarrow\; -z_{\text{view}}
$$

---

## 7. Perspective projection

With focal length $f = 650$, screen scale $s = 170$, and screen anchor
$(c_x, c_y)$:

$$
\text{depth} \;=\; \frac{f}{f + 100\, z_{\text{view}}}
$$

$$
p_x = c_x + s\, x_{\text{view}} \cdot \text{depth}, \qquad
p_y = c_y - s\, y_{\text{view}} \cdot \text{depth}
$$

---

## 8. Shading and color

Triangles are depth-sorted back-to-front (painter's algorithm) using mean
view-space depth. Each triangle's fill color is looked up in a 256-entry
LUT, indexed by a blend of the radial coordinate $r$ (distance from the
flower's center, $0$=center, $1$=petal tip) and normalized camera depth:

$$
\hat z = \frac{z_{\text{view}} - z_{\min}}{z_{\max} - z_{\min}}, \qquad
s_{\text{shade}} = 0.75\,\bar r + 0.25\,\hat z
$$

$$
\text{color} = \text{LUT}\big[\; \lfloor s_{\text{shade}} \cdot 255 \rfloor \;\big]
$$

**Rose LUT:** linear ramp from red $(1, 0, 0)$ to dark red $(0.25, 0, 0)$.

**Violet LUT:** built from the reference MATLAB colormap
`violet_map = [gold_map; blue_map]` — 2 rows of gold
$(255,215,0) \to (250,210,0)$ followed by 20 rows of blueviolet
$(138,43,226) \to$ indigo $(75,0,130)$ — resampled to 256 entries. Keying
this off $r$ rather than raw height keeps the flower's core golden and
fades the petal tips toward indigo.

---

## 9. Stem and leaf geometry

Drawn directly in 2D screen space (not part of the 3-D surface math), so
they stay fixed regardless of bloom or camera.

**Stem** — a gentle sine-based S-curve from the flower's anchor down to
near the bottom of the frame, parameterized by $t \in [0,1]$:

$$
\text{stem}(t) = \big(\,x_{\text{top}} + A_{\text{sway}}\sin(\pi t),\;\; y_{\text{top}} + t\,(y_{\text{bottom}} - y_{\text{top}})\,\big)
$$

**Leaf** — an almond-shaped outline of length $L$ and max width $W$,
tapered with a half-sine profile, rotated by $\theta_{\text{leaf}}$ and
translated to an attach point $\mathbf{a}$ on the stem:

$$
w(u) = \frac{W}{2}\sin(\pi u), \qquad u \in [0,1]
$$

Boundary points $(uL,\, w(u))$ and $(uL,\, -w(u))$ are joined into a closed
polygon, rotated:

$$
\begin{pmatrix}x'\\y'\end{pmatrix} =
\begin{pmatrix}\cos\theta_{\text{leaf}} & -\sin\theta_{\text{leaf}}\\ \sin\theta_{\text{leaf}} & \cos\theta_{\text{leaf}}\end{pmatrix}
\begin{pmatrix}uL\\ \pm w(u)\end{pmatrix} + \mathbf{a}
$$

Each flower has two leaves attached at different points along the stem
($t = 0.42$ and $t = 0.62$), pointing in different directions.
