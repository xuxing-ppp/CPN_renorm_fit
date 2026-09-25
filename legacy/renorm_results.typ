#import "@local/myconf:0.1.0":*

#show: myconf.with(doc_title: [Renormalization Results])

= Renormalization Ansatz

== $"mod"=0$

$ S =& -2 N beta sum_l Re(z^dagger_i e^(i a_l) z_j) - N beta_1 sum_l |z_i^dagger z_j|^2 #linebreak()
&+ alpha/2 sum_p (dif a + 2 pi s)^2 -alpha_1sum_p cos(dif a)  $

== $"mod"=1$

$ S =& -2 N beta sum_l Re(z^dagger_i e^(i a_l) z_j) - "sgn"(beta_1)log(I_0(2 N beta_1 |z_i^dagger z_j|)) #linebreak()
&+ alpha/2 sum_p (dif a + 2 pi s)^2 -alpha_1sum_p cos(dif a)  $

= Renormalization Schemes

The coarse-lattice couplings are extracted by the `full_renorm/` pipeline:
- $beta$ and $alpha + alpha_1$ — 2plaq $a$-distribution fit (it sees the single plaquette coupling $alpha_"eff" = alpha + alpha_1$);
- $beta_1$ — observable matching, two independent estimates: magnetic susceptibility ("mag") and correlation length ("corr");
- $alpha$ — either the 1plaq vortex-distribution fit ("1plaq") or the topological-susceptibility match ("topo").

The cross product of the two $beta_1$ estimates with the two $alpha$ estimates gives $2 times 2 = 4$ renormalized-coupling sets per chain; $beta$ and $alpha + alpha_1$ are common to all four, and $alpha_1 = (alpha + alpha_1) - alpha$. (The topo $alpha$ is matched per $beta_1$, so "mag, topo" uses topo(mag) and "corr, topo" uses topo(corr).)

Two blocking types appear: *$z$-renorm* ($alpha_f = 0$ fine model; $z$-vortex / $z$-connection) and *$U$-renorm* ($alpha_f > 0$ fine model, which the integer $s$-vortex requires; $s$-vortex / $U$-connection).

== 2plaq-Renorm $arrow.r$ $beta$, $alpha + alpha_1$

- $a$-renorm
- $z$-renorm

== Obs-Renorm $arrow.r$ $beta_1$ and $alpha$ ("topo")

- $beta_1$: magnetic susceptibility (mag) / correlation length (corr) matching.
- $alpha$ ("topo"): match the integer-winding topological susceptibility $chi_t$ between the fine lattice (scaled by $L^2$) and the coarse lattice.

== 1plaq-Renorm $arrow.r$ $alpha$ ("1plaq")

Fit distribution of $s_p$ on the coarse lattice.
- $s$-renorm: corresponds to $a$-renorm in 2plaq, i.e.
$ q^((f))_(p^((f))) = s_p^((f)) + (dif a^((f)))_p / (2 pi) $
$ q^((c))_(p^((c))) = s_p^((c)) + (dif a^((c)))_p / (2 pi) $
$ q^((c))_(p^((c))) = sum_(p^((f)) in p^((c))) q^((f))_(p^((f))) $
- $z$-renorm:
$ q^((f))_(p^((f))): "Geometric definition" $
$ q^((c))_(p^((c))) = s_p^((c)) + (dif a^((c)))_p / (2 pi) $
$ q^((c))_(p^((c))) = sum_(p^((f)) in p^((c))) q^((f))_(p^((f))) $

= Renormalization Results

Each single-stage block is a renorm chain at blocking factor $L$ ("$4,z$", "$8,U$", etc.); "$4,z$ & $2,U$" / "$2,U$ & $2,U$" rows are two-stage chains, reported with the *same* $(beta_1, alpha)$ method in both stages. The four combos share $beta$ and $alpha + alpha_1$ (2plaq fit) and differ in $beta_1$ (mag/corr) and $alpha$ (1plaq/topo). An em-dash ($---$) marks a fit that did not converge.

A *composition check* tests whether blocking by $L_1 L_2$ in one stage agrees with blocking by $L_1$ then $L_2$: $8,z$ vs $4,z$ & $2,U$ ($z$-renorm, $beta_f = 1.5$) and $4,U$ vs $2,U$ & $2,U$ ($U$-renorm, $beta_f = 1.0$). The relative-error tables report $(p_("two-stage") - p_("one-stage")) / p_("one-stage")$ per parameter.

== $z$-renorm ($alpha_f = 0$): $beta_f = 1.5$

=== $"mod"=0$

#figure(
table(align: center, columns: 6, stroke: none,
[Renorm], table.vline(), [$beta$], [$beta_1$], [$alpha$], [$alpha_1$], [$alpha + alpha_1$],
table.hline(),
[$4,z$, mag, 1plaq], [5.149], [-5.672], [0.358], [0.671], [1.029],
[$4,z$, mag, topo], [5.149], [-5.672], [0.329], [0.700], [1.029],
[$4,z$, corr, 1plaq], [5.149], [-5.308], [0.358], [0.671], [1.029],
[$4,z$, corr, topo], [5.149], [-5.308], [0.317], [0.712], [1.029],
table.hline(),
[$8,z$, mag, 1plaq], [1.578], [-1.032], [0.332], [0.125], [0.457],
[$8,z$, mag, topo], [1.578], [-1.032], [0.282], [0.175], [0.457],
[$8,z$, corr, 1plaq], [1.578], [-0.811], [0.332], [0.125], [0.457],
[$8,z$, corr, topo], [1.578], [-0.811], [---], [---], [0.457],
table.hline(),
[$4,z$ & $2,U$, mag, 1plaq], [1.725], [-1.386], [0.476], [0.043], [0.519],
[$4,z$ & $2,U$, mag, topo], [1.725], [-1.386], [0.279], [0.240], [0.519],
[$4,z$ & $2,U$, corr, 1plaq], [1.925], [-1.289], [0.399], [0.118], [0.517],
[$4,z$ & $2,U$, corr, topo], [1.925], [-1.289], [0.255], [0.262], [0.517],
)
)

Composition check: $4,z$ & $2,U$ vs $8,z$ (one-stage $8,z$ is the reference):

#figure(
table(align: center, columns: 6, stroke: none,
[Combo], table.vline(), [$Delta beta\/beta$], [$Delta beta_1\/beta_1$], [$Delta alpha\/alpha$], [$Delta alpha_1\/alpha_1$], [$Delta(alpha+alpha_1)/(alpha+alpha_1)$],
table.hline(),
[mag, 1plaq], [+9.3%], [+34.3%], [+43.4%], [-65.6%], [+13.6%],
[mag, topo], [+9.3%], [+34.3%], [-1.1%], [+37.1%], [+13.6%],
[corr, 1plaq], [+22.0%], [+58.9%], [+20.2%], [-5.6%], [+13.1%],
[corr, topo], [+22.0%], [+58.9%], [---], [---], [+13.1%],
)
)

=== $"mod"=1$

#figure(
table(align: center, columns: 6, stroke: none,
[Renorm], table.vline(), [$beta$], [$beta_1$], [$alpha$], [$alpha_1$], [$alpha + alpha_1$],
table.hline(),
[$4,z$, mag, 1plaq], [5.149], [-4.484], [0.358], [0.671], [1.029],
[$4,z$, mag, topo], [5.149], [-4.484], [0.336], [0.693], [1.029],
[$4,z$, corr, 1plaq], [5.149], [-4.295], [0.358], [0.671], [1.029],
[$4,z$, corr, topo], [5.149], [-4.295], [0.319], [0.710], [1.029],
table.hline(),
[$8,z$, mag, 1plaq], [1.578], [-0.971], [0.332], [0.125], [0.457],
[$8,z$, mag, topo], [1.578], [-0.971], [0.282], [0.175], [0.457],
[$8,z$, corr, 1plaq], [1.578], [-0.831], [0.332], [0.125], [0.457],
[$8,z$, corr, topo], [1.578], [-0.831], [---], [---], [0.457],
table.hline(),
[$4,z$ & $2,U$, mag, 1plaq], [1.809], [-1.276], [0.431], [0.055], [0.486],
[$4,z$ & $2,U$, mag, topo], [1.809], [-1.275], [0.283], [0.203], [0.486],
[$4,z$ & $2,U$, corr, 1plaq], [2.013], [-1.223], [0.467], [0.017], [0.484],
[$4,z$ & $2,U$, corr, topo], [2.013], [-1.223], [0.256], [0.228], [0.484],
)
)

Composition check: $4,z$ & $2,U$ vs $8,z$:

#figure(
table(align: center, columns: 6, stroke: none,
[Combo], table.vline(), [$Delta beta\/beta$], [$Delta beta_1\/beta_1$], [$Delta alpha\/alpha$], [$Delta alpha_1\/alpha_1$], [$Delta(alpha+alpha_1)/(alpha+alpha_1)$],
table.hline(),
[mag, 1plaq], [+14.6%], [+31.4%], [+29.8%], [-56.0%], [+6.3%],
[mag, topo], [+14.6%], [+31.3%], [+0.4%], [+16.0%], [+6.3%],
[corr, 1plaq], [+27.6%], [+47.2%], [+40.7%], [-86.4%], [+5.9%],
[corr, topo], [+27.6%], [+47.2%], [---], [---], [+5.9%],
)
)

== $U$-renorm ($alpha_f > 0$)

=== $beta_f = 1.0$, $"mod"=0$: composition test $4,U$ vs $2,U$ & $2,U$

Single-stage $4,U$ ($alpha_f = 0.1, 0.3, 0.5$), the block-2 $2,U$ ($alpha_f = 0.3, 0.5$), and the two-stage $2,U$ & $2,U$ ($alpha_f = 0.3, 0.5$).

#figure(
table(align: center, columns: 6, stroke: none,
[Renorm], table.vline(), [$beta$], [$beta_1$], [$alpha$], [$alpha_1$], [$alpha + alpha_1$],
table.hline(),
[$4,U$, $a_f=0.1$, mag, 1plaq], [0.245], [-0.145], [0.007], [-0.001], [0.006],
[$4,U$, $a_f=0.1$, mag, topo], [0.245], [-0.145], [0.007], [-0.001], [0.006],
[$4,U$, $a_f=0.1$, corr, 1plaq], [0.245], [0.419], [0.007], [-0.001], [0.006],
[$4,U$, $a_f=0.1$, corr, topo], [0.245], [0.419], [0.007], [-0.001], [0.006],
table.hline(),
[$4,U$, $a_f=0.3$, mag, 1plaq], [0.273], [0.000], [0.048], [-0.040], [0.008],
[$4,U$, $a_f=0.3$, mag, topo], [0.273], [0.000], [0.044], [-0.036], [0.008],
[$4,U$, $a_f=0.3$, corr, 1plaq], [0.273], [0.442], [0.048], [-0.040], [0.008],
[$4,U$, $a_f=0.3$, corr, topo], [0.273], [0.442], [0.045], [-0.037], [0.008],
[$2,U$, $a_f=0.3$, mag, 1plaq], [0.553], [0.325], [0.164], [-0.102], [0.062],
[$2,U$, $a_f=0.3$, mag, topo], [0.553], [0.325], [0.168], [-0.106], [0.062],
[$2,U$, $a_f=0.3$, corr, 1plaq], [0.553], [0.436], [0.164], [-0.102], [0.062],
[$2,U$, $a_f=0.3$, corr, topo], [0.553], [0.436], [0.168], [-0.106], [0.062],
[$2,U$ & $2,U$, $a_f=0.3$, mag, 1plaq], [0.263], [-0.002], [0.044], [-0.041], [0.003],
[$2,U$ & $2,U$, $a_f=0.3$, mag, topo], [0.263], [-0.002], [0.045], [-0.042], [0.003],
[$2,U$ & $2,U$, $a_f=0.3$, corr, 1plaq], [0.273], [0.444], [0.045], [-0.043], [0.002],
[$2,U$ & $2,U$, $a_f=0.3$, corr, topo], [0.273], [0.444], [0.045], [-0.043], [0.002],
table.hline(),
[$4,U$, $a_f=0.5$, mag, 1plaq], [0.300], [0.127], [0.119], [-0.099], [0.020],
[$4,U$, $a_f=0.5$, mag, topo], [0.300], [0.127], [0.148], [-0.128], [0.020],
[$4,U$, $a_f=0.5$, corr, 1plaq], [0.300], [0.499], [0.119], [-0.099], [0.020],
[$4,U$, $a_f=0.5$, corr, topo], [0.300], [0.499], [0.148], [-0.128], [0.020],
[$2,U$, $a_f=0.5$, mag, 1plaq], [0.576], [0.360], [0.319], [-0.203], [0.116],
[$2,U$, $a_f=0.5$, mag, topo], [0.576], [0.360], [0.516], [-0.400], [0.116],
[$2,U$, $a_f=0.5$, corr, 1plaq], [0.576], [0.454], [0.319], [-0.203], [0.116],
[$2,U$, $a_f=0.5$, corr, topo], [0.576], [0.454], [0.485], [-0.369], [0.116],
[$2,U$ & $2,U$, $a_f=0.5$, mag, 1plaq], [0.287], [0.120], [0.093], [-0.088], [0.005],
[$2,U$ & $2,U$, $a_f=0.5$, mag, topo], [0.287], [0.120], [0.148], [-0.143], [0.005],
[$2,U$ & $2,U$, $a_f=0.5$, corr, 1plaq], [0.296], [0.492], [0.095], [-0.088], [0.007],
[$2,U$ & $2,U$, $a_f=0.5$, corr, topo], [0.296], [0.492], [0.146], [-0.139], [0.007],
)
)

Composition check: $2,U$ & $2,U$ vs $4,U$ (one-stage $4,U$ is the reference):

#figure(
table(align: center, columns: 7, stroke: none,
[$a_f$], [Combo], table.vline(), [$Delta beta\/beta$], [$Delta beta_1\/beta_1$], [$Delta alpha\/alpha$], [$Delta alpha_1\/alpha_1$], [$Delta(alpha+alpha_1)/(alpha+alpha_1)$],
table.hline(),
[0.3], [mag, 1plaq], [-3.7%], [---], [-8.3%], [+2.5%], [-62.5%],
[0.3], [mag, topo], [-3.7%], [---], [+2.3%], [+16.7%], [-62.5%],
[0.3], [corr, 1plaq], [+0.0%], [+0.5%], [-6.3%], [+7.5%], [-75.0%],
[0.3], [corr, topo], [+0.0%], [+0.5%], [+0.0%], [+16.2%], [-75.0%],
table.hline(),
[0.5], [mag, 1plaq], [-4.3%], [-5.5%], [-21.8%], [-11.1%], [-75.0%],
[0.5], [mag, topo], [-4.3%], [-5.5%], [+0.0%], [+11.7%], [-75.0%],
[0.5], [corr, 1plaq], [-1.3%], [-1.4%], [-20.2%], [-11.1%], [-65.0%],
[0.5], [corr, topo], [-1.3%], [-1.4%], [-1.4%], [+8.6%], [-65.0%],
)
)

=== $beta_f = 1.0$, $"mod"=1$: composition test $4,U$ vs $2,U$ & $2,U$ ($alpha_f = 0.5$)

Only $alpha_f = 0.5$ was run for $"mod"=1$. As in $"mod"=0$, $beta$, the 1plaq $alpha$ and $alpha + alpha_1$ coincide with the $"mod"=0$ values (e.g. $4,U$: $beta = 0.300$, $alpha + alpha_1 = 0.020$); only $beta_1$ and the topo $alpha$ differ.

#figure(
table(align: center, columns: 6, stroke: none,
[Renorm], table.vline(), [$beta$], [$beta_1$], [$alpha$], [$alpha_1$], [$alpha + alpha_1$],
table.hline(),
[$4,U$, $a_f=0.5$, mag, 1plaq], [0.300], [0.254], [0.119], [-0.099], [0.020],
[$4,U$, $a_f=0.5$, mag, topo], [0.300], [0.254], [0.143], [-0.123], [0.020],
[$4,U$, $a_f=0.5$, corr, 1plaq], [0.300], [0.577], [0.119], [-0.099], [0.020],
[$4,U$, $a_f=0.5$, corr, topo], [0.300], [0.577], [0.144], [-0.124], [0.020],
table.hline(),
[$2,U$, $a_f=0.5$, mag, 1plaq], [0.576], [0.470], [0.319], [-0.203], [0.116],
[$2,U$, $a_f=0.5$, mag, topo], [0.576], [0.470], [0.525], [-0.409], [0.116],
[$2,U$, $a_f=0.5$, corr, 1plaq], [0.576], [0.539], [0.319], [-0.203], [0.116],
[$2,U$, $a_f=0.5$, corr, topo], [0.576], [0.539], [0.495], [-0.379], [0.116],
table.hline(),
[$2,U$ & $2,U$, $a_f=0.5$, mag, 1plaq], [0.287], [0.266], [0.092], [-0.086], [0.006],
[$2,U$ & $2,U$, $a_f=0.5$, mag, topo], [0.287], [0.266], [0.148], [-0.142], [0.006],
[$2,U$ & $2,U$, $a_f=0.5$, corr, 1plaq], [0.296], [0.565], [0.098], [-0.091], [0.007],
[$2,U$ & $2,U$, $a_f=0.5$, corr, topo], [0.296], [0.565], [0.148], [-0.141], [0.007],
)
)

Composition check: $2,U$ & $2,U$ vs $4,U$ (one-stage $4,U$ is the reference):

#figure(
table(align: center, columns: 7, stroke: none,
[$a_f$], [Combo], table.vline(), [$Delta beta\/beta$], [$Delta beta_1\/beta_1$], [$Delta alpha\/alpha$], [$Delta alpha_1\/alpha_1$], [$Delta(alpha+alpha_1)/(alpha+alpha_1)$],
table.hline(),
[0.5], [mag, 1plaq], [-4.3%], [+4.7%], [-22.7%], [-13.1%], [-70.0%],
[0.5], [mag, topo], [-4.3%], [+4.7%], [+3.5%], [+15.4%], [-70.0%],
[0.5], [corr, 1plaq], [-1.3%], [-2.1%], [-17.6%], [-8.1%], [-65.0%],
[0.5], [corr, topo], [-1.3%], [-2.1%], [+2.8%], [+13.7%], [-65.0%],
)
)

=== $beta_f = 2.0$: single-stage $8,U$

#figure(
table(align: center, columns: 6, stroke: none,
[Renorm], table.vline(), [$beta$], [$beta_1$], [$alpha$], [$alpha_1$], [$alpha + alpha_1$],
table.hline(),
[$"mod"=0$, $a_f=0.5$, mag, 1plaq], [0.365], [1.201], [0.184], [-0.163], [0.021],
[$"mod"=0$, $a_f=0.5$, mag, topo], [0.365], [1.201], [0.393], [-0.372], [0.021],
[$"mod"=0$, $a_f=0.5$, corr, 1plaq], [0.365], [1.631], [0.184], [-0.163], [0.021],
[$"mod"=0$, $a_f=0.5$, corr, topo], [0.365], [1.631], [0.381], [-0.360], [0.021],
table.hline(),
[$"mod"=0$, $a_f=1.0$, mag, 1plaq], [0.393], [1.171], [0.274], [-0.242], [0.032],
[$"mod"=0$, $a_f=1.0$, mag, topo], [0.393], [1.171], [1.510], [-1.478], [0.032],
[$"mod"=0$, $a_f=1.0$, corr, 1plaq], [0.393], [1.558], [0.274], [-0.242], [0.032],
[$"mod"=0$, $a_f=1.0$, corr, topo], [0.393], [1.558], [1.363], [-1.331], [0.032],
table.hline(),
[$"mod"=1$, $a_f=0.5$, mag, 1plaq], [0.365], [1.109], [0.184], [-0.163], [0.021],
[$"mod"=1$, $a_f=0.5$, mag, topo], [0.365], [1.109], [0.395], [-0.374], [0.021],
[$"mod"=1$, $a_f=0.5$, corr, 1plaq], [0.365], [1.464], [0.184], [-0.163], [0.021],
[$"mod"=1$, $a_f=0.5$, corr, topo], [0.365], [1.464], [0.386], [-0.365], [0.021],
table.hline(),
[$"mod"=1$, $a_f=1.0$, mag, 1plaq], [0.393], [1.093], [0.274], [-0.242], [0.032],
[$"mod"=1$, $a_f=1.0$, mag, topo], [0.393], [1.093], [1.140], [-1.108], [0.032],
[$"mod"=1$, $a_f=1.0$, corr, 1plaq], [0.393], [1.462], [0.274], [-0.242], [0.032],
[$"mod"=1$, $a_f=1.0$, corr, topo], [0.393], [1.462], [1.060], [-1.028], [0.032],
)
)

= Notes

- *Source.* All values are from `full_renorm/results/` (`full_renorm_summary.json` per case). $beta$, $beta_1$, $alpha + alpha_1$ are common to a chain (2plaq fit); the four combos differ only in $beta_1$ (mag/corr) and $alpha$ (1plaq/topo).
- *$"mod"=0$ vs $"mod"=1$.* With $beta_(1 f) = 0$ the 2plaq fit ($beta$, $alpha + alpha_1$) and the 1plaq $alpha$ are identical across mods; only $beta_1$ and the topo $alpha$ differ.
- *Failed fits.* The $8,z$ "corr, topo" $alpha$ did not match in either mod (topo susceptibility outside the coarse $alpha$ scan at the corr $beta_1$); those cells are dashed, and so is its composition-check $alpha$ row. $beta_1$ is unaffected.
- *Composition checks.* For $z$-renorm the two-stage $4,z$ & $2,U$ overestimates $beta$ by $approx 9 dash 28%$ and $beta_1$ by $approx 31 dash 59%$ relative to one-stage $8,z$; $alpha + alpha_1$ agrees within $approx 6 dash 14%$. For $U$-renorm ($2,U$ & $2,U$ vs $4,U$) $beta$ agrees within $approx 5%$ and $alpha$ within $approx 22%$, but $alpha + alpha_1$ differs by $approx 60 dash 75%$ because both numbers are tiny ($< 0.02$), so the relative error is inflated.
- *Relative-error caveats.* $alpha_1$ and $alpha + alpha_1$ relative errors are noisy when the reference is small (e.g. $8,z$ "corr, topo" $alpha$; $U$ $alpha + alpha_1 tilde.eq 0$; $4,U$, $a_f=0.3$ mag $beta_1 tilde.eq 0$) — these are dashed or large-magnitude and should be read as "the two fits disagree", not as a precise discrepancy.
- *$U$-renorm $alpha$ inconsistency.* In the $U$-renorm regime the 1plaq/topo $alpha$ frequently exceeds the 2plaq $alpha + alpha_1$ (e.g. $8,U$, $a_f=1.0$: topo $alpha tilde.eq 1.1 dash 1.5$ vs $alpha + alpha_1 = 0.032$), so $alpha_1$ is negative; the topo match at $a_f = 1.0$ in particular looks unconverged.
