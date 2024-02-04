# BOLD response

Finally, all of the previous equations are combined into a final formula that describes the rate of
change on the BOLD signal, depending on the relative amount of deoxyhaemoglobin and blood
volume in the cortical layer. This final step is evidently dependent also on the strength of the
magnetic field applied. The used parameters are shown in table 2.

\\[
\dot{BOLD} = V0 · (k_1(1 − q(t)) + k_2(1 − \frac{q(t)}{V (t)} ) + k_3(1 − V (t))) 
\\]

\\(k_1, k_2, k_3 \\) are magnetic field strength-dependent parameters, where:

\\[k_1 = 4.3 · ν_0 · E_0 · T_E\\]
\\[k_2 = ϵ · ρ_0 · E_0 · T_E\\]
\\[k_3 = 1 − ε\\]

where:
- \\(ν_0\\) = field-dependent frequency offset at the surface of a blood vessel
- \\(T_E\\) = MRI echo time
- \\(ε\\) = ratio of intra to extravascular fMRI signal contribution
- \\(ρ_0\\) = sensitivity of vascular signal relaxation rate with regards to changes in oxygen saturation
  
However, LBR is observable only in ultra-high field MRI (7+ Tesla), where ε becomes negligible
[22]. Therefore, we use:

\\[k_1 = 4.3 · ν_0 · E_0 · T_E \\]
\\[k_2 = 0 \\]
\\[k_3 = 1 \\]

The BOLD response these equations provide does not start at zero but requires some time to
settle into its baseline. To account for this, all values before the beginning of the stimulus are
set to the value of the BOLD response at the start of the stimulus start. This way, it is ensured
that the BOLD signal does not deflect the stimulus yet but has time to settle into its baseline.


| Symbol | Description | Value |
| --- | --- | ---|
| ϕ | rate constant | 0.6 |
| Φ | rate constant | 1.5 |
| χ | rate constant | 0.6 |
| E0 | baseline oxygen ejection fraction | 0.4 |
| τ | time constant | 2 |
| α | Grubb’s exponent | 0.32 |
| τvs | viscoelastic time constant | 4 |
| V0 | baseline blood volume | 4 |
| ν0 | frequency offset at the blood vessel surface | 188.1 | |
| TE | echo time | 0.028 |
| ε | ration of fMRI signal contribution | 0 |
| r0 | sensitivity | - |
<div style="display: flex; justify-content: center">

**Table 2**: Parameter Values Used in the BOLD Model.
</div>