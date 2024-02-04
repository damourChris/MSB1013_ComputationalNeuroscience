# Haemodynamics


The following equations represent mass balance equations for blood volume and deoxyhaemoglobin
into the cortical column as the venous compartment expands (like a balloon). These
equations describe the physiological aspect of large amounts of blood inflow increasing locally to
support small increases in oxygen metabolism rates and an associated change in oxygen extraction
fraction. {{footnote: Martin Havlicek et al. “Physiologically informed dynamic causal modeling of fMRI data”.
In: NeuroImage 122 (Nov. 2015), pp. 355–372. issn: 1053-8119. doi: 10 . 1016 / J .
NEUROIMAGE.2015.07.078.}}

\\[\dot{V}=\frac{f_{in}(t) − f_{out}(t)}{\tau} \\]

\\[\dot{q} =\frac{
      f_{in}(t)\frac{E_f}{E_0} 
    − f_{out}(t)\frac{q(t)}{V(t)}
    }{τ} \\]

where: 

\\[ \dot{E_{f}} = 1 − \frac{1 − E_0}{1 − f_{in}(t)} \\]


\\[ \dot{f_{out}} = V (t) \frac{1}{\alpha} + \tau_{vs} \\]

- \\(V \\) = blood volume (normalized with respect to resting state value)
-  \\(Q \\) = deoxyhemoglobin levels (normalized with respect to resting state value)
- \\(f_{out}\\) = outflow of blood from the model
-  \\(\tau \\) = time constant
-  \\(E_f\\) = oxygen extraction fraction
-  \\(E_0\\) = baseline oxygen extraction fraction
-  \\(τ_{vs}\\) = viscoelastic time constant