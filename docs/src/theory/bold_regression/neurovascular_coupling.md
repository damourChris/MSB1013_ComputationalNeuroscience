# Neurovascular Coupling
Some of the neural signal information is lost in the neurovascular coupling, thus acting like a lowpass
filter. This filtered signal can be interpreted as the vasoactive signal, which then influences
the blood inflow level. The inflow of blood then affects the stretch of the blood vessels, imposing
feedback on the vasoactive signal and having the neurovascular coupling behaviour reflect the
properties of a damped oscillator. {{footnote: Martin Havlicek et al. “Physiologically informed dynamic causal modeling of fMRI data”.
In: NeuroImage 122 (Nov. 2015), pp. 355–372. issn: 1053-8119. doi: 10 . 1016 / J .
NEUROIMAGE.2015.07.078.}}

\\[\dot{s} = X(t) − ϕ · s(t) \\]

where:

- \\(s\\) = vasoactive signal
- \\(X\\) = neural signal (units)
- \\(ϕ\\) = rate constant (controlling response decay)

\\[f˙ = Φ · s(t) − χ · f_{in}(t) \\]
where:
- \\(f_in\\) = inflow of blood into the model (units?)
- \\(Φ, χ\\) = rate constant (controlling response decay)