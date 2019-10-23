COMMENT
	calcium accumulation into a volume of area*depth next to the
	membrane with a decay (time constant tau) to resting level
	given by the global calcium variable cai0_ca_ion
	Modified to include a resting current (irest) and peak value
	(cmax)
	i is a dummy current needed to force a BREAKPOINT
ENDCOMMENT

COMMENT
        added parameter gamma to model the percentage of free calcium,
        similarly to what is done in the CaDynamics_E2 mechanism, which
        is itself a modified version of Destexhe et al., 1994.
        Daniele Linaro - October 2019
ENDCOMMENT

NEURON {
	SUFFIX cacum
	USEION ca READ ica WRITE cai
	NONSPECIFIC_CURRENT i
	RANGE depth, tau, cai0, cmax, gamma
}

UNITS {
	(mM) = (milli/liter)
	(mA) = (milliamp)
	F = (faraday) (coulombs)
}

PARAMETER {
	depth = 0.1 (um)	: assume volume = area*depth
	irest = 0  (mA/cm2)		: to be initialized in hoc	
	tau = 100 (ms)
	cai0 = 50e-6 (mM)	: Requires explicit use in INITIAL
			: block for it to take precedence over cai0_ca_ion
			: Do not forget to initialize in hoc if different
			: from this default.
	gamma = 0.05 : percent of free calcium (not buffered)
}

ASSIGNED {
	ica (mA/cm2)
	cmax
	i  	 (mA/cm2)
}

STATE {
	cai (mM)
}

INITIAL {
	cai = cai0
	:irest = ica
	cmax=cai
}

BREAKPOINT {
	SOLVE integrate METHOD derivimplicit
	if (cai>cmax) {cmax=cai}
	i=0
}

DERIVATIVE integrate {
	cai' = (1e4) * (gamma*(irest-ica)/(2*F*depth)) + (cai0 - cai)/tau
}
