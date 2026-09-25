#
# Module for Tight-binding code for monolayer hexagonal boron nitride
# Claudio Attaccalite (2023)
#

module Units

ha2ev     =27.211396132
CORE_CONST=2.418884326505
fs2aut    =100.0/CORE_CONST
SPEED_of_LIGHT=137.03599911

AU2J   =4.3597482e-18 # Ha = AU2J Joule
J2AU   =1.0/AU2J     # J  = J2AU Ha
 
AU2M  =5.2917720859e-11  # Bohr = AU2M m
M2AU  =1.0/AU2M        # m    = M2AU Bohr
 
AU2SEC =2.418884326505e-17  # Tau = AU2SEC sec
SEC2AU =1.0/AU2SEC           # sec = SEC2AU Tau

kWCMm22AU=1e7*J2AU/(M2AU^2*SEC2AU)  # kW/cm^2 = kWCMm22AU * AU
AU2KWCMm2=1.0/kWCMm22AU                   # AU      = AU2KWCMm2 kW/cm^2

EAMPAU2VM=5.14220826E11      # Unit of electric field strength

export ha2ev,fs2aut,kWCMm22AU,AU2KWCMm2,SPEED_of_LIGHT,EAMPAU2VM

end
