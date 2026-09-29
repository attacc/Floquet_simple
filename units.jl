#
# Module for Tight-binding code for monolayer hexagonal boron nitride
# Claudio Attaccalite (2023)
#

module Units

const ha2ev     =27.211396132
const CORE_CONST=2.418884326505
const fs2aut    =100.0/CORE_CONST
const SPEED_of_LIGHT=137.03599911
const ANG2BOHR_FQ = 1.889726125


const AU2J   =4.3597482e-18 # Ha = AU2J Joule
const J2AU   =1.0/AU2J     # J  = J2AU Ha
 
const AU2M  =5.2917720859e-11  # Bohr = AU2M m
const M2AU  =1.0/AU2M        # m    = M2AU Bohr
 
const AU2SEC =2.418884326505e-17  # Tau = AU2SEC sec
const SEC2AU =1.0/AU2SEC           # sec = SEC2AU Tau

const kWCMm22AU=1e7*J2AU/(M2AU^2*SEC2AU)  # kW/cm^2 = kWCMm22AU * AU
const AU2KWCMm2=1.0/kWCMm22AU                   # AU      = AU2KWCMm2 kW/cm^2

const EAMPAU2VM=5.14220826E11      # Unit of electric field strength

export ha2ev,fs2aut,kWCMm22AU,AU2KWCMm2,SPEED_of_LIGHT,EAMPAU2VM,ANG2BOHR_FQ

end
