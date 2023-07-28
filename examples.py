import QNMsolver
from SpaceTimeDefinition import defineSchw, defineKerrRFrac, defineKerrThetaFrac, defineTaubNUT, defineTaubNUT2

"""
Schwarzschild Spacetime
"""

# Schwarzschild quasinormal modes
rPaper = [0.967288,0.927701,0.861088,0.787726]
iPaper = [-0.193518,-0.591208,-1.01712,-1.47619]

l=2 # Degree of spherical harmonic
solver1 = QNMsolver.QNMsolver(defineSchw(2))
solver1.setResolution([0,3,0.005])
#solver1.addComparaison([rPaper],[iPaper],["Schwarzschild quasinormal modes"]) #Uncomment to add the Schwarzschild comparaison (for l=2)
solver1.plot(f"Calculated Schwarzschild quasinormal modes (l={l})")

"""
Taub-NUT Spacetime
"""

l=2 # NUT Charge
C=6 # Casimir
solver2 = QNMsolver.QNMsolver(defineTaubNUT(l,C))
solver2.setResolution([0,4,0.01])
solver2.plot(f"Calculated Taub-NUT Quasinormal modes (l={l}, C={C})")


solver1.show()
solver2.show()