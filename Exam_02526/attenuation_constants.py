import numpy as np
import matplotlib.pyplot as plt

data_iron = np.loadtxt("Exam_02526/iron_attenuation_data.txt")
energy_iron = data_iron[:,0] # MeV 
attenuation_iron = data_iron[:,1] # cm^2/g 

data_lead = np.loadtxt("Exam_02526/lead_attenuation_data.txt")
energy_lead = data_lead[:,0] # MeV 
attenuation_lead = data_lead[:,1] # cm^2/g 

data_oxygen = np.loadtxt("Exam_02526/oxygen_attenuation_data.txt")
energy_oxygen = data_oxygen[:,0] # MeV 
attenuation_oxygen = data_oxygen[:,1] # cm^2/g 

data_carbon = np.loadtxt("Exam_02526/carbon_attenuation_data.txt")
energy_carbon = data_carbon[:,0] # MeV 
attenuation_carbon = data_carbon[:,1] # cm^2/g 

data_hydrogen = np.loadtxt("Exam_02526/hydrogen_attenuation_data.txt")
energy_hydrogen = data_hydrogen[:,0] # MeV 
attenuation_hydrogen = data_hydrogen[:,1] # cm^2/g 

# Tree mass approximation from molar masses. 
# Hydrogen: 7%
# Carbon: 40% 
# Oxygen: 53% 

tree_attenuation = 0.07*attenuation_hydrogen + 0.4*attenuation_carbon + 0.53*attenuation_oxygen

# Constraining to plot from 10 to 200 keV 
energyplot_iron = energy_iron[10:21]
energyplot_lead = energy_lead[21:41]
energyplot_tree = energy_hydrogen[8:19]
attenuationplot_iron = attenuation_iron[10:21]
attenuationplot_lead = attenuation_lead[21:41]
attenuationplot_tree = tree_attenuation[8:19]

# Densities
# Steel density: 7.9 g/cm^3 
# Lead density: 11.3 g/cm^3 
# Tree density: 0.8 g/cm^3 

rho_steel = 7.9
rho_lead = 11.3
rho_tree = 0.8
attenuationplot_iron = attenuationplot_iron*rho_steel
attenuationplot_lead = attenuationplot_lead*rho_lead
attenuationplot_tree = attenuationplot_tree*rho_tree

plt.figure(figsize=(9,6))
plt.loglog(energyplot_iron,attenuationplot_iron)
plt.loglog(energyplot_lead,attenuationplot_lead)
plt.loglog(energyplot_tree,attenuationplot_tree)
plt.xlabel('Energy [MeV]')
plt.ylabel(r'Attenuation $\left[ \frac{1}{cm} \right]$ ')
plt.legend(['Iron','Lead','Tree-mass (approx)'])
plt.title('Attenuation constants for energy levels with densities')
plt.grid()
plt.savefig('Exam_02526/attenuation_plot_rho.png')
plt.show()

# 200 keV constants (1/cm)
# Tree = 0.1*1e-2 (cm to mm)
# Iron = 1.14*1e-2
# Lead = 11.3*1e-2

# 40 keV constants (1/cm)
# Tree = 0.20*1e-2 
# Iron = 29*1e-2 
# Lead = 164*1e-2 

quit()

plt.figure(figsize=(9,6))
plt.loglog(energyplot_iron,attenuationplot_iron)
plt.loglog(energyplot_lead,attenuationplot_lead)
plt.loglog(energyplot_tree,attenuationplot_tree)
plt.xlabel('Energy [MeV]')
plt.ylabel(r'Attenuation $\left[ \frac{cm^2}{g} \right]$ ')
plt.legend(['Iron','Lead','Tree-mass (approx)'])
plt.title('Attenuation constants for energy levels')
plt.grid()
plt.savefig('Exam_02526/attenuation_plot.png')
plt.show()
