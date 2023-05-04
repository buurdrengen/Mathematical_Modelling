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
# print(energy_hydrogen[11])
# print(attenuation_hydrogen[11])
# print(energy_carbon[11])
# print(attenuation_carbon[11])
# print(energy_oxygen[11])
# print(attenuation_oxygen[11])
# Tree mass approximation from molar masses
# Beech: 
# Hydrogen: 6%
# Carbon: 49.5% 
# Oxygen: 44.5% 

# Fir:
# Hydrogen: 6% 
# Carbon: 50.5% 
# Oxygen: 43.5% 

tree_attenuation_beech = 0.06*attenuation_hydrogen + 0.495*attenuation_carbon + 0.445*attenuation_oxygen
tree_attenuation_fir = 0.06*attenuation_hydrogen + 0.505*attenuation_carbon + 0.435*attenuation_oxygen
# Constraining to plot from 10 to 200 keV 
energyplot_iron = energy_iron[10:21]
energyplot_lead = energy_lead[21:41]
energyplot_tree = energy_hydrogen[8:19]
attenuationplot_iron = attenuation_iron[10:21]
attenuationplot_lead = attenuation_lead[21:41]
attenuationplot_beech = tree_attenuation_beech[8:19]
attenuationplot_fir = tree_attenuation_fir[8:19]

# print("energy iron: ",energyplot_iron[3])
# print("attenuation iron:",attenuationplot_iron[3])
# print("energy lead: ",energyplot_lead[10])
# print("attenuation lead: ",attenuationplot_lead[10])
# print("energy tree: ",energyplot_tree[3])
# print("attenuation beech: ",attenuationplot_beech[3])
# print("attenuation fir: ",attenuationplot_fir[3])

# Densities
# Steel density: 7.9 g/cm^3 
# Lead density: 11.3 g/cm^3 
# Beech density: 0.8 g/cm^3
# Fir density: 0.53 g/cm^3 

rho_steel = 7.9
rho_lead = 11.3
rho_beech = 0.8
rho_fir = 0.53

attenuationplot_iron = attenuationplot_iron*rho_steel
attenuationplot_lead = attenuationplot_lead*rho_lead
attenuationplot_beech = attenuationplot_beech*rho_beech
attenuationplot_fir = attenuationplot_fir*rho_fir

# Attenutation constants for 30 keV 
# Iron 
print("Attenuation constant pure iron = ",attenuationplot_iron[3], "1/cm")
# Lead 
print("Attenuation constant pure lead = ",attenuationplot_lead[10], "1/cm")
# Beech 
print("Attenuation constant pure beech = ",attenuationplot_beech[3], "1/cm")
# Fir 
print("Attenuation constant pure fir = ",attenuationplot_fir[3], "1/cm")

plt.figure(figsize=(9,6))
plt.loglog(energyplot_iron,attenuationplot_iron)
plt.loglog(energyplot_lead,attenuationplot_lead)
plt.loglog(energyplot_tree,attenuationplot_beech)
plt.loglog(energyplot_tree,attenuationplot_fir)
plt.xlabel('Energy [MeV]')
plt.ylabel(r'Attenuation $\left[ \frac{1}{cm} \right]$ ')
plt.legend(['Iron','Lead','Beech', 'Fir'])
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
plt.loglog(energyplot_tree,attenuationplot_beech)
plt.xlabel('Energy [MeV]')
plt.ylabel(r'Attenuation $\left[ \frac{cm^2}{g} \right]$ ')
plt.legend(['Iron','Lead','Tree-mass (approx)'])
plt.title('Attenuation constants for energy levels')
plt.grid()
plt.savefig('Exam_02526/attenuation_plot.png')
plt.show()

