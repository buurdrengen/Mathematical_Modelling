import numpy as np
import matplotlib.pyplot as plt


# Opgave 1.2)
mu1 = 175.5; mu2 = 162.9
sigma = 6.7
var = sigma**2

x = np.arange(140,200,0.1)

f1 = 1/(np.sqrt(2*np.pi)*sigma) *np.exp(-1/2 * 1/var * np.power(x-mu1,2))
f2 = 1/(np.sqrt(2*np.pi)*sigma) *np.exp(-1/2 * 1/var * np.power(x-mu2,2))


fig1 = plt.figure()

plt.plot(x,f1, label = "Male")
plt.plot(x,f2, label = "Female")
plt.title("Gaussian Distributions")
plt.xlabel("Height [cm]")
plt.ylabel("Probability Density")
plt.legend()
plt.show()

# opgave 1.3)
fig2 = plt.figure()

male = f1/f2
female = f2/f1
probmale = male/(male+1)
probfemale = female/(female+1)

plt.plot(x,probmale, label = "Male")
plt.plot(x,probfemale, label = "Female")
plt.title(" Distribution Given Height")
plt.xlabel("Height [cm]")
plt.ylabel("Probability Distribution")
plt.ylim([0,1])
plt.legend()
plt.grid()
plt.show()

# Opgave 1.4)
# svaret er (mu1 + mu2)/2
print(f"Critical point: {(mu1+mu2)/2}")




# Opgave 1.5)
ax = plt.figure().add_subplot(projection='3d')

# Make data.
x = np.arange(-5,5,0.05)
x1, x2 = np.meshgrid(x, x)

mu1 = 0; mu2 = 1
sigma1 = 2; sigma2 = 3
var1 = sigma1**2; var2 = sigma2**2

g = 1/(2*np.pi) * 1/(sigma1*sigma2) * np.exp(-1/2 * ( 1/var1*np.power(x1 - mu1,2) + 1/var2*np.power(x2 - mu2,2)))

ax.plot_surface(x1,x2,g, cmap = "gist_ncar", linewidth=0)
plt.show()


# Opgave 1.6)
ax = plt.figure().add_subplot(projection='3d')


x = np.arange(-6,6,0.05)
x1, x2 = np.meshgrid(x, x)
# Make data.
covar = sigma1*sigma2; rho = 2/3

g_cor = 1/(2*np.pi) * 1/(sigma1*sigma2) * 1/np.sqrt(1 - rho**2) * np.exp(-1/2 * 1/(1 - rho**2) * ( 1/var1*np.power(x1 - mu1,2) - 2*rho/covar*(x1 - mu1)*(x2 - mu2) + 1/var2*np.power(x2 - mu2,2)))

ax.plot_surface(x1,x2,g_cor, cmap = "gist_ncar", linewidth=0)
plt.show()