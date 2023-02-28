import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import  LinearLocator
import matplotlib.animation as anim


# # Opgave 1.2)
# mu1 = 175.5; mu2 = 162.9
# sigma = 6.7
# var = sigma**2

# x = np.arange(140,200,0.1)

# f1 = 1/(np.sqrt(2*np.pi)*sigma) *np.exp(-1/2 * 1/var * np.power(x-mu1,2))
# f2 = 1/(np.sqrt(2*np.pi)*sigma) *np.exp(-1/2 * 1/var * np.power(x-mu2,2))


# fig1 = plt.figure()

# plt.plot(x,f1, label = "Male")
# plt.plot(x,f2, label = "Female")
# plt.title("Gaussian Distributions")
# plt.xlabel("Height [cm]")
# plt.ylabel("Probability Density")
# plt.legend()
# plt.show()

# # opgave 1.3)
# fig2 = plt.figure()

# male = f1/f2
# female = f2/f1
# probmale = male/(male+1)
# probfemale = female/(female+1)

# plt.plot(x,probmale, label = "Male")
# plt.plot(x,probfemale, label = "Female")
# plt.title(" Distribution Given Height")
# plt.xlabel("Height [cm]")
# plt.ylabel("Probability Distribution")
# plt.ylim([0,1])
# plt.legend()
# plt.grid()
# plt.show()

# # Opgave 1.4)
# # svaret er (mu1 + mu2)/2
# print(f"Critical point: {(mu1+mu2)/2}")




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
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
plt.rcParams['text.usetex'] = True

x = np.arange(-6,6,0.05)
x1, x2 = np.meshgrid(x, x)
# Make data.
covar = sigma1*sigma2; rho = 2/3


def g_cor(rho):
    return 1/(2*np.pi) * 1/(sigma1*sigma2) * 1/np.sqrt(1 - rho**2) * np.exp(-1/2 * 1/(1 - rho**2) * ( 1/var1*np.power(x1 - mu1,2) - 2*rho/covar*(x1 - mu1)*(x2 - mu2) + 1/var2*np.power(x2 - mu2,2)))
g_cors = g_cor(0)

surf = ax.plot_surface(x1,x2,g_cors, cmap = "gist_ncar", linewidth=0)
ax.zaxis.set_major_locator(LinearLocator(6))
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_zlabel(r'$P(x_1,x_2)$')

ax.set_zlim([0,0.03])

# Make an animation
name = "Covariant Gaussian Distribution"
fps = 24
movie_writer = anim.writers['ffmpeg']
metadata = dict(title=name)
movie = movie_writer(fps=fps, metadata=metadata)

with movie.saving(fig, name + ".mp4", 100):
    for i in range(100):
        fig.suptitle(r'$\rho$ = ' + f'{0.01*i:0.2f}', fontsize=16)
        new_data = g_cor(0.01*i)
        surf.remove()
        surf = ax.plot_surface(x1,x2,new_data, cmap = "gist_ncar", linewidth=0)
        plt.draw()
        movie.grab_frame()


new_data = g_cor(2/3)
surf.remove()
surf = ax.plot_surface(x1,x2,new_data, cmap = "gist_ncar", linewidth=0)
plt.show()