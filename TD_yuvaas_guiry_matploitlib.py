import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D


# Exercise 1: Simple Polynomial Plot
x1 = np.linspace(-10, 10, 100)
y1 = 2*x1**3 - 5*x1**2 + 3*x1 - 7

plt.figure(figsize=(10, 6))
plt.plot(x1, y1, 'b-', label='y = 2x³ - 5x² + 3x - 7')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Function Plot')
plt.legend()
plt.grid(True)
plt.show()


# Exercise 2: Exponential and Logarithmic Plot
x2 = np.linspace(0.1, 10, 500)
y2_exp = np.exp(x2)
y2_log = np.log(x2)

plt.figure(figsize=(10, 6))
plt.plot(x2, y2_exp, 'r-', label='y = exp(x)')
plt.plot(x2, y2_log, 'g--', label='y = log(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Exponential and Logarithmic Plot')
plt.legend()
plt.grid(True)
plt.savefig('exponential_logarithmic_plot.png', dpi=100)
plt.show()


# Exercise 3: Figure and Subplots
x3_1 = np.linspace(-2*np.pi, 2*np.pi, 500)
y3_tan = np.tan(x3_1)
y3_arctan = np.arctan(x3_1)

x3_2 = np.linspace(-2, 2, 500)
y3_sinh = np.sinh(x3_2)
y3_cosh = np.cosh(x3_2)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# First subplot
axes[0].plot(x3_1, y3_tan, 'm-', label='y = tan(x)')
axes[0].plot(x3_1, y3_arctan, 'c--', label='y = arctan(x)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('Tan and Arctan Functions')
axes[0].legend()
axes[0].grid(True)

# Second subplot
axes[1].plot(x3_2, y3_sinh, 'b-', label='y = sinh(x)')
axes[1].plot(x3_2, y3_cosh, 'r--', label='y = cosh(x)')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title('Sinh and Cosh Functions')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()



# Exercise 4: Plot a Histogram


# Generate an array of 1000 values from a normal distribution
n = np.random.randn(1000)

# Plot the histogram with 30 bins
plt.figure(figsize=(10, 6))
plt.hist(n, bins=30, color='purple', edgecolor='black', alpha=0.7)

# Customization
plt.title('Histogram of Normally Distributed Values')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.xlim([n.min(), n.max()])
plt.grid(axis='y', alpha=0.75)

# Show the plot
plt.show()



# Exercise 5: Scatter Plot
x5 = np.random.uniform(-10, 10, 500)
y5 = np.sin(x5) + np.random.normal(0, 0.2, 500)
sizes = np.abs(y5) * 100
colors = y5

plt.figure(figsize=(10, 6))
plt.scatter(x5, y5, s=sizes, c=colors, cmap='viridis', alpha=0.7, edgecolors='black')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot of y = sin(x) + noise')
plt.grid(True)
plt.xticks([])
plt.yticks([])
plt.savefig('scatter_plot.pdf')
plt.show()

# Exercise 6: Contour Plot
x6 = np.linspace(-5, 5, 200)
y6 = np.linspace(-5, 5, 200)
X6, Y6 = np.meshgrid(x6, y6)
Z6 = np.sin(np.sqrt(X6**2 + Y6**2))

plt.figure(figsize=(8, 6))
contour = plt.contourf(X6, Y6, Z6, cmap='coolwarm')
plt.colorbar(contour)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Contour Plot of sin(sqrt(x² + y²))')
plt.show()

# Exercise 7: 3D Surface Plot
x7 = np.arange(-5, 5.25, 0.25)
y7 = np.arange(-5, 5.25, 0.25)
X7, Y7 = np.meshgrid(x7, y7)
Z7 = np.cos(np.sqrt(X7**2 + Y7**2))

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X7, Y7, Z7, cmap='plasma')
fig.colorbar(surf)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Surface Plot of cos(sqrt(x² + y²))')
ax.view_init(elev=30, azim=45)
plt.show()

# Exercise 8: Line and Marker Styles
x8 = np.linspace(-2, 2, 10)
y8_1 = x8**2
y8_2 = x8**3
y8_3 = x8**4

plt.figure(figsize=(8, 6))
plt.plot(x8, y8_1, 'ro-', label='y = x²')
plt.plot(x8, y8_2, 'bs--', label='y = x³')
plt.plot(x8, y8_3, 'g^-.', label='y = x⁴')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Different Line and Marker Styles')
plt.legend()
plt.grid(True)
plt.show()

# Exercise 9: Logarithmic Scale
x9 = np.logspace(0, 2, 50)
y9_1 = 2**x9
y9_2 = np.log2(x9)

plt.figure(figsize=(12, 6))
plt.plot(x9, y9_1, 'r-', label='y = 2^x')
plt.plot(x9, y9_2, 'b--', label='y = log2(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Logarithmic Scale Plot')
plt.yscale('log')
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.show()



# Exercise 10: Changing Viewing Angle
fig, axes = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(12, 6))
surf1 = axes[0].plot_surface(X7, Y7, Z7, cmap='plasma')
surf2 = axes[1].plot_surface(X7, Y7, Z7, cmap='plasma')

axes[0].view_init(elev=30, azim=30)
axes[0].set_title("Viewing Angle 30° Azimuth")

axes[1].view_init(elev=60, azim=60)
axes[1].set_title("Viewing Angle 60° Azimuth")

plt.show()

# Exercise 11: 3D Wireframe Plot
X11, Y11 = np.meshgrid(np.arange(-5, 5.25, 0.25), np.arange(-5, 5.25, 0.25))
Z11 = np.sin(X11) * np.cos(Y11)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X11, Y11, Z11, color='black')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Wireframe Plot of sin(X) * cos(Y)')
ax.view_init(elev=40, azim=30)
plt.show()

# Exercise 12: 3D Contour Plot
X12, Y12 = np.meshgrid(np.arange(-5, 5.25, 0.25), np.arange(-5, 5.25, 0.25))
Z12 = np.exp(-0.1 * (X12**2 + Y12**2))

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
contour3D = ax.contour3D(X12, Y12, Z12, 50, cmap='coolwarm')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Contour Plot of exp(-0.1 * (X² + Y²))')
plt.show()

# Exercise 13: 3D Parametric Plot
t13 = np.linspace(0, 2*np.pi, 100)
X13 = np.sin(t13)
Y13 = np.cos(t13)
Z13 = t13

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(X13, Y13, Z13, color='purple', linewidth=2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Parametric Plot')
plt.show()



# Exercise 14: 3D Bar Plot
x14 = np.linspace(-5, 5, 10)
y14 = np.linspace(-5, 5, 10)
X14, Y14 = np.meshgrid(x14, y14)
Z14 = np.exp(-0.1 * (X14**2 + Y14**2))

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.bar3d(X14.ravel(), Y14.ravel(), np.zeros_like(Z14.ravel()), 1, 1, Z14.ravel(), color='orange')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Bar Plot')
plt.show()

# Exercise 15: 3D Vector Field
x15 = np.linspace(-5, 5, 10)
y15 = np.linspace(-5, 5, 10)
z15 = np.linspace(-5, 5, 10)
X15, Y15, Z15 = np.meshgrid(x15, y15, z15)
U15 = -Y15
V15 = X15
W15 = Z15

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.quiver(X15, Y15, Z15, U15, V15, W15, color='blue', length=0.5, normalize=True)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Vector Field')
plt.show()

# Exercise 16: 3D Scatter Plot
x16 = np.random.randn(100)
y16 = np.random.randn(100)
z16 = np.random.randn(100)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x16, y16, z16, c=z16, cmap='coolwarm')
fig.colorbar(sc, ax=ax, label="Z Value")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Scatter Plot')
plt.show()

# Exercise 17: 3D Line Plot
t17 = np.linspace(0, 4*np.pi, 100)
X17 = np.sin(t17)
Y17 = np.cos(t17)
Z17 = t17

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(X17, Y17, Z17, color='purple', linewidth=2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Line Plot')
plt.show()


# Exercise 18: 3D Filled Contour Plot
x18 = np.arange(-5, 5.1, 0.1)
y18 = np.arange(-5, 5.1, 0.1)
X18, Y18 = np.meshgrid(x18, y18)
Z18 = np.sin(np.sqrt(X18**2 + Y18**2))

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
contour = ax.contourf(X18, Y18, Z18, 50, cmap='plasma')
fig.colorbar(contour, ax=ax)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Filled Contour Plot')
plt.show()

# Exercise 19: 3D Heatmap
x19 = np.linspace(-5, 5, 50)
y19 = np.linspace(-5, 5, 50)
X19, Y19 = np.meshgrid(x19, y19)
Z19 = np.sin(np.sqrt(X19**2 + Y19**2))

plt.figure(figsize=(8, 6))
plt.imshow(Z19, extent=[-5, 5, -5, 5], origin='lower', cmap='coolwarm', aspect='auto')
plt.colorbar(label='Z Value')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('3D Heatmap')
plt.show()

# Exercise 20: 3D Density Plot
x20 = np.random.randn(1000)
y20 = np.random.randn(1000)
z20 = np.random.randn(1000)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x20, y20, z20, alpha=0.3, color='blue')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Density Plot')
plt.show()
