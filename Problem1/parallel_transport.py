import numpy as np
import matplotlib.pyplot as plt

# Cartesian to Spherical Coordinates
def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)  # polar angle (0 <= theta <= pi)
    phi = np.arctan2(y, x)    # azimuthal angle (-pi <= phi <= pi)
    return r, theta, phi

# Spherical to Cartesian Coordinates
def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

# Cartesian to Cylindrical Coordinates
def cartesian_to_cylindrical(x, y, z):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi, z

# Cylindrical to Cartesian Coordinates
def cylindrical_to_cartesian(rho, phi, z):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y, z

# Basis Transformation: Cartesian to Spherical
def cartesian_to_spherical_basis(theta, phi):
    # Convert the basis vectors from Cartesian to Spherical
    e_r = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    e_theta = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)])
    e_phi = np.array([-np.sin(phi), np.cos(phi), 0])
    return e_r, e_theta, e_phi

# Basis Transformation: Cartesian to Cylindrical
def cartesian_to_cylindrical_basis(phi):
    # Convert the basis vectors from Cartesian to Cylindrical
    e_rho = np.array([np.cos(phi), np.sin(phi), 0])
    e_phi = np.array([-np.sin(phi), np.cos(phi), 0])
    e_z = np.array([0, 0, 1])
    return e_rho, e_phi, e_z
pos = []
e_rvals = []
e_thetavals = []
e_phivals = []
points = []
ax = plt.figure().add_subplot(111, projection='3d')
for i in range(0, 360, 20):
    for j in range(0, 180, 20):
        x, y, z = spherical_to_cartesian(1, np.radians(j), np.radians(i))
        e_r, e_theta, e_phi = cartesian_to_spherical_basis(np.radians(j), np.radians(i))
        vec = [x, y, z]
        pos.append(vec)
        points.append(e_r)
        e_rvals.append(e_r)
        e_thetavals.append(e_theta)
        e_phivals.append(e_phi)

pos=np.array(pos)
e_rvals=np.array(e_rvals)
e_thetavals=np.array(e_thetavals)
e_phivals=np.array(e_phivals)
theta_vals = np.linspace(0, np.pi, 20)  # polar angle (0 to pi)
phi_vals = np.linspace(0, 2 * np.pi, 20)  # azimuthal angle (0 to 2pi)

theta, phi = np.meshgrid(theta_vals, phi_vals)

x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

ax.plot_surface(x, y, z, color='blue', alpha=0.3, edgecolor='lightgrey')
ax.quiver(pos[:,0], pos[:,1], pos[:,2], e_rvals[:,0], e_rvals[:,1], e_rvals[:,2], color='r', label='e_r', length=0.1)
ax.quiver(pos[:,0], pos[:,1], pos[:,2], e_thetavals[:,0], e_thetavals[:,1], e_thetavals[:,2], color='g', label='e_theta', length=0.1)
ax.quiver(pos[:,0], pos[:,1], pos[:,2], e_phivals[:,0], e_phivals[:,1], e_phivals[:,2], color='b', label='e_phi', length=0.1)
plt.show()
plt.savefig('spherical_basis.png')

def compute_local_coordinate_system(f, x_range, y_range, dx=0.01, dy=0.01):
    """
    Generate the local coordinate system on a given surface z = f(x, y).

    Parameters:
        f (function): A function f(x, y) representing the surface.
        x_range (tuple): The range of x values (xmin, xmax).
        y_range (tuple): The range of y values (ymin, ymax).
        dx (float): Step size in x-direction.
        dy (float): Step size in y-direction.

    Returns:
        X, Y, Z: Meshgrid coordinates of the surface.
        e_r: Normal vectors (extrinsic to the surface).
        e_x, e_y: Tangent vectors along x and y directions.
    """
    # Generate meshgrid for x and y values
    x_vals = np.arange(x_range[0], x_range[1], dx)
    y_vals = np.arange(y_range[0], y_range[1], dy)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Compute surface values
    Z = f(X, Y)
    
    # Compute gradients (partial derivatives) of the surface
    dZdx, dZdy = np.gradient(Z, dx, dy, axis=(1, 0))

    # Compute normal vector (extrinsic to the surface)
    normal_magnitude = np.sqrt(dZdx**2 + dZdy**2 + 1)
    e_r = np.stack((-dZdx / normal_magnitude, -dZdy / normal_magnitude, 1 / normal_magnitude), axis=-1)
    print(np.shape(np.ones_like(X)))
    print(np.shape(dZdx))
    # Compute tangent vectors (intrinsic to the surface)
    e_x = np.dstack((np.ones_like(X), np.zeros_like(X), dZdx)) / np.sqrt(1 + dZdx**2)[:, :, np.newaxis]
    e_y = np.dstack((np.zeros_like(Y), np.ones_like(Y), dZdy)) / np.sqrt(1 + dZdy**2)[:, :, np.newaxis]
    return X, Y, Z, e_r, e_x, e_y

# Example surface function z = f(x, y)
def surface_function(x, y):
    return np.sin(x) * np.cos(y)

# Generate local coordinate system for the surface
X, Y, Z, e_r, e_x, e_y = compute_local_coordinate_system(surface_function, (-2, 2), (-2, 2), 0.1, 0.1)

# Plot the surface and local coordinate system
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.4, edgecolor='k')

# Plot the local coordinate vectors
stride = 5  # Controls vector density
ax.quiver(X[::stride, ::stride], Y[::stride, ::stride], Z[::stride, ::stride],
          e_r[::stride, ::stride, 0], e_r[::stride, ::stride, 1], e_r[::stride, ::stride, 2], 
          color='r', length=0.2, label='Normal Vectors')

ax.quiver(X[::stride, ::stride], Y[::stride, ::stride], Z[::stride, ::stride],
          e_x[::stride, ::stride, 0], e_x[::stride, ::stride, 1], e_x[::stride, ::stride, 2], 
          color='b', length=0.2, label='Tangent e_x')

ax.quiver(X[::stride, ::stride], Y[::stride, ::stride], Z[::stride, ::stride],
          e_y[::stride, ::stride, 0], e_y[::stride, ::stride, 1], e_y[::stride, ::stride, 2], 
          color='g', length=0.2, label='Tangent e_y')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Local Coordinate System on Surface')

plt.legend()
plt.show()
plt.savefig('local_coordinate_system.png')