import numpy as np
import matplotlib.pyplot as plt

# Part A
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
# Part B, C
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

# Part D

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

# Part E

# Constants
theta0 = np.pi / 5  # Initial position near the north pole (theta_0)
phi0 = 0            # Initial azimuthal angle
theta_end = np.pi / 2  # End position at the equator
num_steps = 10  # Number of steps for discretization

# Generate theta values along the path
theta_vals = np.linspace(theta0, theta_end, num_steps)
phi_vals = np.zeros_like(theta_vals)  # Since φ = 0

# Initial vector components
alpha = 1.0  # Arbitrary choice for θ-component
beta = 0.5   # Arbitrary choice for φ-component
n0_magnitude = np.sqrt(alpha**2 + (beta * np.sin(theta0))**2)

# Initialize vector components along the transport path
n_theta = np.full_like(theta_vals, alpha)
n_phi = beta * np.sin(theta0) / np.sin(theta_vals)  # Maintain normalization

# Ensure normalization
norm = np.sqrt(n_theta**2 + (n_phi * np.sin(theta_vals))**2)
n_theta /= norm
n_phi /= norm

# Compute positions on the sphere
x_vals, y_vals, z_vals = spherical_to_cartesian(1, theta_vals, phi_vals)

# Compute transported vector components in Cartesian coordinates
e_theta = np.vstack([np.cos(theta_vals), np.zeros_like(theta_vals), -np.sin(theta_vals)]).T
e_phi = np.vstack([np.zeros_like(phi_vals), np.ones_like(phi_vals), np.zeros_like(phi_vals)]).T

transported_vectors = n_theta[:, np.newaxis] * e_theta + n_phi[:, np.newaxis] * e_phi

# Plotting the unit sphere and transported vectors
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot unit sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 50)
X = np.outer(np.cos(u), np.sin(v))
Y = np.outer(np.sin(u), np.sin(v))
Z = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(X, Y, Z, color='c', alpha=0.3)

# Plot transport path
ax.plot(x_vals, y_vals, z_vals, 'r-', label='Transport Path')

# Plot vectors
ax.quiver(x_vals, y_vals, z_vals,
          transported_vectors[:, 0], transported_vectors[:, 1], transported_vectors[:, 2],
          color='b', length=0.2, normalize=True, label='Transported Vector')

# Labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Parallel Transport of a Vector on the Sphere')
plt.legend()
plt.show()
plt.savefig('parallel_transport.png')
# Part F

# Constants
theta0 = np.pi / 4  # Fixed θ0
phi_start = 0       # Starting φ
phi_end = 2 * np.pi # Ending φ
num_steps = 20     # Number of steps for discretization

# Generate φ values along the path
phi_vals = np.linspace(phi_start, phi_end, num_steps)
theta_vals = np.full_like(phi_vals, theta0)  # θ remains constant

# Initial vector components
alpha = 1.0  # Arbitrary choice for θ-component
beta = 0.5   # Arbitrary choice for φ-component
n0_magnitude = np.sqrt(alpha**2 + beta**2)  # Magnitude of initial vector

# Compute the initial vector in spherical basis
n_theta = alpha
n_phi = beta

# Normalize the vector
n_theta /= n0_magnitude
n_phi /= n0_magnitude

# Initialize arrays for transported vector components
transported_vectors_theta = []
transported_vectors_phi = []

# During transport, n_θ remains constant, but n_φ may change to maintain normalization
for phi in phi_vals:
    transported_vectors_theta.append(n_theta)
    transported_vectors_phi.append(n_phi)

# Convert to Cartesian coordinates for visualization
def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

# Compute positions on the sphere
x_vals, y_vals, z_vals = spherical_to_cartesian(1, theta_vals, phi_vals)

# Compute transported vector components in Cartesian coordinates
e_theta = np.vstack([np.cos(theta_vals) * np.cos(phi_vals),
                     np.cos(theta_vals) * np.sin(phi_vals),
                     -np.sin(theta_vals)]).T
e_phi = np.vstack([-np.sin(phi_vals),
                   np.cos(phi_vals),
                   np.zeros_like(phi_vals)]).T

# Combine components to get the transported vectors in Cartesian coordinates
transported_vectors = (np.array(transported_vectors_theta)[:, np.newaxis] * e_theta +
                       np.array(transported_vectors_phi)[:, np.newaxis] * e_phi)

# Plotting the unit sphere and transported vectors
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, color='c', alpha=0.3)
# Plot transport path
ax.plot(x_vals, y_vals, z_vals, 'r-', label='Transport Path')

# Plot vectors along the path
ax.quiver(x_vals, y_vals, z_vals,
          transported_vectors[:, 0], transported_vectors[:, 1], transported_vectors[:, 2],
          color='b', length=0.2, normalize=True, label='Transported Vector')

# Labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Parallel Transport of a Vector Around a Constant Latitude')
plt.legend()
plt.show()
plt.savefig('parallel_transport_constant_latitude.png')

# Part G

# Constants
num_theta0 = 100  # Number of θ0 samples
phi_start = 0
phi_end = 2 * np.pi
num_steps = 200

# Generate θ0 values
theta0_vals = np.linspace(0, np.pi, num_theta0)

# Inner product storage
inner_products = []

# Function to calculate transported vector
def parallel_transport_vector(theta0):
    # Initial vector (normalized in spherical basis)
    alpha = 1.0
    beta = 0.5
    n0_magnitude = np.sqrt(alpha**2 + beta**2 * np.sin(theta0)**2)
    n_theta = alpha / n0_magnitude
    n_phi = beta / n0_magnitude

    # Spherical basis vectors
    phi_vals = np.linspace(phi_start, phi_end, num_steps)
    theta_vals = np.full_like(phi_vals, theta0)

    e_theta = np.vstack([np.cos(theta_vals) * np.cos(phi_vals),
                         np.cos(theta_vals) * np.sin(phi_vals),
                         -np.sin(theta_vals)]).T
    e_phi = np.vstack([-np.sin(phi_vals),
                       np.cos(phi_vals),
                       np.zeros_like(phi_vals)]).T

    # Initial vector in Cartesian coordinates
    initial_vector = n_theta * e_theta[0] + n_phi * e_phi[0]

    # Final vector after parallel transport (at φ = 2π)
    final_vector = n_theta * e_theta[-1] + n_phi * e_phi[-1]

    # Calculate inner product between initial and final vectors
    inner_product = np.dot(initial_vector, final_vector)
    return inner_product

# Compute inner product for each θ0
for theta0 in theta0_vals:
    inner_products.append(parallel_transport_vector(theta0))

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(theta0_vals, inner_products, label="Inner Product (Holonomy Strength)")
plt.axhline(1, color='r', linestyle='--', label="No Holonomy (Reference)")
plt.xlabel(r"$\theta_0$ (rad)", fontsize=14)
plt.ylabel("Inner Product", fontsize=14)
plt.title("Holonomy Strength vs Initial Latitude ($\\theta_0$)", fontsize=16)
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('holonomy_strength.png')

