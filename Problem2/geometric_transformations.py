import numpy as np
import matplotlib.pyplot as plt

def stereographic_projection(x, y, z):
    """ Maps points from the unit sphere to the xy-plane using stereographic projection """
    X = x / (1 - z)
    Y = y / (1 - z)
    return X, Y

# Generate unit sphere points
theta = np.linspace(0, np.pi, 30, endpoint=False)  # Latitude
phi = np.linspace(0, 2*np.pi, 60)
theta = theta[1:]  # Longitude
'''phi, theta = np.meshgrid(phi, theta)

x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)
# Apply stereographic projection
X, Y = stereographic_projection(x, y, z)

# Plot sphere and stereographic projection
fig, ax = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})

ax[0].plot_surface(x, y, z, color='c', alpha=0.6, edgecolor='k')
ax[0].set_title("Unit Sphere")

ax[1] = plt.subplot(122)
ax[1].scatter(X, Y, c=theta.flatten(), cmap='viridis', s=2)
ax[1].set_title("Stereographic Projection")

plt.show()
plt.savefig('stereographic_projection.png')'''

# Generate grid points
phi_grid, theta_grid = np.meshgrid(phi, theta)
x = np.sin(theta_grid) * np.cos(phi_grid)
y = np.sin(theta_grid) * np.sin(phi_grid)
z = np.cos(theta_grid)

# Define two families of curves: meridians (constant theta) and parallels (constant phi)
phi_curves = np.linspace(0, 2*np.pi, 8)  # Select a few meridians
theta_curves = np.linspace(0.2, np.pi - 0.2, 6)  # Select a few parallels

# Store original and projected curves
meridian_curves = []
parallel_curves = []
projected_meridians = []
projected_parallels = []

# Compute curves before and after projection
for p in phi_curves:
    x_m = np.sin(theta) * np.cos(p)
    y_m = np.sin(theta) * np.sin(p)
    z_m = np.cos(theta)
    meridian_curves.append((x_m, y_m, z_m))
    projected_meridians.append(stereographic_projection(x_m, y_m, z_m))

for t in theta_curves:
    x_p = np.sin(t) * np.cos(phi)
    y_p = np.sin(t) * np.sin(phi)
    z_p = np.full_like(phi, np.cos(t))
    parallel_curves.append((x_p, y_p, z_p))
    projected_parallels.append(stereographic_projection(x_p, y_p, z_p))

# Plot original curves on the unit sphere
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)

ax1.set_title("Curves on Unit Sphere")
ax2.set_title("Stereographic Projection")

# Plot meridians and parallels on sphere
for (x_m, y_m, z_m) in meridian_curves:
    ax1.plot(x_m, y_m, z_m, 'b')

for (x_p, y_p, z_p) in parallel_curves:
    ax1.plot(x_p, y_p, z_p, 'r')

ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")

# Plot projected curves
for (X_m, Y_m) in projected_meridians:
    ax2.plot(X_m, Y_m, 'b')

for (X_p, Y_p) in projected_parallels:
    ax2.plot(X_p, Y_p, 'r')

ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_aspect('equal')
ax2.set_xlim(-5,5)
ax2.set_ylim(-5,5)

plt.show()
plt.savefig('stereographic_projection_curves.png')

# 1. Equator: fixed theta = 0, vary phi
equator_x = np.cos(phi)
equator_y = np.sin(phi)
equator_z = np.zeros_like(phi)

meridian_curves = []
projected_meridians = []
for ind in [-1,1]:
    x_m = np.sin(theta) * ind
    y_m = np.zeros_like(theta)
    z_m = np.cos(theta)
    meridian_curves.append((x_m, y_m, z_m))
    projected_meridians.append(stereographic_projection(x_m, y_m, z_m))

# 3. Tilted great circle: rotated around an axis (45-degree tilt)
tilt_curves = []
projected_tilts = []
for tilt in [np.pi/8,np.pi/4,3*np.pi/8, np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8]:
    tilt_x = np.cos(phi)
    tilt_y = np.sin(phi) * np.cos(tilt)
    tilt_z = np.sin(phi) * np.sin(tilt)
    tilt_curves.append((tilt_x, tilt_y, tilt_z))
    projected_tilts.append(stereographic_projection(tilt_x, tilt_y, tilt_z))

# Apply stereographic projection
equator_X, equator_Y = stereographic_projection(equator_x, equator_y, equator_z)
tilt_X, tilt_Y = stereographic_projection(tilt_x, tilt_y, tilt_z)

# Plot sphere with great circles
fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(equator_x, equator_y, equator_z, 'r', label="Equator")
# Plot meridians and parallels on sphere
for (x_m, y_m, z_m) in meridian_curves:
    ax1.plot(x_m, y_m, z_m, 'g', label="Meridian")
for (x_m, y_m, z_m) in tilt_curves:
    ax1.plot(x_m, y_m, z_m, 'b', label="Tilts")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")
ax1.set_title("Great Circles on the Sphere")
ax1.legend()

# Plot stereographic projection of great circles
ax2 = fig.add_subplot(122)
ax2.plot(equator_X, equator_Y, 'r', label="Equator")
# Plot projected curves
for (X_m, Y_m) in projected_meridians:
    ax2.plot(X_m, Y_m, 'g', label="Meridian")
for (X_m, Y_m) in projected_tilts:
    ax2.plot(X_m, Y_m, 'b', label="Tilts")
ax2.set_xlabel("X'")
ax2.set_ylabel("Y'")
ax2.set_xlim(-5,5)
ax2.set_ylim(-5,5)
ax2.set_title("Stereographic Projection of Great Circles")
ax2.legend()

plt.show()
plt.savefig('stereographic_projection_great_circles.png')

# Constants
theta0 = np.pi / 4  # Initial latitude (fixed θ)
phi_start = 0       # Start longitude
phi_end = 2 * np.pi # Complete loop
num_steps = 50      # Discretization steps

# Generate φ values along the path
phi_vals = np.linspace(phi_start, phi_end, num_steps)
theta_vals = np.full_like(phi_vals, theta0)  # Constant latitude

# Initial vector components
alpha = 1.0  # Arbitrary choice for θ-component
beta = 0.2   # Arbitrary choice for φ-component
n0_magnitude = np.sqrt(alpha**2 + beta**2)  # Normalize

# Compute initial vector in spherical basis
n_theta = alpha / n0_magnitude
n_phi = beta / n0_magnitude

# Compute positions on the sphere
def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

x_vals, y_vals, z_vals = spherical_to_cartesian(1, theta_vals, phi_vals)

# Compute transported vector components in Cartesian coordinates
e_theta = np.vstack([np.cos(theta_vals) * np.cos(phi_vals),
                     np.cos(theta_vals) * np.sin(phi_vals),
                     -np.sin(theta_vals)]).T
e_phi = np.vstack([-np.sin(phi_vals),
                   np.cos(phi_vals),
                   np.zeros_like(phi_vals)]).T

# Compute transported vectors
transported_vectors = n_theta * e_theta + n_phi * e_phi

# Apply stereographic projection
X_vals, Y_vals = stereographic_projection(x_vals, y_vals, z_vals)

# Project transported vectors
TX, TY = stereographic_projection(x_vals + transported_vectors[:, 0],
                                  y_vals + transported_vectors[:, 1],
                                  z_vals + transported_vectors[:, 2])

# Compute projected vectors
transported_proj_vectors = np.array([TX - X_vals, TY - Y_vals]).T

# Plot sphere and stereographic projection
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot sphere with transport path
ax1 = fig.add_subplot(121, projection='3d')
# Generate sphere surface for visualization
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 25)
X_sphere = np.outer(np.cos(u), np.sin(v))
Y_sphere = np.outer(np.sin(u), np.sin(v))
Z_sphere = np.outer(np.ones_like(u), np.cos(v))

# Corrected surface plot
ax1.plot_surface(X_sphere, Y_sphere, Z_sphere, color='c', alpha=0.1, edgecolor='k')
ax1.plot(x_vals, y_vals, z_vals, 'r-', label='Transport Path')
ax1.quiver(x_vals, y_vals, z_vals, transported_vectors[:, 0], transported_vectors[:, 1], transported_vectors[:, 2], color='b', length=0.2, label='Transported Vectors')
ax1.set_title("Parallel Transport on the Sphere")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")
ax1.legend()

# Plot stereographic projection
ax2 = fig.add_subplot(122)
ax2.plot(X_vals, Y_vals, 'r-', label='Projected Path')
ax2.quiver(X_vals, Y_vals, transported_proj_vectors[:, 0], transported_proj_vectors[:, 1], color='b', scale=10, label='Projected Vectors')
ax2.set_xlabel("X'")
ax2.set_ylabel("Y'")
ax2.set_title("Stereographic Projection of Parallel Transport")
ax2.legend()

plt.show()
plt.savefig('parallel_transport_sphere.png')

# Define a set of points on the unit sphere
theta_vals = np.linspace(np.pi/2, np.pi, 20, endpoint=False)  # Varying latitude (from equator to pole)
phi_vals = np.linspace(0, 2*np.pi, 20)    # Varying longitude
theta_vals = theta_vals[1:]  # Exclude poles

# Initialize storage for inner products
inner_product_before = []
inner_product_after = []

# Loop through different points on the sphere
for theta in theta_vals:
    for phi in phi_vals:
        # Compute Cartesian coordinates of the point on the sphere
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        # Compute two tangent vectors at this point
        #v1 = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)])  # Along latitude
        #v2 = np.array([-np.sin(phi), np.cos(phi), 0])  # Along longitude
        # Generate two random tangent vectors at this point
        v1 = np.random.randn(3)
        v2 = np.random.randn(3)

        # Ensure they are tangent to the sphere (orthogonal to the normal)
        normal = np.array([x, y, z])
        v1 -= np.dot(v1, normal) * normal  # Project v1 onto the tangent plane
        v2 -= np.dot(v2, normal) * normal  # Project v2 onto the tangent plane

        # Normalize the vectors
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)
        # Compute inner product before projection
        inner_product_before.append(np.dot(v1, v2))

        # Apply stereographic projection to the point
        X, Y = stereographic_projection(x, y, z)

        # Compute stereographic projection of tangent vectors using the Jacobian
        J = np.array([[1/(1 - z), 0], [0, 1/(1 - z)]])  # Jacobian of stereographic projection
        v1_proj = J @ np.array([v1[0], v1[1]])
        v2_proj = J @ np.array([v2[0], v2[1]])

        # Compute inner product after projection
        inner_product_after.append(np.dot(v1_proj, v2_proj))

# Convert lists to NumPy arrays for plotting
inner_product_before = np.array(inner_product_before)
inner_product_after = np.array(inner_product_after)

# Plot inner products before and after projection
plt.figure(figsize=(8, 6))
plt.scatter(inner_product_before, inner_product_after, color='b', label="Projected Inner Products")
plt.plot([-1, 1], [-1, 1], 'r--', label="y = x (Perfect Preservation)")
plt.xlabel("Inner Product Before Projection")
plt.ylabel("Inner Product After Projection")
plt.title("Comparison of Inner Products Before and After Projection")
plt.legend()
plt.grid()
plt.show()
plt.savefig('inner_product_comparison.png')

# Constants for parallel transport
theta0 = np.pi / 3  # Initial latitude (fixed θ)
phi_vals = np.linspace(0, 2 * np.pi, 100)  # Full loop

# Compute points on the sphere for the loop
x_vals = np.sin(theta0) * np.cos(phi_vals)
y_vals = np.sin(theta0) * np.sin(phi_vals)
z_vals = np.cos(theta0) * np.ones_like(phi_vals)

# Initial vector at φ = 0
n_theta = 1.0  # Initial θ-component
n_phi = 0.5    # Initial φ-component

# Normalize the vector
n0_magnitude = np.sqrt(n_theta**2 + n_phi**2)
n_theta /= n0_magnitude
n_phi /= n0_magnitude

# Transport vector components
transported_vectors_theta = []
transported_vectors_phi = []

for phi in phi_vals:
    transported_vectors_theta.append(n_theta)
    transported_vectors_phi.append(n_phi)

# Convert to Cartesian coordinates for visualization
def spherical_to_cartesian_vector(n_theta, n_phi, theta, phi):
    """ Convert vector components in spherical basis to Cartesian basis. """
    e_theta = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)])
    e_phi = np.array([-np.sin(phi), np.cos(phi), 0])
    return n_theta * e_theta + n_phi * e_phi

# Compute transported vectors in Cartesian coordinates
transported_vectors = np.array([
    spherical_to_cartesian_vector(n_t, n_p, theta0, phi)
    for n_t, n_p, phi in zip(transported_vectors_theta, transported_vectors_phi, phi_vals)
])

# Apply stereographic projection
X_vals, Y_vals = stereographic_projection(x_vals, y_vals, z_vals)

# Project transported vectors
TX, TY = stereographic_projection(x_vals + transported_vectors[:, 0],
                                  y_vals + transported_vectors[:, 1],
                                  z_vals + transported_vectors[:, 2])

# Compute projected vectors
transported_proj_vectors = np.array([TX - X_vals, TY - Y_vals]).T

# Compute holonomy angle before and after projection
initial_vector = transported_vectors[0]
final_vector = transported_vectors[-1]

# Compute angle between initial and final transported vectors
holonomy_angle_before = np.arccos(np.dot(initial_vector, final_vector) / 
                                  (np.linalg.norm(initial_vector) * np.linalg.norm(final_vector)))

# Compute holonomy in projected space
initial_vector_proj = transported_proj_vectors[0]
final_vector_proj = transported_proj_vectors[-1]
holonomy_angle_after = np.arccos(np.dot(initial_vector_proj, final_vector_proj) / 
                                 (np.linalg.norm(initial_vector_proj) * np.linalg.norm(final_vector_proj)))

# Plot sphere and stereographic projection
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot transport path on sphere
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X_sphere, Y_sphere, Z_sphere, color='c', alpha=0.1, edgecolor='k')
ax1.plot(x_vals, y_vals, z_vals, 'r-', label='Transport Path')
ax1.quiver(x_vals, y_vals, z_vals, transported_vectors[:, 0], transported_vectors[:, 1], transported_vectors[:, 2], color='b', length=0.2, label='Transported Vectors')
ax1.set_title("Parallel Transport on the Sphere")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")
ax1.legend()

# Plot stereographic projection
ax2 = fig.add_subplot(122)
ax2.plot(X_vals, Y_vals, 'r-', label='Projected Path')
ax2.quiver(X_vals, Y_vals, transported_proj_vectors[:, 0], transported_proj_vectors[:, 1], color='b', scale=10, label='Projected Vectors')
ax2.set_xlabel("X'")
ax2.set_ylabel("Y'")
ax2.set_title("Stereographic Projection of Parallel Transport")
ax2.legend()

plt.show()
plt.savefig('parallel_transport_holonomy.png')
# Print holonomy comparison
print(f"Holonomy angle before projection: {np.degrees(holonomy_angle_before):.2f} degrees")
print(f"Holonomy angle after projection: {np.degrees(holonomy_angle_after):.2f} degrees")
