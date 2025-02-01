import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# -----------------------------
# 1. Load data and compute the convex hull and Delaunay triangulation
# -----------------------------
data = np.loadtxt('mesh.dat', skiprows=1)

def graham_scan(data):
    def polar_angle(p1, p2):
        return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    y_min_ind = np.argmin(data[:, 1])
    y_min = data[y_min_ind]
    polar_dict = {}
    for point in data:
        polar_dict[tuple(point)] = polar_angle(y_min, point)
    polar_dict = dict(sorted(polar_dict.items(), key=lambda item: item[1]))
    polar_points = [key for key in polar_dict]
    hull = []
    for point in polar_points:
        while len(hull) > 1 and np.cross(
            np.array(hull[-1]) - np.array(hull[-2]),
            np.array(point) - np.array(hull[-1])
        ) <= 0:
            hull.pop()
        hull.append(point)
    hull.append(y_min)
    return np.array(hull)

# Compute the convex hull using the Graham scan
hull = graham_scan(data)

# Compute the Delaunay triangulation (in the original 2D plane)
tri = Delaunay(data)

# Plot the triangulation and convex hull (unchanged)
plt.figure(figsize=(8, 6))
plt.triplot(data[:, 0], data[:, 1], tri.simplices, color='green', label='Delaunay Triangulation')
plt.plot(hull[:, 0], hull[:, 1], color='red', lw=2, label='Convex Hull')
plt.scatter(data[:, 0], data[:, 1], color='blue', s=20, label='Data Points')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Delaunay Triangulation and Convex Hull")
plt.legend()
plt.tight_layout()
plt.savefig("delaunay_triangulation.png")
plt.show()

# -----------------------------
# 2. Define the lifting map for the new surface: f(x,y) = x^2 + x*y + y^2
# -----------------------------
def lift_point(p):
    x, y = p
    return np.array([x, y, x**2 + x*y + y**2])

# -----------------------------
# 3. Compute 2D and 3D triangle areas and plot the area ratio heatmap
# -----------------------------
def triangle_area_2d(p1, p2, p3):
    return 0.5 * abs((p2[0] - p1[0])*(p3[1] - p1[1]) - (p3[0] - p1[0])*(p2[1] - p1[1]))

def triangle_area_3d(p1, p2, p3):
    p1_3d = lift_point(p1)
    p2_3d = lift_point(p2)
    p3_3d = lift_point(p3)
    v1 = p2_3d - p1_3d
    v2 = p3_3d - p1_3d
    return 0.5 * np.linalg.norm(np.cross(v1, v2))

area_ratios = []
for simplex in tri.simplices:
    p1, p2, p3 = data[simplex[0]], data[simplex[1]], data[simplex[2]]
    area2d = triangle_area_2d(p1, p2, p3)
    area3d = triangle_area_3d(p1, p2, p3)
    ratio = area3d / area2d if area2d != 0 else 0
    area_ratios.append(ratio)
area_ratios = np.array(area_ratios)

plt.figure(figsize=(8, 6))
tpc = plt.tripcolor(data[:, 0], data[:, 1], tri.simplices,
                    facecolors=area_ratios, edgecolors='k', cmap='viridis', clim=(0,50))
plt.colorbar(tpc, label='Area Ratio (3D / 2D)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Heatmap of Area Ratio after Lifting')
plt.tight_layout()
plt.savefig("area_ratio_heatmap2.png")
plt.show()

# -----------------------------
# 4. Compute the induced metric for the new surface
# -----------------------------
def induced_metric(x, y):
    # f_x = 2x + y, f_y = x + 2y
    g_xx = 1 + (2*x + y)**2
    g_xy = (2*x + y) * (x + 2*y)
    g_yy = 1 + (x + 2*y)**2
    return np.array([[g_xx, g_xy],
                     [g_xy, g_yy]])

# Example usage at (x,y) = (1,1)
x, y = 1, 1
g = induced_metric(x, y)
print("Induced metric at (1,1):\n", g)

# -----------------------------
# 5. Lift all points and compute their (analytic) surface normals
# -----------------------------
X = data[:, 0]
Y = data[:, 1]
Z = X**2 + X*Y + Y**2   # new surface
points3d = np.column_stack((X, Y, Z))

# For f(x,y) = x^2 + x*y + y^2, we have:
# f_x = 2x+y,  f_y = x+2y.
# So the unit normal is:
# n = (-(2x+y), -(x+2y), 1) / sqrt((2x+y)^2+(x+2y)^2+1)
denom = np.sqrt((2*X + Y)**2 + (X + 2*Y)**2 + 1)
normals = np.empty((len(X), 3))
normals[:, 0] = -(2*X + Y) / denom
normals[:, 1] = -(X + 2*Y) / denom
normals[:, 2] = 1 / denom

# -----------------------------
# 6. Compute face normals at the centroids of each triangle using the analytic formula
# -----------------------------
face_centroids = []
face_normals = []
for triangle in tri.simplices:
    i, j, k = triangle
    x0, y0, z0 = points3d[i]
    x1, y1, z1 = points3d[j]
    x2, y2, z2 = points3d[k]
    x_c = (x0 + x1 + x2) / 3.0
    y_c = (y0 + y1 + y2) / 3.0
    z_c = (z0 + z1 + z2) / 3.0
    centroid = np.array([x_c, y_c, z_c])
    face_centroids.append(centroid)
    # Compute analytic normal at (x_c, y_c):
    # f_x = 2*x_c + y_c,  f_y = x_c + 2*y_c
    d = np.sqrt((2*x_c + y_c)**2 + (x_c + 2*y_c)**2 + 1)
    n_face = np.array([-(2*x_c + y_c), -(x_c + 2*y_c), 1]) / d
    face_normals.append(n_face)

face_centroids = np.array(face_centroids)
face_normals = np.array(face_normals)

# -----------------------------
# 7. Compute vertex normals by averaging the face normals of adjacent triangles
# -----------------------------
num_vertices = len(data)
vertex_normals = np.zeros((num_vertices, 3))
for v in range(num_vertices):
    adjacent_normals = []
    for i, triangle in enumerate(tri.simplices):
        if v in triangle:
            adjacent_normals.append(face_normals[i])
    if adjacent_normals:
        adjacent_normals = np.array(adjacent_normals)
        avg_normal = np.mean(adjacent_normals, axis=0)
        norm_val = np.linalg.norm(avg_normal)
        if norm_val > 0:
            avg_normal /= norm_val
        vertex_normals[v] = avg_normal

# -----------------------------
# 8. Plot the lifted mesh with face and vertex normals
# -----------------------------
# Plot face normals (evaluated at triangle centroids)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(X, Y, Z, triangles=tri.simplices, cmap='viridis',
                alpha=0.5, edgecolor='gray')
ax.quiver(face_centroids[:, 0], face_centroids[:, 1], face_centroids[:, 2],
          face_normals[:, 0], face_normals[:, 1], face_normals[:, 2],
          length=0.5, color='blue', label='Face Normals')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Lifted Mesh with Face Normals')
plt.tight_layout()
plt.savefig("lifted_mesh_with_face_normals2.png")
plt.show()

# Plot vertex normals (averaged from adjacent face normals)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(X, Y, Z, triangles=tri.simplices, cmap='viridis',
                alpha=0.5, edgecolor='gray')
ax.quiver(points3d[:, 0], points3d[:, 1], points3d[:, 2],
          vertex_normals[:, 0], vertex_normals[:, 1], vertex_normals[:, 2],
          length=0.5, color='red', label='Vertex Normals')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Lifted Mesh with Vertex Normals')
plt.tight_layout()
plt.savefig("lifted_mesh_with_vertex_normals2.png")
plt.show()

# -----------------------------
# 9. Compute the second fundamental form and shape operator using vertex normals
# -----------------------------
# For f(x,y)= x^2+xy+y^2, we have:
#   f_xx = 2, f_xy = 1, f_yy = 2.
# At a vertex with vertex normal n_v = (n_x, n_y, n_z), we get:
#   L = 2 * n_z,   M = 1 * n_z,   N = 2 * n_z.
L_coeffs = 2 * vertex_normals[:, 2]
M_coeffs = np.zeros_like(L_coeffs) + 1 * vertex_normals[:, 2]  # M = n_z
N_coeffs = 2 * vertex_normals[:, 2]

print("Second Fundamental Form Coefficients (per vertex):")
for i in range(len(data)):
    print(f"Vertex {i}: L = {L_coeffs[i]:.4f}, M = {M_coeffs[i]:.4f}, N = {N_coeffs[i]:.4f}")

# Compute shape operator S = I^{-1} II at each vertex and extract curvatures
k1s = []   # principal curvature 1
k2s = []   # principal curvature 2
Gauss = [] # Gaussian curvature
Mean = []  # Mean curvature

for i, (x, y) in enumerate(data):
    # First fundamental form I:
    E = 1 + (2*x + y)**2
    F = (2*x + y) * (x + 2*y)
    G = 1 + (x + 2*y)**2
    I_mat = np.array([[E, F],
                      [F, G]])
    detI = E*G - F**2
    I_inv = np.array([[G, -F],
                      [-F, E]]) / detI

    # Second fundamental form II:
    n_z = vertex_normals[i, 2]
    L_val = 2 * n_z
    M_val = 1 * n_z
    N_val = 2 * n_z
    II_mat = np.array([[L_val, M_val],
                       [M_val, N_val]])

    S = I_inv @ II_mat  # Shape operator (2x2 matrix)
    eigvals = np.linalg.eigvals(S)
    eigvals = np.sort(eigvals)[::-1]  # sort descending: k1 >= k2
    k1 = eigvals[0]
    k2 = eigvals[1]
    k1s.append(k1)
    k2s.append(k2)
    Gauss.append(k1 * k2)
    Mean.append((k1 + k2) / 2)

k1s = np.array(k1s)
k2s = np.array(k2s)
Gauss = np.array(Gauss)
Mean = np.array(Mean)

# -----------------------------
# 10. Visualize the curvatures on the (x,y) domain
# -----------------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

sc0 = axs[0, 0].scatter(X, Y, c=k1s, cmap='viridis')
axs[0, 0].set_title('Principal Curvature k₁')
fig.colorbar(sc0, ax=axs[0, 0])
axs[0, 0].set_xlabel('x')
axs[0, 0].set_ylabel('y')

sc1 = axs[0, 1].scatter(X, Y, c=k2s, cmap='viridis')
axs[0, 1].set_title('Principal Curvature k₂')
fig.colorbar(sc1, ax=axs[0, 1])
axs[0, 1].set_xlabel('x')
axs[0, 1].set_ylabel('y')

sc2 = axs[1, 0].scatter(X, Y, c=Gauss, cmap='viridis')
axs[1, 0].set_title('Gaussian Curvature')
fig.colorbar(sc2, ax=axs[1, 0])
axs[1, 0].set_xlabel('x')
axs[1, 0].set_ylabel('y')

sc3 = axs[1, 1].scatter(X, Y, c=Mean, cmap='viridis')
axs[1, 1].set_title('Mean Curvature')
fig.colorbar(sc3, ax=axs[1, 1])
axs[1, 1].set_xlabel('x')
axs[1, 1].set_ylabel('y')

plt.suptitle('Curvature Visualizations on the Lifted Surface (z = x² + xy + y²)')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("curvatures_on_lifted_surface2.png")
plt.show()