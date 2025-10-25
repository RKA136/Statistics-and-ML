import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'


# Define grid
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Parameters for the Gaussian
mu_x, mu_y = 0, 0
sigma_x, sigma_y = 2, 1
rho = 0.99  # correlation coefficient between x and y

# Covariance matrix and its determinant
Sigma = np.array([[sigma_x**2, rho * sigma_x * sigma_y],
                  [rho * sigma_x * sigma_y, sigma_y**2]])
det_Sigma = np.linalg.det(Sigma)
inv_Sigma = np.linalg.inv(Sigma)

# Compute exponent term
XY = np.dstack((X - mu_x, Y - mu_y))
exponent = np.einsum('...i,ij,...j', XY, inv_Sigma, XY)
Z = (1 / (2 * np.pi * np.sqrt(det_Sigma))) * np.exp(-0.5 * exponent)

# 3D Plot using Plotly
fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Viridis')])
fig.update_layout(
    title='2D Gaussian Probability Density Function',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='P(X, Y)'
    ),
    width=800,
    height=700
)
fig.show()

# Rotate the frame by angle theta and calculate new covariance matrix
def rotate_covariance(sigma_x, sigma_y, rho, theta):
    # Original covariance matrix
    Sigma = np.array([[sigma_x**2, rho * sigma_x * sigma_y],
                      [rho * sigma_x * sigma_y, sigma_y**2]])
    
    # Rotation matrix
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    
    # Rotated covariance matrix
    Sigma_rotated = R @ Sigma @ R.T
    return Sigma, Sigma_rotated

# Example rotation
theta = np.pi / 4  # 45 degrees
Sigma, Sigma_rotated = rotate_covariance(sigma_x, sigma_y, rho, theta)
print("Original Covariance Matrix:\n", Sigma)
print("Rotated Covariance Matrix:\n", Sigma_rotated)