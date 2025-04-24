
from fenics import *
from mshr import *
import numpy as np

# Create a rectangular domain, split into 2 parts with a separation in the middle. 
# Set a pressure difference of one between bottom and top.
# Then compute the vertical force applied on the separation due to the difference of pressure
# between top and bottom

Lx = 2.0

channel = Rectangle(Point(0, -1), Point(Lx, 1))
vertices = [Point(0., -0.1), Point(Lx, -0.1), Point(Lx, 0.1), Point(0., 0.1), Point(0., -0.1)]
obstacle = Polygon(vertices)
domain = channel - obstacle
mesh = generate_mesh(domain, 16)

# Create function space
Q = FunctionSpace(mesh, 'P', 1)
p = Function(Q)
p_values = p.vector()

# Create subdomains
class Top(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] >= 0.1 - 1.e-10

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] <= -0.1 + 1.e-10

class Wing(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1.e-10
        return on_boundary and x[1] <= 0.1 + tol and x[1] >= -0.1 - tol

subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
subdomains.set_all(0)
top = Top()
bottom = Bottom()
wing = Wing()
top.mark(subdomains, 2)
bottom.mark(subdomains, 1)
wing.mark(subdomains, 3)

# Assign the pressure to the top and bottom subdomains
dofmap = Q.dofmap()
dofs = dofmap.dofs()

# Assign values based on subdomain
for i in range(len(dofs)):
    x = Q.tabulate_dof_coordinates()[i]
    if bottom.inside(x, False):
        p_values[i] = 1.0
    elif top.inside(x, False):
        p_values[i] = 0.0
xdmffile_p = XDMFFile('testlift_pressure.xdmf')
xdmffile_p.write(p, 0)

# Compute the lift
# Mark the boundary so we can compute the lift/drag
boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_markers.set_all(0)
obstacle_boundary = Wing()
obstacle_boundary.mark(boundary_markers, 1)

# Define the measure for the boundary
ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
# Compute the normal vector
normal = FacetNormal(mesh)

# Compute the lift
lift = assemble(p * normal[1] * ds(1))
print(f'lift = {lift}')
