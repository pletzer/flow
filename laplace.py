from fenics import *
from mshr import *
import numpy as np
import utils

xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0

# Define boundary condition
class Hole(SubDomain):
    
    def __init__(self, x0, y0, radius, n):
        super().__init__()
        the = np.linspace(0, 2*np.pi, n + 1)
        self.xc = x0 + radius*np.cos(the)
        self.yc = y0 + radius*np.sin(the)
        self.radius = radius
        self.x0 = x0
        self.y0 = y0
        
    def get_vertices(self):
        return [Point(self.xc[i], self.yc[i]) for i in range(len(self.xc))]

    def inside(self, x, on_boundary):
        delta = 0.01 # min distance of hole to the box boundary
        return on_boundary and (x[0] > xmin + delta) and (x[1] > ymin + delta) \
            and (x[0] < xmax - delta) and (x[1] < ymax - delta)

hole = Hole(x0=xmin + (xmax - xmin)*0.5, y0=ymin + (ymax - ymin)*0.5, radius=0.1, n=16)

# Define mesh and function space
box = Rectangle(Point(xmin, ymin), Point(xmax, ymax))
domain = box - Polygon(hole.get_vertices())
mesh = generate_mesh(domain, 16)

# solution space
U = FunctionSpace(mesh, 'P', 1)

# Define boundaries
bcs = [DirichletBC(U, Constant(0), hole), 
       DirichletBC(U, Constant(1), 'near(x[0], 0)'),
       DirichletBC(U, Constant(0), 'near(x[0], 1)')]


# Define trial and test functions
u = TrialFunction(U) # solution
v = TestFunction(U) # trial

# Define source term
f = Constant(0)

# Define bilinear and linear forms
a = inner(grad(u), grad(v))*dx
L = dot(f, v)*dx

# Compute solution
u = Function(U)
solve(a == L, u, bcs)

uvec = u.vector().get_local()
print(f'min/avg/max u = {np.min(uvec)}/{np.mean(uvec)}/{np.max(uvec)}')

xdmffile_u = XDMFFile('laplace.xdmf')
xdmffile_u.write(u, 0)