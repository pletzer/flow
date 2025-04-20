from fenics import *
from mshr import *
import numpy as np

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
        return on_boundary and ((x[0] - self.x0)**2 + (x[1] - self.y0)**2 <= self.radius**2)

hole = Hole(0.5, 0.5, 0.1, 16)

# Define mesh and function space
box = Rectangle(Point(0, 0), Point(1, 1))
domain = box - Polygon(hole.get_vertices())
mesh = generate_mesh(domain, 16)

# solution space
U = FunctionSpace(mesh, 'P', 1)

# Define boundaries
bcs = [DirichletBC(U, Constant(0), hole), 
       DirichletBC(U, Constant(0), 'near(x[0], 0) || near(x[1], 0) || near(x[1], 1)'),    
       DirichletBC(U, Constant((1)), 'near(x[1], 1)')]


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

xdmffile_u = XDMFFile('t.xdmf')
xdmffile_u.write(u, 0)