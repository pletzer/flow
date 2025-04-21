from fenics import *
from mshr import *
import numpy as np
import utils

nobs = 200
nres = 64

# Define boundary condition
class Hole(SubDomain):
    
    def __init__(self, xc, yc):
        super().__init__()
        self.xc = xc
        self.yc = yc
        
    def get_vertices(self):
        return [Point(self.xc[i], self.yc[i]) for i in range(len(self.xc))]

    def inside(self, x, on_boundary):
        # we don't need to check the inside of the hole, we just need to remove the extranal boundary 
        # so we're left with the hole
        delta = 0.01
        return (on_boundary and \
            (x[0] > 0.0 + delta and x[1] > 0.0 + delta and x[0] < 1.0 - delta and x[1] < 1.0 - delta))
        #return utils.isInsideContour3(x, self.xc, self.yc, tol=1e-12)

xfoil, yfoil = utils.NACAFoilPoints(nobs, m=0.0, p=0.3, t=0.1)
xc = 0.4*xfoil + 0.3
yc = 0.4*yfoil + 0.4
print(f'number of hole points = {len(xc)}')
hole = Hole(xc, yc)

# Define mesh and function space
box = Rectangle(Point(0, 0), Point(1, 1))
domain = box - Polygon(hole.get_vertices())
mesh = generate_mesh(domain, nres)

# solution space
U = FunctionSpace(mesh, 'P', 1)

# Define boundary
bc = DirichletBC(U, Constant(1), hole)
print(dir(bc))

u = Function(U)
u.vector().set_local(u.vector().get_local() * 0)
bc.apply(u.vector())
print(f'number of non-zero values = {np.sum(u.vector().get_local())}')

xdmffile_u = XDMFFile('boundary.xdmf')
xdmffile_u.write(u, 0)

# import matplotlib.pyplot as plt

# # Plot the function
# plot(u)
# plt.show()
