"""
FEniCS tutorial demo program: Incompressible Navier-Stokes equations
for flow around a obstacle using the Incremental Pressure Correction
Scheme (IPCS).

  u' + u . nabla(u) - div(sigma(u, p)) = f
                                 div(u) = 0
"""

from fenics import *
from mshr import *
import numpy as np
import utils
import defopt
import pandas as pd

def main(*, Lx: float=2.5, Ly: float=2.0, nres: int=32, nobst: int=100, nsteps: int=100, \
    alpha_deg: float=10., Re: float=1000, normThickness: float=0.1, output_dir: str='fin_results'):
    """
    Run lif/drag computation
    @param Lx x domain size
    @param Ly y domain size
    @param nres domain resolution
    @param nobst number of points describing the obstacle
    @param Re Reynolds number
    @param nsteps number of time steps, the time step size depends on Re
    @param alpha_deg attack angle in degrees
    @param normThickness normalised thickness of the fin
    @param output_dir results will be written to this directory
    """

    vinput = 1.0
    dt = 0.01*(1/Re) # 0.001 * np.sqrt(Lx * Ly / nres**2)/ vmax # time step
    tmax = nsteps * dt           # final time
    alpha = -alpha_deg * np.pi/180

    # generate the geometry of the fin
    xc, yc = utils.NACAFoilPoints(nobst, m=0.0, p=0.3, t=normThickness)
    # rotate the foil
    xc2 = xc*np.cos(alpha) - yc*np.sin(alpha)
    yc2 = xc*np.sin(alpha) + yc*np.cos(alpha)
    # shift/scale to the right location
    xfoil = xc2 + Lx/4.
    yfoil = yc2 + 2*Ly/3

    # Create the mesh
    channel = Rectangle(Point(0, 0), Point(Lx, Ly))
    vertices = [Point(xfoil[i], yfoil[i]) for i in range(len(xfoil))]

    obstacle = Polygon(vertices)
    domain = channel - obstacle
    mesh = generate_mesh(domain, nres)

    # Define function spaces
    V = VectorFunctionSpace(mesh, 'P', 2)
    Q = FunctionSpace(mesh, 'P', 1)

    # Define boundaries
    inflow   = 'near(x[0], 0)'
    outflow  = f'near(x[0], {Lx})'
    walls    = f'near(x[1], 0) || near(x[1], {Ly})'

    # Define the boundary conditions
    class ObstacleBoundary(SubDomain):
        def __init__(self, xc, yc):
            super().__init__()
            self.xc = xc
            self.yc = yc
            self.xmin = min(xc)
            self.xmax = max(xc)
            self.ymin = min(yc)
            self.ymax = max(yc)
            
        def inside(self, x, on_boundary):
            tol = 0.01*min(Lx, Ly)
            return (on_boundary and \
                (x[0] > self.xmin - tol) and (x[0] < self.xmax + tol) and \
                (x[1] > self.ymin - tol) and (x[1] < self.ymax + tol))

    obstacle_boundary = ObstacleBoundary(xc=xfoil, yc=yfoil)

    # Define boundary conditions
    bcu_inflow = DirichletBC(V, Constant((vinput, 0)), inflow)
    bcu_walls = DirichletBC(V, Constant((vinput, 0)), walls)
    bcu_obstacle = DirichletBC(V, Constant((0, 0)), obstacle_boundary) # no slip
    bcp_outflow = DirichletBC(Q, Constant(0), outflow) # only for the pressure
    bcu = [bcu_inflow, bcu_walls, bcu_obstacle]
    bcp = [bcp_outflow]

    # Mark the boundary so we can compute the lift/drag
    boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary_markers.set_all(0)
    obstacle_boundary.mark(boundary_markers, 1)
    # Define the measure for the boundary
    ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
    # Compute the normal vector
    normal = FacetNormal(mesh)


    # Define trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)
    p = TrialFunction(Q)
    q = TestFunction(Q)

    # Define functions for solutions at previous and current time steps
    u_n = Function(V)
    u_  = Function(V)
    p_n = Function(Q)
    p_  = Function(Q)

    # Define expressions used in variational forms
    U  = 0.5*(u_n + u)
    n  = FacetNormal(mesh)
    f  = Constant((0, 0))
    k  = Constant(dt)
    reynolds = Constant(Re)

    # Define symmetric gradient
    def epsilon(u):
        return sym(nabla_grad(u))

    # Define stress tensor
    def sigma(u, p):
        return 2*epsilon(u)/reynolds - p*Identity(len(u))

    # Define variational problem for step 1
    F1 = dot((u - u_n) / k, v)*dx \
    + dot(dot(u_n, nabla_grad(u_n)), v)*dx \
    + inner(sigma(U, p_n), epsilon(v))*dx \
    + dot(p_n*n, v)*ds - dot(nabla_grad(U)*n/reynolds, v)*ds \
    - dot(f, v)*dx
    a1 = lhs(F1)
    L1 = rhs(F1)

    # Define variational problem for step 2
    a2 = dot(nabla_grad(p), nabla_grad(q))*dx
    L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx

    # Define variational problem for step 3
    a3 = dot(u, v)*dx
    L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx

    # Assemble matrices
    A1 = assemble(a1)
    A2 = assemble(a2)
    A3 = assemble(a3)

    # Apply boundary conditions to matrices
    [bc.apply(A1) for bc in bcu]
    [bc.apply(A2) for bc in bcp]

    # Create XDMF files for visualization output
    xdmffile_u = XDMFFile(f'{output_dir}/velocity.xdmf')
    xdmffile_p = XDMFFile(f'{output_dir}/pressure.xdmf')

    # Create progress bar
    progress = Progress('Time-stepping')
    #set_log_level(16) #(PROGRESS)

    # # Set PETSc options to turn off solver messages
    # PETScOptions.set('ksp_monitor', False)
    # PETScOptions.set('ksp_view', False)
    # PETScOptions.set('log_view', False)


    # Time-stepping
    t = 0
    for n in range(nsteps):

        # Update current time
        t += dt

        # Step 1: Tentative velocity step
        b1 = assemble(L1)
        [bc.apply(b1) for bc in bcu]
        solve(A1, u_.vector(), b1, 'bicgstab', 'hypre_amg')

        # Step 2: Pressure correction step
        b2 = assemble(L2)
        [bc.apply(b2) for bc in bcp]
        solve(A2, p_.vector(), b2, 'bicgstab', 'hypre_amg')

        # Step 3: Velocity correction step
        b3 = assemble(L3)
        #[bc.apply(b3) for bc in bcu] # should this be added?????
        solve(A3, u_.vector(), b3, 'cg', 'sor')

        # Update previous solution
        u_n.assign(u_)
        p_n.assign(p_)
        

        # Define the lift and drag integrals
        lift = assemble(p_ * normal[1] * ds(1)) # ds(1) means over contour tagged with 1
        drag = assemble(p_ * normal[0] * ds(1))

        print('-'*30)
        print(f"{n} Lift: {lift:.6f} Drag: {drag:.6f} L/D: {lift/drag:.6f}  alpha={alpha:.3f} rad Lsin(a)-Dcos(a)={lift*np.sin(-alpha)-drag*np.cos(-alpha):.3f}")



    # Save solution to file (XDMF/HDF5)
    xdmffile_u.write(u_, 0)
    xdmffile_p.write(p_, 0)
    
    # save the parameters used in this run
    data = {
        'Re': [Re],
        'alpha_rad': [alpha],
        'lift': [lift],
        'drag': [drag],
        'normThickness': [normThickness],
        'nsteps': [nsteps],
        'nres': [nres],
        'nobst': [nobst],
    }
    pd.DataFrame(data).to_csv(f'{output_dir}/results.csv')
    
if __name__ == '__main__':
    defopt.run(main)



