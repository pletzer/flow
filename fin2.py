"""
FEniCS tutorial demo program: Incompressible Navier-Stokes equations
for flow around a obstacle using the Incremental Pressure Correction
Scheme (IPCS).

  u' + u . nabla(u)) - div(sigma(u, p)) = f
                                 div(u) = 0
"""

from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
import utils
import sys
import pandas as pd

Lx, Ly = 1.0, 1.0 # domain size

nresolution = 32
nobstacle = 100
vmax = 2.0 # used to set the time step
dt = 0.1 * np.sqrt(Lx * Ly / nresolution**2)/ vmax # time step
num_steps = 2000   # max number of time steps
T = num_steps * dt           # final time
mu = 0.001 # 0.0010518 #  dynamic viscosity of water at 18 deg C #0.001
rho = 1         # density
finThickness = 0.10 # normalized to its length 

# attack angle
results = {'alpha': [], 'lift': [], 'drag': [], 'std_drag': [], 'std_lift': []}
for alpha in np.linspace(0 * np.pi/180, 12 * np.pi/180, 21):

    # t is the thickness
    xc, yc = utils.NACAFoilPoints(nobstacle, m=0.0, p=0.3, t=finThickness)
    # rotate the foil
    xc2 = xc*np.cos(alpha) + yc*np.sin(alpha)
    yc2 = -xc*np.sin(alpha) + yc*np.cos(alpha)
    # shift/scale to the right location
    xfoil = 0.3*xc2 + Lx/4.
    yfoil = 0.3*yc2 + Ly/2.5

    # Create mesh
    channel = Rectangle(Point(0, 0), Point(Lx, Ly))
    vertices = [Point(xfoil[i], yfoil[i]) for i in range(len(xfoil))]

    obstacle = Polygon(vertices)
    domain = channel - obstacle
    mesh = generate_mesh(domain, nresolution)
    # plot(mesh)
    # plt.show()

    # Define function spaces
    V = VectorFunctionSpace(mesh, 'P', 2)
    Q = FunctionSpace(mesh, 'P', 1)

    # Define boundaries
    inflow   = 'near(x[0], 0)'
    outflow  = f'near(x[0], {Lx})'
    walls    = f'near(x[1], 0) || near(x[1], {Ly})'

    # Define boundary condition

    class ObstacleBoundary(SubDomain):
        def __init__(self, xc, yc):
            super().__init__()
            self.xc = xc
            self.yc = yc
            
        def inside(self, x, on_boundary):
            delta = 0.01
            # box around the obstacle
            return on_boundary and (x[0] > 0 + delta) and (x[0] < Lx - delta) and \
                                   (x[1] > 0 + delta) and (x[1] < Ly - delta)

    obstacle_boundary = ObstacleBoundary(xc=xfoil, yc=yfoil)

    # Define inflow profile
    inflow_profile = (f'1.0', '0')

    # Define boundary conditions
    bcu_inflow = DirichletBC(V, Expression(inflow_profile, degree=2), inflow)
    bcu_walls = DirichletBC(V, Constant((0, 0)), walls)
    bcu_obstacle = DirichletBC(V, Constant((0, 0)), obstacle_boundary)
    bcp_outflow = DirichletBC(Q, Constant(0), outflow)
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
    mu = Constant(mu)
    rho = Constant(rho)

    # Define symmetric gradient
    def epsilon(u):
        return sym(nabla_grad(u))

    # Define stress tensor
    def sigma(u, p):
        return 2*mu*epsilon(u) - p*Identity(len(u))

    # Define variational problem for step 1
    F1 = rho*dot((u - u_n) / k, v)*dx \
    + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
    + inner(sigma(U, p_n), epsilon(v))*dx \
    + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
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
    xdmffile_u = XDMFFile(f'fin2_results/velocity_w{finThickness}_a{alpha * 180/np.pi:.2}.xdmf')
    xdmffile_p = XDMFFile(f'fin2_results/pressure_w{finThickness}_a{alpha * 180/np.pi:.2}.xdmf')

    # # Save mesh to file
    # File('fin_results/obstacle.xml.gz') << mesh

    # Create progress bar
    progress = Progress('Time-stepping')
    #set_log_level(16) #(PROGRESS)

    # Time-stepping
    lifts = []
    drags = []
    t = 0
    for n in range(num_steps):

        # Update current time
        t += dt

        # Step 1: Tentative velocity step
        b1 = assemble(L1)
        [bc.apply(b1) for bc in bcu]
        solve(A1, u_.vector(), b1, 'bicgstab', 'hypre_amg')
        #solve(A1, u_.vector(), b1, 'gmres', 'ilu')

        # Step 2: Pressure correction step
        b2 = assemble(L2)
        [bc.apply(b2) for bc in bcp]
        solve(A2, p_.vector(), b2, 'bicgstab', 'hypre_amg')
        #solve(A2, p_.vector(), b2, 'gmres', 'ilu')

        # Step 3: Velocity correction step
        b3 = assemble(L3)
        solve(A3, u_.vector(), b3, 'cg', 'sor')

        # Plot solution
        # plot(u_, title='Velocity')
        # plot(p_, title='Pressure')

        # Save solution to file (XDMF/HDF5)
        xdmffile_u.write(u_, t)
        xdmffile_p.write(p_, t)

        # Update previous solution
        u_n.assign(u_)
        p_n.assign(p_)
        

        # Define the lift and drag integrals
        lift = assemble(p_ * normal[1] * ds(1)) # ds(1) means over contour tagged with 1
        drag = assemble(p_ * normal[0] * ds(1))

        print('-'*30)
        print(f"time {n} attack angle deg: {alpha*180/np.pi:.2f} Lift: {lift:.5f} Drag: {drag:.5f} L/D: {lift/drag:.3f}")
    
        # exit condition
        nlast = num_steps // 10
        lifts.append(lift)
        drags.append(drag)
        std_lifts = np.std(lifts[-nlast:])
        std_drags = np.std(drags[-nlast:])
        avg_lifts = np.mean(lifts[-nlast:])
        avg_drags = np.mean(drags[-nlast:])
        if n > 2*nlast and avg_drags > 0 and avg_lifts > 0 and \
            std_lifts / avg_lifts < 0.05 and \
            std_drags / avg_drags < 0.05:
            break
        
        
    print('='*40)
    # store the last computed values
    results['alpha'].append(alpha)
    results['lift'].append(avg_lifts)
    results['drag'].append(avg_drags)
    results['std_lift'].append(std_lifts)
    results['std_drag'].append(std_drags)
    
df = pd.DataFrame(results)
df.to_csv(f'results_finThickness{finThickness}.csv')
    
plt.plot(np.fabs(np.array(results['alpha'])*180/np.pi), np.array(results['lift'])/np.array(results['drag']))
plt.title('L/D vs angle of attack (deg)')
plt.show()


