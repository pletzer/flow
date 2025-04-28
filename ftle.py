import defopt
import numpy as np
import vtk

def main(*, velocity_xdmf: str='velocity.xdmf', time: float=1, time_step=0.1, forward=True, 
         output_filename: str='ftle.vtk'):
    """
    Compute the finite time Lyapunov exponent
    @param velocity_xdmf: file containing the velocity data
    @param time: integration time
    @param time_step: time step
    @param forward: True or False to integrate backward
    """
    # read the data
    reader = vtk.vtkXdfmfReader()
    reader.SetFileName(velocity_xdmf)
    reader.Update()
    grid = reader.GetOuput()
    points = grid.GetPoints()
    velocity = grid.GetPointData().GetArray(0)
    
    # integrate the points
    streamline = vtk.vtkStreamTracer()
    streamline.SetSurfaceStreamlines(True)
    streamline.SetInputData(grid)
    # source data needs to be a vtkPolyData object
    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputData(grid)
    geometry_filter.Update()
    streamline.SetSourceData(geometry_filter.GetOutput())
    streamline.SetMaximumPropagation(time)
    streamline.SetIntegrationStepUnit(vtk.vtkStreamTracer.TIME_UNIT)
    streamline.SetInitialIntegrationStep(time_step)
    streamline.SetIntegratorTypeToRungeKutta4()
    if forward:
        streamline.SetIntegrationDirectionToForward()
    else:
        streamline.SetIntegrationDirectionToBackward()
    # now integrate
    streamline.Update()
    
    # get the points
    points_integrated = streamline.GetOutput().GetPoints()
    
    cells = grid.GetCells()
    ptIds = vtk.vtkIdList()
    ftle = vtk.vtkDoubleArray()
    ftle.SetName("Eigenvalues")
    ftle.SetNumberOfComponents(1)
    ftle.SetNumberOfTuples(cells.GetNumberOfCells())
    # loop over the cells
    for icell in cells.GetNumberOfCells():
        cells.GetCellAtId(icell, ptIds)
        # get the starting points of this cell
        x0, y0, _ = points.GetPoint(ptIds.GetId(0))
        x1, y1, _ = points.GetPoint(ptIds.GetId(1))
        x2, y2, _ = points.GetPoint(ptIds.GetId(2))
        
        # get the integrated points
        X0, Y0, _ = points_integrated.GetPoint(ptIds.GetId(0))
        X1, Y1, _ = points_integrated.GetPoint(ptIds.GetId(1))
        X2, Y2, _ = points_integrated.GetPoint(ptIds.GetId(2))
        
        dx1, dy1 = x1 - x0, y1 - y0
        dx2, dy2 = x2 - x0, y2 - y0
        dX1, dY1 = X1 - X0, Y1 - Y0
        dX2, dY2 = X2 - X0, Y2 - Y0
        
        D = dx1*dy2 - dx2*dy1
        # if D == 0:
        #     # this is a degenerate triangle
        #     ftle.InsertNextTuple3(0, 0, 0)
        #     continue
        
        # compute the Jacobian TO CHECK
        jac = np.array([[(+dX1*dy2 - dX2*dy1), 
                         (-dX1*dx2 + dX2*dx1)], 
                        [(+dY1*dy2 - dY2*dy1), 
                         (-dY1*dx2 + dY2*dx1)]]) / D
        
        Delta = jac.T @ jac
        
        # compute the eigenvalues
        eigvals, _ = np.linalg.eig(Delta)
        
        ftle.InsertNextTuple1(np.log(np.max(eigvals)) / (2 * time))
            
    # add the FTLE to the grid
    grid.GetCellData().AddArray(ftle)
    
    # write the output
    writer = vtk.vtkVTKWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(grid)
    writer.Write()
    print(f"FTLE written to {output_filename}")
    return 0

if __name__ == "__main__":
    defopt.run(main)