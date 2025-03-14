import numpy as np

def isInsideContour(p, xc, yc, tol):
    """
    Check if a point is inside closed contour

    @param p point (2d array)
    @param xc array of x points, anticlockwise and must close
    @param yc array of y points, anticlockwise and must close
    @return True if p is inside, False otherwise
    """

    # number of segments
    numSeg = len(xc) - 1

    # vectors from point p to start point of segment
    a = np.array([xc[:-1], yc[:-1]])
    a[0, :] -= p[0]
    a[1, :] -= p[1]

    # vectors from point p to end point of segment
    b = np.array([xc[1:], yc[1:]])
    b[0, :] -= p[0]
    b[1, :] -= p[1]

    areas = a[0, :]*b[1, :] - a[1, :]*b[0, :]
   
    # return True if all the areas are positive
    return not np.any(areas < -tol)

def NACAFoilPoints(npts, m, p, t):
    """
    Create foil contour points
    @param npts: approx number of points
    @param m: camber
    @param p: 
    @param t: thickness
    @return foil in normalized coordinates 0 <= x <= 1
    """
    assert(p > 0 and p < 1)
    n1 = int(npts * p / 2)
    n2 = int(npts * (1 - p) / 2)
    
    # first part
    x1 = np.linspace(0., p, n1)
    yc1 = m*(2*p*x1 - x1**2)/p**2
    
    # second part
    x2 = np.linspace(p+1.e-3, 1., n2)
    yc2 = m*((1-2*p) + 2*p*x2 - x2**2)/(1 - p)**2
    
    x = np.concatenate((x1, x2))
    # chord
    yc = np.concatenate((yc1, yc2))
    thickness = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    
    xContour = np.concatenate((x, x[::-1]))
    yContour = np.concatenate((yc - thickness, yc[::-1] + thickness[::-1]))
    
    return xContour, yContour
    
def testFoil():
    xc, yc = NACAFoilPoints(21, m=0.05, p=0.4, t = 0.1)
    print(xc)
    print(yc)
        

def testIsInsideContour():
    ts = np.linspace(0., 2*np.pi, 41) + np.pi/45.
    xc = np.cos(ts)
    yc = np.sin(ts)
    assert not isInsideContour((2.0, 0.0), xc=xc, yc=yc, tol=1.e-10) 
    assert not isInsideContour((1.1, 0.0), xc=xc, yc=yc, tol=1.e-10)
    assert not isInsideContour((1.01, 0.0), xc=xc, yc=yc, tol=1.e-10)
    assert not isInsideContour((1.001, 0.0), xc=xc, yc=yc, tol=1.e-10)
    assert not isInsideContour((1.00001, 0.0), xc=xc, yc=yc, tol=1.e-10)
    assert isInsideContour((0.5, 0.0), xc=xc, yc=yc, tol=1.e-10)
    assert isInsideContour((0.9, 0.0), xc=xc, yc=yc, tol=1.e-10)
    assert isInsideContour((0.99, 0.0), xc=xc, yc=yc, tol=1.e-10)
    assert isInsideContour((0.999, 0.0), xc=xc, yc=yc, tol=0.01)
    
if __name__ == '__main__':
    testFoil()
    testIsInsideContour()