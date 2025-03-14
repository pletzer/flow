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


def test():
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
    test()