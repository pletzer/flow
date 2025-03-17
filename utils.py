import numpy as np

def isInsideContour2(x, xc, yc, tol):
    # Ray-casting algorithm for point-in-polygon test
    n = len(xc)
    inside = False
    p1x, p1y = xc[0], yc[0]
    for i in range(n + 1):
        p2x, p2y = xc[i %n], yc[i %n]
        if x[1] > min(p1y, p2y):
            if x[1] <= max(p1y, p2y):
                if x[0] <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (x[1] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x[0] <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def isInsideContour3(x, xc, yc, tol):
    """
    Shoot a ray in the direction -u and count the number of intersections
    """
    # compute the intersection parameter for each segment
    # random direction
    u, v = -1, 0
    
    # the target point
    xp, yp = x[0], x[1]
    
    # starting points of the segments
    x0, y0 = xc[:-1], yc[:-1]
    x1, y1 = xc[1:], yc[1:]
    
    # vectors between start/end positions
    dx = x1 - x0
    dy = y1 - y0
    
    det = dx*v - dy*u
    # handle the case of zero determinant. 2 cases can give rise to det == 0:
    # 1) the xp is degenerate with the first point of the segment (x0). In this case
    # we return "inside"
    # 2) the directions of u and the segment dx are parallel but the points are distinct,
    # there is no intersection. 
    
    
    if det.any() == 0:
        print(f'*** zero det: dx={dx} dy={dy}')
    assert(det.all() != 0)
    
    dxp = xp - x0
    dyp = yp - y0
    
    lam = (+ v *dxp - u *dyp ) / det
    tee = (- dy*dxp + dx*dyp ) / det
    distance = np.sqrt( dxp**2 + dyp**2 )
    
    # count the number of ray-segment intersections such that 0 <= lam < 1 and t > 0
    
    numIntersects = ( (distance < tol) \
           | ((0 <= lam) & (lam < 1) & (tee > 0)) ).sum()
    # print(f'lam = {lam}')
    # print(f'tee = {tee}')
    # print(f'condition  {(0 - tol <= lam) * (lam <= 1 + tol) * (tee > 0 - tol)}')
    # print(f'numinteresects = {numIntersects}')
    # print(f'target point = {x} x0={x0[-2]},{y0[-2]} x1={x1[-2]},{y1[-2]} distance={distance[-1]} lam={lam[-2]} tee={tee[-2]}')
    # print(f'target point = {x} x0={x0[-1]},{y0[-1]} x1={x1[-1]},{y1[-1]} distance={distance[-1]} lam={lam[-1]} tee={tee[-1]}')
    if numIntersects % 2 == 0:
        # point x is outside
        return False
    # point x is inside
    return True
    



def isInsideContour(p, xc, yc, tol):
    """
    Check if a point is inside closed contour. This only works if the region is convex

    @param p point (2d array)
    @param xc array of x points, anticlockwise and must close
    @param yc array of y points, anticlockwise and must close
    @return True if p is inside, False otherwise
    """
    # vectors from point p to start point of segment
    a = np.array([xc[:-1], yc[:-1]])
    a[0, :] -= p[0]
    a[1, :] -= p[1]

    # vectors from point p to end point of segment
    b = np.array([xc[1:], yc[1:]])
    b[0, :] -= p[0]
    b[1, :] -= p[1]
    
    vecprod = a[0, :]*b[1, :] - a[1, :]*b[0, :]
    dotprod = a[0, :]*b[0, :] + a[1, :]*b[1, :]
    sumangles = np.sum( np.arctan2( vecprod, dotprod ) )
    
    # print(f'sumangles/(2*np.pi) = {sumangles/(2*np.pi)}')
    if sumangles/(2*np.pi) > 0.5:
        return True
    return False

    # # I don't think works well if the contour is concave (?)
    # areas = a[0, :]*b[1, :] - a[1, :]*b[0, :]
   
    # # return True if all the areas are positive
    # return not np.any(areas < -tol)

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
    dx = 1./npts # normalized distance
    
    # first part
    x1 = np.linspace(0., p, n1)
    yc1 = m*(2*p*x1 - x1**2)/p**2
    
    # second part
    x2 = np.linspace(p + dx, 1. - dx, n2)
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
        
def testIsInsideContour3():
    ts = np.linspace(0., 2*np.pi, 81) #+ np.pi/45.
    # ts = np.linspace(0., 2*np.pi, 5)
    xc = np.cos(ts)
    yc = np.sin(ts)
    assert not isInsideContour3((2.0, 0.0), xc=xc, yc=yc, tol=1.e-10) 
    assert not isInsideContour3((1.1, 0.0), xc=xc, yc=yc, tol=1.e-10)
    assert not isInsideContour3((1.01, 0.0), xc=xc, yc=yc, tol=1.e-10)
    assert not isInsideContour3((1.001, 0.0), xc=xc, yc=yc, tol=1.e-10)
    assert not isInsideContour3((1.00001, 0.0), xc=xc, yc=yc, tol=1.e-10)
    assert isInsideContour3((0.0, 0.0), xc=xc, yc=yc, tol=1.e-10)
    assert isInsideContour3((0.5, 0.0), xc=xc, yc=yc, tol=1.e-10)
    assert isInsideContour3((0.9, 0.0), xc=xc, yc=yc, tol=1.e-10)
    assert isInsideContour3((0.99, 0.0), xc=xc, yc=yc, tol=1.e-10)
    assert isInsideContour3((0.999, 0.0), xc=xc, yc=yc, tol=1.e-10)
    assert isInsideContour3((0.9999, 0.0), xc=xc, yc=yc, tol=1.e-10)
    assert isInsideContour3((0.99999, 0.0), xc=xc, yc=yc, tol=0.001)


def testIsInsideContour():
    ts = np.linspace(0., 2*np.pi, 81) + np.pi/45.
    # ts = np.linspace(0., 2*np.pi, 5)
    xc = np.cos(ts)
    yc = np.sin(ts)
    assert not isInsideContour((2.0, 0.0), xc=xc, yc=yc, tol=1.e-10) 
    assert not isInsideContour((1.1, 0.0), xc=xc, yc=yc, tol=1.e-10)
    assert not isInsideContour((1.01, 0.0), xc=xc, yc=yc, tol=1.e-10)
    assert not isInsideContour((1.001, 0.0), xc=xc, yc=yc, tol=1.e-10)
    assert not isInsideContour((1.00001, 0.0), xc=xc, yc=yc, tol=1.e-10)
    assert isInsideContour((0.0, 0.0), xc=xc, yc=yc, tol=1.e-10)
    assert isInsideContour((0.5, 0.0), xc=xc, yc=yc, tol=1.e-10)
    assert isInsideContour((0.9, 0.0), xc=xc, yc=yc, tol=1.e-10)
    assert isInsideContour((0.99, 0.0), xc=xc, yc=yc, tol=1.e-10) # this should actually be slightly outside
    assert isInsideContour((0.999, 0.0), xc=xc, yc=yc, tol=1.e-10) # this should actually be slightly outside
    
if __name__ == '__main__':
    testFoil()
    testIsInsideContour3()
    testIsInsideContour()