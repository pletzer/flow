import numpy as np
import matplotlib.pyplot as plt


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
    
    # numIntersects = ( (distance < tol) \
    #        | ((0+tol <= lam) & (lam < 1-tol) & (tee > 0+tol)) ).sum()
    numIntersects = ( (distance < tol) \
           | ((0-tol <= lam) & (lam < 1) & (tee > 0)) ).sum()
    # print(f'lam = {lam}')
    # print(f'tee = {tee}')
    # print(f'condition  {(0 - tol <= lam) * (lam <= 1 + tol) * (tee > 0 - tol)}')
    # print(f'numinteresects = {numIntersects}')
    # print(f'target point = {x} x0={x0[-2]},{y0[-2]} x1={x1[-2]},{y1[-2]} distance={distance[-1]} lam={lam[-2]} tee={tee[-2]}')
    # print(f'target point = {x} x0={x0[-1]},{y0[-1]} x1={x1[-1]},{y1[-1]} distance={distance[-1]} lam={lam[-1]} tee={tee[-1]}')
    # rad = np.sqrt( (x[0] - 0.5)**2 + (x[1] - 0.5)**2 ) # debug
    if numIntersects % 2 == 0:
        # point x is outside
        # if (x[0] - 0.5)**2 + (x[1] - 0.5)**2 <= 0.1**2:
        #     print(f'WARNING: point {x} found to be outside but is INSIDE rad = {rad:10.6f} numIntersects = {numIntersects}')
        #     return True
        return False
    # point x is inside
    print(f'point {x} is inside numIntersects = {numIntersects}')
    return True


def NACAFoilPoints(npts, m, p, t, beta=1.5):
    """
    Create foil contour points
    @param npts: approx number of points
    @param m: camber
    @param p: breaking point of the foil
    @param t: thickness
    @param beta: mesh packing exponent (> 1 to pack more points at the leading edge and tail)
    @return foil in normalized coordinates 0 <= x <= 1
    """
    assert(p > 0 and p < 1)
    n1 = int(npts * p / 2)
    n2 = int(npts * (1 - p) / 2)
    dx = 1./npts # normalized distance
    
    # first part
    x1 = np.linspace(0., p, n1)
    # add resolution at the leading edge
    x1 = p*(x1/p)**beta
    yc1 = m*(2*p*x1 - x1**2)/p**2
    
    # second part
    x2 = np.linspace(p + dx, 1., n2)
    # add resolution at the tail
    x2 = 1. - (1 - p)*((1. - x2)/(1. - p))**beta
    yc2 = m*((1-2*p) + 2*p*x2 - x2**2)/(1 - p)**2
    
    x = np.concatenate((x1, x2))
    # chord
    yc = np.concatenate((yc1, yc2))
    thickness = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    
    xContour = np.concatenate((x, x[::-1]))
    yContour = np.concatenate((yc - thickness, yc[::-1] + thickness[::-1]))
    
    return xContour, yContour
    
def testFoil():
    xc, yc = NACAFoilPoints(21, m=0.05, p=0.4, t=0.1)
    print(xc)
    print(yc)
    
def testIsInsideContour3_2():
    n = 200
    # contour
    xc, yc = NACAFoilPoints(n, m=0.05, p=0.4, t=0.1)
    # contour slighlty inside
    xc2, yc2 = NACAFoilPoints(n, m=0.05, p=0.4, t=0.5*0.1)
    count = 0
    for i in range(1, len(xc2) - 1):
        pt =(xc2[i], yc2[i])
        try:
            assert isInsideContour3(pt, xc=xc, yc=yc, tol=1.e-10)
        except:
            print(f'ERROR point {pt} {i} found to be outside, but should be inside')
            plt.plot(xc, yc, 'k-', [pt[0]], [pt[1]], 'r*')
            plt.axis('equal')
            plt.show()
            count += 1
    # contour slightly outside
    xc2, yc2 = NACAFoilPoints(n, m=0.05, p=0.4, t=1.5*0.1)
    for i in range(1, len(xc2) - 1):
        pt =(xc2[i], yc2[i])
        try:
            assert not isInsideContour3(pt, xc=xc, yc=yc, tol=1.e-10)
        except:
            print(f'ERROR point {pt} {i} found to be inside, but should be outsides')
            plt.plot(xc, yc, 'k-', [pt[0]], [pt[1]], 'ro')
            plt.axis('equal')
            plt.show()
            count += 1
    print (f'Found {count} misplaced points')
    
    
        
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
    
if __name__ == '__main__':
    testFoil()
    testIsInsideContour3_2()
    testIsInsideContour3()
