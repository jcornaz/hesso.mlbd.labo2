from sklearn.decomposition import PCA
import skimage.draw
from matplotlib import gridspec

def get_plant_points(img):
    """
    Given the image of a plant, returns a Nx2 array containing the (x, y) coordinate
    of all non-masked points
    """
    # this returns (y, x) tuple
    points = np.transpose(np.nonzero(~img.mask)[:2])
    # makes that an (x, y) tuple
    points = np.roll(points, 1, axis=1)
    return points

def extract_contour(img):
    """
    Wrapper around OpenCV's contour extraction methods. This returns only the longest
    contour
    """
    contours, hierarchy = cv2.findContours(
        skimage.img_as_ubyte(~img.mask[:,:,0]),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    # only take longest contour into account
    longest_contour = np.argmax([len(c) for c in contours])
    cnt = contours[longest_contour]
    cnt = np.squeeze(cnt)
    return cnt

def eccentricity(img, plot=False):
    """Compute eccentricity : the ratio of the eigenvalues of principal components"""
    points = get_plant_points(img)
    pca = PCA(n_components=2).fit(points)
    centroid = pca.mean_
    eccentricity = pca.explained_variance_[0] / pca.explained_variance_[1]

    if plot:
        pl.title('eccentricity : %f' % eccentricity)
        pl.imshow(img)
        
        # direction of the two principal components
        d1 = pca.components_[0,:]
        d2 = pca.components_[1,:]
        
        scale = 100
        p1 = centroid + scale * d1 * pca.explained_variance_ratio_[0]
        pl.plot([centroid[0], p1[0]], [centroid[1], p1[1]], c='r')
        
        p2 = centroid + scale * d2 * pca.explained_variance_ratio_[1]
        pl.plot([centroid[0], p2[0]], [centroid[1], p2[1]], c='y')
        
        pl.xlim((0, img.shape[1]))
        pl.ylim((img.shape[0], 0))
        
    return eccentricity

def get_ellipse_image(yradius, xradius):
    img = ma.masked_all((512, 512), dtype=np.float)
    rr, cc = skimage.draw.ellipse(256, 256, yradius=yradius, xradius=xradius, shape=(512, 512))
    img[rr, cc] = 1
    return img

def circumcircle_radius(p1, p2, p3):
    """
    Given 3 points of a triangle, computes the radius of the circumcircle
    http://www.mathopenref.com/trianglecircumcircle.html

    This function can return np.inf if the triangle has an area of zero
    """
    a = la.norm(p2 - p1)
    b = la.norm(p3 - p2)
    c = la.norm(p1 - p3)
    #print a, b, c

    # Check for degenerate case (triangle has an area of zero)
    denom = np.fabs((a+b+c)*(b+c-a)*(c+a-b)*(a+b-c))
    if abs(denom) < 1e-5:
        return np.inf
    else:
        r = (a * b * c)/np.sqrt(denom)
        return r

def curvature_from_cnt(line, step=10):
    """
    Curvature computation for a line given as a Nx2 array
    We assumes the line is closed (ie the last and first point are contiguous)

    Each set of 3 consecutive (see step) points on the line forms a triangle
    and the radius of the circumcircle of this triangle is the radius of
    the curve for that set. The curvature is 1/r where r is this radius.

    Taking immediatly consecutive points doesn't work well, so `step` specify
    how points triplets are formed : (i-step, i, i+step)
    """
    # Append line end at beginning and beginning at end to simulate closed
    # line
    l = np.r_[line[-1].reshape(-1, 2),
              np.array(line),
              line[0].reshape(-1, 2)]
    curv = []
    linelen = line.shape[0]
    for i in xrange(linelen):
        # indices in l are shifted by -1
        #r = circumcircle_radius(l[i], l[i+1], l[i+2])
        r = circumcircle_radius(line[(i-step)%linelen],
                                line[i],
                                line[(i+step)%linelen])
        if np.isinf(r):
            # Flat => curvature = 0
            curv.append(0)
        else:
            curv.append(1.0 / r)
    return np.array(curv)

def curvature(img, step=10, plot=False, gs=None):
    cnt = extract_contour(img)
    
    cvt = curvature_from_cnt(cnt, step=step)
    
    if plot:
        if gs is None:
            gs = gridspec.GridSpec(1, 2)
        vimg = skimage.img_as_ubyte(img)
        cv2.drawContours(vimg, [cnt], 0, (0,255,0), 3)
        
        pl.subplot(gs[0])
        pl.title('contour')
        pl.imshow(vimg)
        pl.axis('off')
        
        pl.subplot(gs[1])
        pl.title('curvature')
        pl.imshow(img)
        pl.scatter(cnt[:,0], cnt[:,1], c=cvt, linewidths=0, s=5)
        pl.xlim((0, img.shape[1]))
        pl.ylim((img.shape[0], 0))
        pl.axis('off')
        
    return cvt