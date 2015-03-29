import mlbd
import numpy as np
import pylab as pl
import cv2

def curvature_hist( img, step=10, plot=False, nbins=10, vmin=0, vmax=0.4):     
   cvt = mlbd.curvature(img, step=step)
			
   bins = np.linspace(vmin, vmax, nbins + 1, endpoint=True)
   h, _ = np.histogram(cvt, bins=bins, range=(vmin, vmax))
   h = h / float(len(cvt))

   if plot:
       pl.title('histogram of curvatures')
       pl.bar(bins[:-1], h, width=0.02, align='center')
       pl.xlim((bins[0], bins[-1]))

   return h

def ratio_hull_concave(img):
	cnt = mlbd.extract_contour(img)
	hull = cv2.convexHull(cnt)
	return cv2.contourArea(hull)/cv2.contourArea(cnt)
	
def extract_features( meta_elem ):
	img = mlbd.load_img( meta_elem['basename'] )
	res = curvature_hist( img ).tolist()
	res.append( ratio_hull_concave( img ) )
	res.append( mlbd.eccentricity( img ) )
	return res
	
def extract_dataset( meta ):
	classids = {}
	classid = 0
	
	features = []
	classes = []
	for i in range( 0, len( meta ) - 1 ):
		elem = meta.iloc[i]
		if not elem['classid'] in classids :
			classids[elem['classid']] = classid
			classid = classid + 1
			
	for i in range( 0, len( meta ) - 1 ):
		elem = meta.iloc[i]
		features.append( extract_features( elem ) )
		c = []
		for j in range( 0, len( classids ) ):
			if classids[elem['classid']] == j:
				c.append(1)
			else:
				c.append(0)
		classes.append(c)
	
	return features, classes
	
def train_knn( features, classes ):
	# TODO split dataset
	# TODO normalize dataset
	# TODO build and return classifier