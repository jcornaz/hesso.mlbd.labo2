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
	
	return [features, classes]
	
def train_knn( features ):
	for i in range(0,len(meta)-1):
		feature = extract_features( meta.iloc[i] )
		# build input matrix
		# build target matrix
		
	# split dataset
	# normalize dataset
	# build a classifier

def split_tab(features,classes, test_percent=0.25):
	X_split = int(len(X)*test_percent)
	y_split = int(len(y)*test_percent)
	
	X_test  = []#X[X_split:]
	y_test  = []#y[y_split:]
	
	for i in range(0,X_split):
		X_test.append(features.pop(random.randrange(len(features))))# pop a random features
		
	for i in range(0,y_split):
		y_test.append(classes.pop(random.randrange(len(classes))))

	X_train = features#X[0:X_split]
	y_train = classes#y[0:y_split]
	
	return X_train, X_test, y_train, y_test 

