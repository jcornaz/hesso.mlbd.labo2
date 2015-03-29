import mlbd
import numpy as np
import pylab as pl
import cv2
import copy
import random
import sklearn as sk

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
	
def create_classlabel_encoder( meta ):
	le = sk.preprocessing.LabelEncoder()
	
	classids = {}
	
	for i in range( 0, len( meta ) - 1 ):
		elem = meta.iloc[i]
		if not elem['classid'] in classids :
			classids[elem['classid']] = True
	
	le.fit( classids.keys() )
	
	return le
	
def extract_features( meta_elem ):
	img = mlbd.load_img( meta_elem['basename'] )
	res = curvature_hist( img ).tolist()
	res.append( ratio_hull_concave( img ) )
	res.append( mlbd.eccentricity( img ) )
	return np.matrix( res )
	
def extract_dataset( meta, labelEncoder ):
	features = np.zeros((len( meta ), 12))
	classes = np.zeros((len( meta ), 1))
	
	for i in range( 0, len( meta ) - 1 ):
		elem = meta.iloc[i]
		features[i,:] = extract_features( elem )
		classes[i,:] = labelEncoder.transform( elem['classid'] )
	
	return features, classes
	
def train_knn( features, classes ):
	# TODO normalize dataset
	knn = sk.neighbors.KNeighborsClassifier( weights='uniform' )
	knn.fit( features, classes )
	return knn