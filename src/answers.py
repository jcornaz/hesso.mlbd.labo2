import mlbd
import numpy as np
import pylab as pl
import cv2 as cv
import sklearn as sk
import sklearn.metrics as skm
import matplotlib.cm as cm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from pybrain.datasets import ClassificationDataSet
from sklearn.preprocessing import MultiLabelBinarizer
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.utilities           import percentError

def curvature_hist( img, step=10, plot=False, nbins=10, vmin=0.03, vmax=0.3 ):    
	curvatures = mlbd.curvature( img, step=step )
	bins = np.linspace( vmin, vmax, nbins + 1 )
	res, _ = np.histogram( curvatures, bins=bins, range=( vmin, vmax ) )
	res = res / float( len( curvatures ) )
	
	if plot:
		pl.title( 'histogram of curvatures' )
		pl.bar( bins[1:], res, width=0.02, align='center' )
		pl.xlim( (vmin, vmax) )
	
	return res

def ratio_hull_concave(img):
	cnt = mlbd.extract_contour(img)
	hull = cv.convexHull(cnt)
	return cv.contourArea(hull) / cv.contourArea(cnt)
	
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
	res = np.zeros((1,12))
	res[0,0:10] = curvature_hist( img ).reshape((1,10))
	res[0,10] = ratio_hull_concave( img )
	res[0,11] = mlbd.eccentricity( img )
	return res

def normalize( features ):
	nbi, nbf = features.shape
	res = np.zeros((nbi,nbf))
	for i in range(0,nbf-1):
		scaler = MinMaxScaler()
		scaler.fit( features[:,i] )
		res[:,i] = scaler.transform( features[:,i] )
	return res
	
def extract_dataset( meta, labelEncoder ):
	features = np.zeros((len( meta ), 12))
	classes = np.zeros((len( meta ), 1))
	
	for i in range( 0, len( meta ) - 1 ):
		elem = meta.iloc[i]
		features[i,:] = extract_features( elem )
		classes[i,:] = labelEncoder.transform( elem['classid'] )
	
	return normalize( features ), classes
	
def train_knn( features, classes ):
	knn = KNeighborsClassifier()
	knn.fit( features, classes[:,0] )
	return knn

def train_fnn( ds_train, ds_test, n_hidden_units=17, lr_param=0.00001, momentum_param=0.9999, n_epochs=90, bias_param=True,  plot=True ):

	fnn = buildNetwork(ds_train.indim, n_hidden_units, ds_train.outdim, outclass=SoftmaxLayer,bias=bias_param)

	results = []

	trainer = BackpropTrainer(fnn, ds_train, learningrate = lr_param, momentum = momentum_param )
	epochs = range(0,n_epochs)

	for i in epochs:
		  trainer.trainEpochs(1)
		  result_train = percentError(trainer.testOnClassData(),ds_train['class'] )
		  result_test = percentError(trainer.testOnClassData(dataset=ds_test ), ds_test['class'] )
		  results.append([result_train, result_test])

	if plot:
		pl.figure( figsize=(10,10) )
		pl.plot( epochs, results)
	
	return fnn
	
	
def plot_report( y_pred, y_true, labelEncoder ):
	report = skm.classification_report( y_true, y_pred, labels=np.arange(len(labelEncoder.classes_)), target_names=labelEncoder.classes_)
	confmat = skm.confusion_matrix( y_true, y_pred )
	pl.figure(figsize=(10, 10))
	mlbd.plot_confusion_matrix(confmat, labelEncoder.classes_, cmap=cm.gray_r)
	print report
	return report
