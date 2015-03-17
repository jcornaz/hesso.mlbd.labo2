import mlbd
import pylab as pl
from matplotlib import gridspec

def curvature_hist( img, step=10, plot=False ):
	gs = pl.GridSpec( 2, 2 )
	c = mlbd.curvature( img, step, plot, gs )
	if plot :
		pl.subplot( gs[1,:] )
		pl.title( 'histogram of curvatures' )
		pl.hist( c )
	return pl.histogram( c )[1]

def extract_features( meta_element ):
	pass # TODO
	
def train_knn( features ):
	for i in range(0,len(meta)-1)
		feature = extract_features( meta.iloc[i] )
		# build input matrix
		# build target matrix
		
	# split dataset
	# normalize dataset
	# build a classifier