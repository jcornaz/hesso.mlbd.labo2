import mlbd
import pylab as pl
from matplotlib import gridspec

def curvature_hist( img, step=10, plot=False ):
	gs = pl.GridSpec( 2, 2 )
	c = mlbd.curvature( img, step, plot, gs )
	res = pl.histogram( c )
	if plot :
		pl.subplot( gs[1,:] )
		pl.title( 'histogram of curvatures' )
		pl.hist( c )
	return res[1]