import mlbd
import pylab as pl
from matplotlib import gridspec

def curvature_hist( img, step=10, plot=False, min=0, max=100 ):
	gs = pl.GridSpec( 2, 2 )
	c = mlbd.curvature( img, step, plot, gs )
	if plot :
		pl.subplot( gs[1,:] )
		pl.title( 'histogram of curvatures' )
		pl.hist( c )
	return pl.histogram( c )[0]