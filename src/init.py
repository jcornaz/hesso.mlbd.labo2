import os
import pandas as pd
import numpy as np
import numpy.ma as ma
import skimage
import skimage.io

np.set_printoptions(precision=5, suppress=True)

DIR = '../statement/data/preprocessed/'
	
np.set_printoptions(precision=5, suppress=True)
meta = pd.io.pickle.read_pickle(os.path.join(DIR, 'meta.pkl'))

def load_img(name):
    """Loads the given image by name and returns a masked array"""
    img = skimage.io.imread(os.path.join(DIR, 'imgs', name + ".png"))
    img = skimage.img_as_float(img)
    img = ma.masked_where(img == 0, img)
    # same mask for all 3 axes
    mask = np.all(img == 0, axis=2)
    img = ma.array(img)
    for i in xrange(img.shape[2]):
        img.mask[:,:,i] = mask
    
    return img