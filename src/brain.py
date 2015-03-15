import pybrain
from pybrain.datasets import ClassificationDataSet
from sklearn.preprocessing import MultiLabelBinarizer

# turns integer labels vector like [0, 2, 3, 2] into a binary
# matrix [[1, 0, 0, 0],
#         [0, 0, 1, 0],
#         [0, 0, 0, 1],
#         [0, 0, 1, 0]]
binarizer = MultiLabelBinarizer().fit(y.reshape(-1, 1))

def pybrain_ds_from_Xy(X, y, binarizer):
    """
    Args:
        binarizer: A MultiLabelBinarizer instance
    """
    ds = ClassificationDataSet(X.shape[1], nb_classes=len(lencoder.classes_), class_labels=lencoder.classes_)
    ds.setField('input', X)
    ds.setField('class', y.reshape(-1, 1))
    ds.setField('target', binarizer.transform(y.reshape(-1, 1)))
    
    #print '--'
    #print "class  :\n", ds['class'][5:10]
    #print "target :\n", ds['target'][5:10]
    #print "y      :\n", y[5:10]
    
    return ds

def print_ds(ds):
    print "Number of training patterns: ", len(ds)
    print "Input and output dimensions: ", ds.indim, ds.outdim
    print "First sample (input, target, class):"
    print ds['input'][10], ds['target'][10], ds['class'][10]
    print ds['input'][55], ds['target'][55], ds['class'][55]

ds_train = pybrain_ds_from_Xy(X_train, y_train, binarizer)
ds_test = pybrain_ds_from_Xy(X_test, y_test, binarizer)