"""
Reads Image data from train/test folders and creates .pkl objects for models to use later
"""

def get_img(i, size):
    """
    Returns a binary image from my file directory with index i
    """
    img = Image.open('/users/derekjanni/pyocr/train/'+ str(i+1) + '.Bmp')
    img = img.convert("L")
    img = img.resize((size,size))
    image = np.asarray(img)
    image.setflags(write=True)
    thresh = threshold_otsu(image)
    binary = image > thresh
    return binary

if __name__=='main':

    print "Loading Data..."
    
    # data labels live in here
    df = pd.read_csv('trainLabels.csv', header=0)
    X = np.asarray([get_img(i, 50) for i in df.index]).astype(np.float32).reshape(-1, 1, 50, 50)
    Y = np.asarray(df['Class'])
    
    # method for index-shuffle & dataset split
    indices = list(zip(X, Y)) 
    random.shuffle(indices)
    X, Y = zip(*indices)
    X = np.asarray(X)
    Y = np.asarray(Y)
       
    # partition data with about 4/5 training, 1/5 test 
    X_train, X_test, Y_train, Y_test = X[:5000], X[5000:], Y[:5000], Y[5000:]

    print "Saving to File..."

    with open('classifier_data.pkl', 'w') as outfile:
        pickle.dump([X_train, Y_train, X_test, Y_test], outfile)
    
    print "Done"
