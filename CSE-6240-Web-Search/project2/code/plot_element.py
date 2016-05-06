import matplotlib.pyplot as plt

def plot_first_k_numbers(X, k):
    '''quoted from https://triangleinequality.wordpress.com/2014/08/12/theano-autoencoders-and-mnist/
    
    Used for showing the features learned.
    '''
    m = X.shape[0]
    k = min(m,k)
    j = int(round(k / 10.0))

    fig, ax = plt.subplots(j,10)
    
    for i in range(k):
        w=X[i,:]       
        w=w.reshape(50, 50)
        ax[i/10, i%10].imshow(w, cmap=plt.cm.gist_yarg,
                      interpolation='nearest', aspect='equal')
        ax[i/10, i%10].axis('off')

    plt.tick_params(\
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    plt.tick_params(\
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left='off', 
        right='off',    # ticks along the top edge are off
        labelleft='off')

    fig.show()