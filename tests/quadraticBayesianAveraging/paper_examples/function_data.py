import numpy as np
import pickle, os

# Create the function
def fun(x):
    return (np.square(np.square(x))+10.0*np.sin(x)+10.0*np.sin(5.0*x))[0]
    #return (np.square(np.square(x))+10.0*np.sin(x))[0]

# Time step minimum
h = 0.000001

def fun_all(x,r1=0.0, r2=0.0, r3=0.0, bool_zero=False):
    """
    r1 : additive error on f
    r2 : additive error on grad
    r3 : additive error on Hess
    """
    f = fun(x) + r1
    der = (fun(x+h)-fun(x-h))/(2*h)
    derder = (fun(x+h)+fun(x-h)-2*fun(x))/(h**2)
    grad = np.array([der]) + r2
    Hess = np.array([[derder]]) + r3
    if bool_zero:
        Hess = Hess*0.0
        grad = grad*0.0
    return f, grad, Hess



"""
Initializes all the information need to create consistent plots across
the different modules
"""
# Model to plot
model_ind = 0

# Plot ranges
x_range = [-3.0, 3.0]
y_range = [-150.0, 150.0]

# Number of x points to plot
num_points = 200

# Resolution of saved images
dpi=300

ub = 3.0
lb = -3.0

bounding_box = np.array([[-4.0, 4.0]])

def constr1(x):
    return (ub - x)[0, :]

def constr2(x):
    return (x - lb)[0, :]

if os.path.isfile('function_data.p'):
    # If pickle data is already there, then just use the pickle data
    saved_data = pickle.load(open('function_data.p', 'rb'))
    X_complete = saved_data['X_complete']
    f_complete = saved_data['f_complete']
    grad_complete = saved_data['grad_complete']
    Hess_complete = saved_data['Hess_complete']
else:
    # Generate the data and save into the pickle

    # Sequence of x positions to try
    X_complete = np.array([[-0.7, -0.82, -0.98, 0.11, 0.2, 0.82]])

    # Observations
    f_complete = []
    grad_complete = []
    Hess_complete = []

    for k in xrange(X_complete.shape[1]):
        # Get next
        x_next = X_complete[:,k]
        x = x_next
        if k == 0:
            X = np.array([x_next])
        else:
            X = np.hstack([X, np.array([x_next])])

        # Make up errors
        r1 = (20.0+20*np.sin(x)[0])*(np.random.rand(1,1)[0,0]-0.5)
        r2 = (40.0+40*np.sin(x)[0])*(np.random.rand(1,1)[0,0]-0.5)
        r3 = (60.0+60*np.sin(x)[0])*(np.random.rand(1,1)[0,0]-0.5)

        # Get y, grad, hess, and update corresponding lists
        f, grad, Hess = fun_all(x, r1, r2, r3)

        f_complete.append(f)
        grad_complete.append(grad)
        Hess_complete.append(Hess)
        
    # Put everything into a dict and save
    saved_dict = {'X_complete' : X_complete,
                  'f_complete' : f_complete,
                  'grad_complete' : grad_complete,
                  'Hess_complete' : Hess_complete}
    pickle.dump(saved_dict, open('function_data.p', 'wb'))
