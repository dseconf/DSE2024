
# import packages used
import numpy as np
from scipy import interpolate


def setup():   
    """
    The function "setup" initializes and sets up the parameters and grids for a the simple Zurcher model.
    
    Returns:
      the parameter object `par` which contains various parameters and grids used in the code.
    """

    class par: pass
    
    # Spaces
    par.n = 175                      # Number of grid points
    par.max = 450                    # Max of mileage

    # structual parameters
    par.p = np.array([0.0937, 0.4475, 0.4459, 0.0127])   # Transition probability
    par.RC = 11.7257                                     # Replacement cost
    par.c = 2.45569 * 0.001                              # Cost parameter
    par.beta = 0.99                                    # Discount factor
    
    # Parameters for VFI
    par.max_iter = 10000   # maximum number of iterations
    par.tol = 1.0e-8 #convergence tol. level 
    
    # Create grid
    par.grid = np.arange(0,par.n) # milage grid
    
    # Create transition arrays
    par.transition = np.array([0,1,2]) # Transition grid
    par.p = np.array([1/3, 1/3, 1/3]) # Transition probability grid
    
    # Simulated extreme value taste shocks
    np.random.seed(1987)
    par.num_eps = 10000
    par.sigma_eps = 1.0
    par.eps_keep_gumb = np.random.gumbel(loc=-par.sigma_eps * np.euler_gamma,scale=par.sigma_eps,size=par.num_eps)
    par.eps_replace_gumb = np.random.gumbel(loc=-par.sigma_eps * np.euler_gamma, scale=par.sigma_eps,size=par.num_eps)
    
    # Gaussian taste shocks
    par.eps_keep_norm = np.random.normal(0,1,par.num_eps)
    par.eps_replace_norm = np.random.normal(0,1,par.num_eps)
    
    # For vectorized
    par.P1 = create_transition_matrix(0, par)
    par.P2 = create_transition_matrix(1, par)
    par.cost_grid = cost(par.grid, par)
        
    return par

def solve_SA(par, vectorized = False, **kwargs):
    """
    The function `solve_SA` solves a performs successive approximations(aka. value function iteration)    
    Args:
      par: The parameter `par` is an object that contains various parameters
      vectorized: The "vectorized" parameter is a boolean flag that determines whether to use a
    vectorized implementation of the Bellman equation solver or not. If set to True, the function
    "bellman_vector" will be used to solve the Bellman equation. If set to False, the function "bellman.
    Defaults to False
    
    Returns:
      an object of the class `sol`, which contains the following attributes:
    - `V`: a numpy array representing the value function
    - `pk`: a numpy array representing the policy function
    - `it`: an integer representing the number of iterations performed
    - `delta`: a float representing the difference between the current and previous value functions
    """
    
    class sol: pass
    sol.V = np.zeros([par.n]) #arbitrary starting values
    sol.pk = np.zeros([par.n]) #arbitrary starting values
    sol.it = 0
    sol.delta = 2000
        
    while (par.max_iter>= sol.it and par.tol<sol.delta):
        
        if vectorized==False:
            V_now, pk = bellman(sol.V, par, **kwargs)
        else:
            V_now, pk = bellman_vector(sol.V, par, **kwargs)

        sol.delta = np.amax(np.abs(V_now - sol.V))
        sol.it += 1
        sol.V = V_now
        sol.pk = pk
    print(f'Finished after {sol.it} iterations')
    print(f'Convergence achieved: {sol.delta < par.tol}')
    
    return sol



def bellman(V_next, par, taste_shocks = 'None', stochastic_transition = False):
    """
    The function `bellman` evaluates the integreated value bellmann-operator in a dynamic programming problem.
    ie. For a given guess of the value function for the next period, it calculates a new guess of the value and also the choice probabilities.
    problem using the Bellman equation, considering different types of taste shocks and transition
    probabilities.
    
    Args:
      V_next: The value function for the next period, which is a numpy array of size par.n.
      par: The parameter object `par` contains various parameters.
      taste_shocks: The parameter "taste_shocks" determines the type of taste shocks used in the
    calculation.
      stochastic_transition: The parameter "stochastic_transition" determines whether the transition
    probabilities between states are stochastic or deterministic. If it is set to False, the transition
    probabilities are deterministic. If it is set to True, the transition probabilities are stochastic.
    Defaults to False
    
    Returns:
      The function `bellman` returns two arrays: `V_now` and `pk`.
    """
    
    V_now = np.zeros([par.n]) # Initialize value function array
    pk = np.zeros([par.n]) # Initialize choice probability array
    
    for  x in par.grid: # Loop over states
            
        # Calculate expected future value across states for each choice
        if stochastic_transition == False:
            pass # Fill in using EV_deterministic
        else: #Exercise 2
            pass # Fill in using EV_stochastic
        
        # Calculate value of each choice
        value_keep = None # Fill in
        value_replace = None # Fill in
        
        # Find the maximum value across choices
        pass # Fill in
        
        ### Update value and choice
        
        # Exercise 1
        if taste_shocks == 'None':
            pass # Fill in
        
        # Exercise 3
        elif taste_shocks == 'Extreme Value':
            pass # Fill in
        
        # Exercise 4
        elif taste_shocks == 'Monte Carlo Extreme Value':
            values = np.column_stack([value_keep + par.eps_keep_gumb, value_replace + par.eps_replace_gumb])
            choices = np.argmax(values, axis = 1)

            V_now[x] = values[np.arange(par.num_eps), choices].mean()
            pk[x] = 1 - choices.mean()
        
        # Exercise 4
        elif taste_shocks == 'Normal':
            pass # Fill in
            
    return V_now, pk


def cost(x, par):
    """
    The cost function calculates the cost of bus maintenance.
    
    Args:
      x: The parameter "x" represents the current state of the system.
      par: The parameter object `par` contains various parameters.

    Returns:
      The cost of bus maintenance.
    """
    return par.c * x 

def util(d, x, par):
    """
    The function calculates the utility based on the choice and state.
    
    Args:
      d: The parameter "d" represents a decision variable.
      x: The parameter `x` is a variable that represents a state. 
      par: The parameter object `par` contains various parameters.
    
    Returns:
      The utility of a choice given the state.
    """
    
    if d == 0:
        return - cost(x, par)
    if d == 1:
        return - par.RC - cost(0, par)

def EV_deterministic(d, x, V_next, par):
    """
    The function `EV_deterministic` calculates the expected value of the next period's value function
    given the current state and decision.
    
    Args:
      d: The parameter "d" represents the probability of an event occurring. It is a value between 0 and
    1, where 0 means the event will not occur and 1 means the event will definitely occur.
      x: The parameter "x" represents the current state of the system.
      V_next: The parameter V_next represents the value function for the next period. It is a vector
    that contains the value of each state in the next period.
      par: The parameter object `par` contains various parameters.
    
    Returns:
      the value of next period
    """
    x_next = None # fill in
    x_next = np.fmin(x_next, par.n-1) # Ensure that x_next is within grid
    return None # Fill in

def EV_stochastic(d, x, V_next, par):
    """
    The function `EV_stochastic` calculates the expected value of a variable `V_next` given a set of
    probabilities `par.p` and transition probabilities `par.transition`.
    
    Args:
      d: The parameter "d" represents the probability of an event occurring. It is a value between 0 and
    1, where 0 means the event will not occur and 1 means the event will definitely occur.
      x: The parameter "x" represents the current state of the system.
      V_next: The parameter V_next represents the value function for the next period. It is a vector
    that contains the value of each state in the next period.
      par: The parameter object `par` contains various parameters.
    
    Returns:
      the expected value (EV) of the next period's value function (V_next) given the current state (x),
    the decision variable (d), and the transition probabilities (par.transition) and probabilities
    (par.p).
    """
    pass # Fill in
    
    return None # Fill in

def bellman_vector(V_next, par):
    """
    The function `bellman_vector` calculates the value and choice probabilities for each choice in a
    dynamic programming problem using the Bellman equation.
    
    Args:
      V_next: V_next is a vector representing the value function in the next period. It has dimensions
    (n, 1), where n is the number of states in the model. Each element of V_next represents the value of
    being in a particular state in the next period.
      par: The parameter `par` is an object that contains various parameters.
    
    Returns:
      the updated value of the current state (V_now) and the probability of choosing to replace the
    current state (pk).
    """
    
    pass # Fill in
            
    return None # Fill in

def create_transition_matrix(d, par):
    """
    The function `create_transition_matrix` creates a transition matrix based on the given parameters,
    where the transition probabilities depend on the choice.
    
    Args:
      d: The parameter "d" is used to determine the type of transition matrix to create. If "d" is 0, it
    means we want to create a transition matrix for the "keep" choice. If "d" is 1, it means we want
    to create a transition matrix for the 'replace' choice.
      par: The parameter `par` is an object that contains various parameters.

    
    Returns:
      the transition matrix, which is a numpy array of shape (n, n).
    """
    p = np.append(par.p,1-np.sum(par.p)) # Get transition probabilities
    P = np.zeros((par.n,par.n)) # Initialize transition matrix
    
    if d == 0: # keep
        # Loop over rows
        for i in range(par.n):
            # Check if p vector fits entirely
            if i <= par.n-len(p):
                P[i][i:i+len(p)]=p
            else:
                P[i][i:] = p[:par.n-len(p)-i]
                P[i][-1] = 1.0-P[i][:-1].sum()

    if d == 1: # replace
        # Loop over rows
        for i in range(par.n):
            P[i][:len(p)]=p
    
    return P
