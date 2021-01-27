import matplotlib.pyplot as plt
import scipy
from scipy import optimize
import numpy as np 
import warnings
from casadi import *



def estimate(S0,I0,I_true,nr_betas):

    nr_steps = len(I_true)

    x = MX.sym('x',3,1) # System states
    p = MX.sym('p',2,1)
    beta = p[0]
    gamma = p[1]

    Sx = x[0]
    Ix = x[1]
    Rx = x[2]

    # Ode of SIR model
    rhs = vertcat(-beta*Sx*Ix, 
                    beta*Sx*Ix - gamma*Ix,
                    gamma*Ix)

    # ODE declaration with free parameter
    ode = {'x':x,'p':p,'ode':rhs}

    # Construct a Function that integrates over 1s
    F = integrator('F','cvodes',ode,{'tf':1})

    # Optimiz variable
    Z = MX.sym('z',nr_betas + 1,1)
    betas = Z[0:nr_betas]
    gamma = Z[nr_betas]

    sim_steps_per_beta = ceil(nr_steps/nr_betas)

    x = [S0, I0, 0]  # Initial state
    cost = 0
    for k in range(nr_steps):

        beta_indx = int(np.floor(k/sim_steps_per_beta))
        params = vertcat(betas[beta_indx],gamma)

        res = F(x0=x,p=params)
        x = res["xf"]

        Ik = x[1]
        cost += (I_true[k] - Ik)**2


    # NLP declaration
    nlp = {'x':Z,'f':cost}

    # Solve using IPOPT
    
    beta_ub = np.ones(nr_betas)*1e-4
    beta_lb = np.ones(nr_betas)*0
    gamma_ub = 1.5
    gamma_lb = 1e-2

    UB = np.hstack((beta_ub,gamma_ub))
    LB = np.hstack((beta_lb,gamma_lb))

    beta0 = np.ones(nr_betas)*1e-5
    gamma0 = 0.4
    x0 = np.hstack((beta0,gamma0))
    #solver = nlpsol('solver','ipopt',nlp,{'max_iter':100})
    #solver = nlpsol('solver','ipopt',nlp, {'ipopt':{'max_iter':90}})
    #solver = nlpsol('solver','ipopt',nlp, {'ipopt':{'max_iter':60}})
    solver = nlpsol('solver','ipopt',nlp, {'ipopt':{'max_iter':100}})
    res = solver( x0=x0, lbx=LB, ubx=UB)
    Z = res['x']
    Z = np.array(Z)
    betas = Z[0:nr_betas]
    gamma = Z[nr_betas]
    print(res)
    return (np.array(betas),np.array(gamma))




def simulate(S0,I0,betas,gamma,nr_steps):


    x = MX.sym('x',3,1) # System states
    p = MX.sym('p',2,1)
    beta_p = p[0]
    gamma_p = p[1]

    Sx = x[0]
    Ix = x[1]
    #Rx = x[2]

    # Ode of SIR model
    rhs = vertcat(-beta_p*Sx*Ix, 
                    beta_p*Sx*Ix - gamma_p*Ix,
                    gamma_p*Ix)

    print(rhs)

    # ODE declaration with free parameter
    ode = {'x':x,'p':p,'ode':rhs}

    # Construct a Function that integrates over 1s
    F = integrator('F','cvodes',ode,{'tf':1})
    nr_betas = len(betas)
    sim_steps_per_beta = ceil(nr_steps/nr_betas)

    x = [S0, I0, 0]  # Initial state

    S_SIM = [S0]
    I_SIM = [I0]
    R_SIM = [0]
    BETAS = [betas[0]]
    GAMMAS = [gamma]

    for k in range(nr_steps):

        beta_indx = int(np.floor(k/sim_steps_per_beta))

        params = vertcat(betas[beta_indx],gamma)

        res = F(x0=x,p=params)

        x = res["xf"]
        x = np.array(x)

        S_SIM.append(x[0])
        I_SIM.append(x[1])
        R_SIM.append(x[2])
        BETAS.append(params[0])
        GAMMAS.append(params[1])


    return (np.array(S_SIM),np.array(I_SIM), np.array(R_SIM), np.array(BETAS), np.array(GAMMAS))




def simulate_and_pred(S0,I0,betas,gamma,nr_steps,predict_steps=0):


    x = MX.sym('x',3,1) # System states
    p = MX.sym('p',2,1)
    beta_p = p[0]
    gamma_p = p[1]

    Sx = x[0]
    Ix = x[1]
    #Rx = x[2]

    # Ode of SIR model
    rhs = vertcat(-beta_p*Sx*Ix, 
                    beta_p*Sx*Ix - gamma_p*Ix,
                    gamma_p*Ix)

    print(rhs)

    # ODE declaration with free parameter
    ode = {'x':x,'p':p,'ode':rhs}

    # Construct a Function that integrates over 1s
    F = integrator('F','cvodes',ode,{'tf':1})
    nr_betas = len(betas)
    sim_steps_per_beta = ceil(nr_steps/nr_betas)

    x = [S0, I0, 0]  # Initial state

    S_SIM = [S0]
    I_SIM = [I0]
    R_SIM = [0]
    BETAS = [betas[0]]
    GAMMAS = [gamma]

    for k in range(nr_steps):

        beta_indx = int(np.floor(k/sim_steps_per_beta))

        params = vertcat(betas[beta_indx],gamma)

        res = F(x0=x,p=params)

        x = res["xf"]
        x = np.array(x)

        S_SIM.append(x[0])
        I_SIM.append(x[1])
        R_SIM.append(x[2])
        BETAS.append(params[0])
        GAMMAS.append(params[1])


    S_PRED = [x[0]]
    I_PRED = [x[1]]
    R_PRED = [x[2]]
    BETAS_P = [betas[-1]]
    GAMMAS_P = [gamma]
    days_predict = [nr_steps-1]

    # Average delta beta -> use to predict next step
    avg_delta_beta = np.mean(betas[1:] - betas[0:-1])
    

    for k in range(predict_steps):

        k_step =  np.floor(k/sim_steps_per_beta)
        predicted_beta = (k_step+1)*avg_delta_beta + betas[-1]

        params = vertcat(predicted_beta,gamma)

        res = F(x0=x,p=params)

        x = res["xf"]

        S_PRED.append(x[0])
        I_PRED.append(x[1])
        R_PRED.append(x[2])
        BETAS_P.append(params[0])
        GAMMAS_P.append(params[1])
        days_predict.append(k+nr_steps)



    return (np.array(S_SIM),np.array(I_SIM), np.array(R_SIM), 
    np.array(BETAS), np.array(GAMMAS), np.array(S_PRED), np.array(I_PRED),np.array(R_PRED),
    np.array(BETAS_P), np.array(GAMMAS_P), np.array(days_predict))









