import parse_excel
import matplotlib.pyplot as plt
import scipy
from scipy import optimize
import numpy as np 
import estimate_model
import filter_data

(dates,nr_cases) = parse_excel.read_iva_excel('covid_data_vg2.xls')
nr_cases = np.array(nr_cases)

#nr_cases = filter_data.smooth_filter(nr_cases)
#nr_cases = filter_data.median_filter(nr_cases)


# Nr of people in Västra Götaland
population = 1.7e6
population = 1e4
nr_steps = len(nr_cases)
nr_var_switch = 10
predict_steps = 60

# Model on raw data

initial_I = nr_cases[0]

(betas,gamma) = estimate_model.estimate(population,initial_I,nr_cases,nr_var_switch)

print(betas)
print(gamma)
(S,I,R,betas,gammas,S_P,I_P,R_P,betas_P,gammas_P,days_P) = estimate_model.simulate_and_pred(population,initial_I,betas,gamma,nr_steps,predict_steps)
print("")

print(S[0])
print(nr_cases[0])
print(I[0])

fig1 = plt.figure()
plt.subplot(3,1,1)
plt.plot(S)
plt.plot(days_P,S_P,'--')
plt.ylabel('S')

plt.subplot(3,1,2)
plt.plot(I)
plt.plot(nr_cases,'k--')
plt.plot(days_P,I_P,'b--')
plt.ylabel('I')
plt.legend(['Model','IVA','Prediction'])

plt.subplot(3,1,3)
plt.plot(R)
plt.plot(days_P,R_P,'--')
plt.ylabel('R')

plt.savefig('SIR.png')

fig2 = plt.figure()
plt.subplot(2,1,1)
plt.plot(betas)
plt.plot(days_P,betas_P)
plt.ylabel('beta')

plt.subplot(2,1,2)
plt.plot(gammas)
plt.plot(days_P,gammas_P)
plt.ylabel('gamma')
plt.savefig('param_traj.png')

fig3 = plt.figure()
plt.plot(nr_cases,'k--')
plt.plot(I)
plt.plot(days_P,I_P,'b--')
plt.ylabel('I')
plt.legend(['IVA','Model','Prediction'])
plt.grid('on')

plt.savefig('predic.png',dpi=300)

plt.show()

