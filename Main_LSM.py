# LSM Valuation of BDH Example
# This program implements valuation of real options for a partiular oil production project, called BDH, 
# using the LSM valuation technique (Longstaff & Schwartz, 2005)
###############################################################################
# Description of input variables

# Input parameters: 
# 1. Oil field input parameters
# period            Number of periods of production, years
# reserve           Initial reserve, MMBD
# decline           Yearly decline rate, fraction
# prod_level        Initial yearly production level, MMBD
# op_cost           Variable operational cost per barrel of oil at Year 0, $
# op_cost_rate      Yearly growth rate for variable operational cost, fraction 
# price             Initial oil price per barrel, $
# price_growth      Yearly growth rate of oil price, fraction
# fixed_cost        Yearly fixed cost, MM$
# profit            Profit sharing rate, fraction
# init_invest       Up=front investment, MM$

# 2. Price model input Parameters:
# p0            start price at time 0
# mu            annulazied drift parameter of GBM process
# sigma         annulazied volatility parameter of GBM process
# trialno       number of simulation trials or paths
# n             granularity of time-period (also called dt)
# T             total time in years (also called deltaT)
# rev_rate      Reversion speed per annum (also called as eta)
# mean_level    Long term equilibrium price, $
# strike        strike price
# rf            risk-free rate
# dr            Risk-adjusted discount rate
# DR            discount rate coefficient (=1/(1+r))
################################################################################
# Importing required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import sys
import os
import importlib
import random
from mpl_toolkits.mplot3d import Axes3D

# Appending the current working directory to the system path of python
curr_path=os.getcwd()
sys.path.append(curr_path)
# Importing LSM module
import realOptionValuation_LSM_PathAlternatives
importlib.reload(realOptionValuation_LSM_PathAlternatives)
from realOptionValuation_LSM_PathAlternatives import ROV_with_LSM
################################################################################
# All input variables related to BDH example
# 1. Oil field input parameters
period=10
reserve=90
prod_level=9
decline=0.15
op_cost=10
op_cost_rate=0.02
price=25
price_growth=0.03
fixed_cost=5
profit=0.25
init_invest=180

# 2. Price model input parameters
# 2.1. Oil price model (OP)
p0_OP=price

# The price model can be chosen to be either a Geometrical Brownian Motion or a Mean Reversion model

# Specify the valaution approach in terms of risk preference (could be either RN (risk-neutral) or RA (risk-adjusted) approach):
valuation_approach='RN'
# 2.1.1. Geometrical Brownian Motion (GBM) Model
if valuation_approach=='RA':
    mu_GBM_OP=price_growth
elif valuation_approach=='RN':
    mu_GBM_OP=0     # Risk-neutral evaluation
sigma_GBM_OP=decline
# 2.1.2. Mean Reversion (MR) Model
sigma_MR_OP=sigma_GBM_OP
mean_level_OP=0.8*p0_OP     # The factor 0.8 is selected heuristically to show a lower mean equilibrium level than the initial price
reversion_rate_OP=0.35      # Selected heuristically
# 2.2. Variable operating cost model (VOC)
p0_VOC=op_cost
# 2.2.1. Geometrical Brownian Motion (GBM) Model
mu_GBM_VOC=op_cost_rate
sigma_GBM_VOC=0.1
# 2.2.2. Mean Reversion (MR) Model
sigma_MR_VOC=sigma_GBM_VOC
mean_level_VOC=0.8*p0_VOC   # The factor 0.8 is selected heuristically to show a lower mean equilibrium level than the initial price
reversion_rate_VOC=0.35     # Selected heuristically
# 2.2. Variable operating cost model (VOC)

# 3. Model configuration
trialno=10000
n=1
T=10
rf=0.05
dr=0.1
DR=1/(1+rf)         # risk-free discount rate
step = int(T/n)     # Total number of time steps
################################################################################
# Additional settings
plot_option='N'     # No plots for price models are created (it can take either 'Y' or 'N' values)

# Stochastic models for future price simulation
all_price_model=['GBM','MR']        # GBM: Geometrical Brownian Motion, MR: Mean Reversion
price_model=all_price_model[0]      # Default price model is "GBM"

# Regression models: There are 4 different models that could be selected:
# 1. K-fold Cross Validation (CV) that uses k=5 splits of data when regressing (Mean absolute percentage error is utilized as the regression performance)
# 2. Gradient Boosting (Mean absolute percentage error is utilized as the regression performance)
# 3. Polynomial Regression that selectes the best polynomial order from 1 to 9 based on MSE criterion
# 4. Feature Selection that tries to generate all possible cross combinations of all unceratin variables (by increasing powers of each variable from 1 to 9) as new features. It then picks up the best and most-informative features by minimizing the MSE measure
allRegModels=['kfold','GradientBoosting','Polynomial','FeatureSelection']
modelName=allRegModels[2]       # Default case is "Polynomial" regression

# Path selection for regression equation
# There are 5 path alternatives that could be employed in LSM technique:
# 1. Alternative1 means that both options should be in the money at the current period
# 2. Alternative2 requires that the options to be in the money either at the current period or later on until the expiration date
# 3. Alternative3 means that either of the options could be in the money at the current period
# 4. Alternative4 means that only paths that have positive cash flows at the current time period are in-the-money and should be selected
# 5. Alternative5 means that all the paths are considered regardless of being in or out of the money
path_type=['alternative1','alternative2','alternative3','alternative4','alternative5']
path=path_type[-1]      # Default case is "alternative5" where all paths are considered

# Setting the number of replications for running LSM simulation repeatedly and then averaging over all trials:
replication=10
################################################################################
# Combining input properties into integrated arrays for passing into the LSM valuation class
# Integrating oil field properties
oilFieldProps=[period,reserve,prod_level,decline,op_cost,op_cost_rate,price,price_growth,fixed_cost,profit,init_invest]
# Integrating oil price properties
oilPriceProps=[p0_OP,mu_GBM_OP,sigma_GBM_OP,sigma_MR_OP,mean_level_OP,reversion_rate_OP]
# Integrating variable operating cost properties
varOpCostProps=[p0_VOC,mu_GBM_VOC,sigma_GBM_VOC,sigma_MR_VOC,mean_level_VOC,reversion_rate_VOC]
# Integrating price model configurations
priceModelConfig=[trialno,n,T,rf,dr]
################################################################################
# Flag setting (There are different flags defined in this code to specify different tasks or options. Each flag can be assigned only two logical values including "True" or "False".)
# 1. Defining conditions when no option is considered in the model (Base case without option) (True: No option case, False: Options will be considered later)
flag_noOption=False          
# 2. The "seed" (which is used for random number generation in stochatic price models) could be set either constant (in case we want to run the simulation using differetn scenarios but keeping the same price predictions in all scenarios) or random (defualt case) (True: Constant seed is used (seed is generated prior to passing to the stochastic price models), False: Random seed is used within the stochastic price models)
flag_constantSeed=False
# 3. When constant "seed" is to be used, it could be loaded from a csv file (which has been generated and saved previously) or generted at the moment and then saved for later use (True: seed is loaded from a csv file, False: seed is generated and then passed to the stochastic model; the seed is saved in a csv file in this case so that it can be loaded and re-used later on)
flag_loadSeed=False
# 4. Specifying whether the scatter plot of "Continuation Values" vs. "Uncertain Variables" is plotted in 3D or not (True: 3D view is generated, False: 2D plot is generated; in this case only one of uncertain variables could be plotted)
flag_3D=True
################################################################################# Real option valuation using LSM approach
# 1. Option definition
if flag_noOption:
    # Defining no options
    option_dict={}
else:
    # definition of option dictionary: The list of options must be written in the form of a dictionary with the period number as the key, and another dictionary as the value. The nested value dictionary includes option label as the key and option expense and cash flow reward (in the form of a list) as the value.
    # Original BDH example: Three options including Buyout, Divest and Continuation are considered at Year 5 (which is equivalent to index 4 in Python format, starting from 0). For divest option, there is no additional cost, but there is $100 milltion as reward. For buyout option, we can buy out other 25% partnership (we already own 75% of the field) at the expense of $40 million. Continuing the project has no cost or profit since it is not a real option in reality (but we have included "continue" as an option for more simplicity)
    # Option at Year 5
    option_dict={'4':{'continue':[0,0],'buy out':[40,25],'divest':[0,100]}}
    # if you want to add options to more than one period (e.g. to periods Year 5 and Year 7), then use the following format (simply add another dictionary to the option_dict)
    # option_dict={'4':{'continue':[0,0],'buy out':[40,25],'divest':[0,100]},'6':{'continue':[0,0],'buy out':[40,25],'divest':[0,100]}}

# 2. LSM Valuation
# 2.1. seed settings for random number generation in stochastic price models
if flag_constantSeed:
    if flag_loadSeed:
        # Loading seed array from csv file
        from numpy import asarray
        from numpy import loadtxt
        SEED = loadtxt('seed.csv', delimiter=',')
        seed=SEED.tolist()
    else:
        # Defining random seeds for random number generation in GBM approximation
        seed_array=[]
        seed_OP=[random.randint(0,2**32-1) for i in range(trialno)]
        seed_VOC=[random.randint(0,2**32-1) for i in range(trialno)]
        seed=[seed_OP,seed_VOC]
        seed_array.append(seed)
        seed=seed_array[0]
        # Saving seed Array to CSV File for later use
        from numpy import asarray
        from numpy import savetxt
        savetxt('seed.csv', seed, delimiter=',')
else:
    seed='False'

# 2.3. Making an instance of the LSM valuation class

# 2.3.1. Definition of output variables
# NPV                   Final NPV of the project (with or without options, depending on the defintion of real options)
# cash_flow             Cash flow matrix containing the amount of cash flows at all periods (denoted by columns) for all simulation trials (represented by rows. It is independent of options since it is generated prior to LSM valuation, however it is created based on evolution of uncertain variables using stochastic price models
# casg_flow_matrix      Cash flow matrix as employed in Longstaff & Schwartz notation. It contains the project value at each period (denoted by columns) for all simulation trials (represnted by rows)
# stopping_rule         Stopping rule as employed in Longstaff & Schwartz notation. It displays the earliest exercise date of real options alongn all periods (denoted by columns) for all simulation trials (represnted by rows). It contains binary values where 1 indicates "option exercise" and 0 denotes no exercise.
# continuation          Continuation values for the periods containing option. It contains the continuation values for all simulation trials.
# reg_model             Properties and outputs of regression model including:
    # ITM_ratio             The ratio of ITM paths to all paths for all periods
    # reg_measure           The regression performance measure (mape or mse) for all periods
    # best_order            Best polynomial order used in best regression model
    # coef                  Coefficients of best regression model
    # model                 Best regression model (It can be used to predict the continuation values vs. the independent variables employed in making the regression function)
    # score                 Correlation coefficient offered by best regression equation
    # exercise@all_periods  Label of optimal policy for each period along all simulation trials
    # x                     X values used in tuning regression function (the length of X is equal to the number of ITM paths)
    # y                     Y values used in tuning regression function (the length of Y is equal to the number of ITM paths)

# Initializing an empty list for storing NPV values for all LSM replications
NPVi=[]
# When more than one run is executed (replication>1), the obtained results (except for NPV) are not stored per each replication. Therefore only the results for the last replication are available for more analysis.
# 2.3.2. 
solution_LSM=ROV_with_LSM(oilFieldProps,oilPriceProps,varOpCostProps,priceModelConfig,plot_option,price_model)
for i in range(replication):
    # LSM valuation 
    NPV,cash_flow,cash_flow_matrix,stopping_rule,continuation,reg_model=solution_LSM.LSM_Application(seed,option_dict,modelName,which_path=path)
    # Storing NPV values
    NPVi.append(NPV)
# Averaging NPV over all replications
NPV_final=np.mean(NPVi)
print('Final NPV = ',NPV_final)
# Printing optimal policies for all periods
df_ex=pd.DataFrame(reg_model['exercise@all_periods'])
df_ex.columns = [f"T{str(i)}" for i in range(1,step+1)]
print('Optimal policies at all periods are as fllows \n', df_ex)
################################################################################
# Scatter plot of Continuation Value vs. Uncertain Variables along with the regression plane (in 3D) or regression line (in 2D)
# Extracting x and y data from output arrays
# Extracting the number of periods with option
if not flag_noOption:
    n_option=len(option_dict)
    xs=reg_model['x']       # Containing uncertain variables including oil price and variable operating cost
    ys=reg_model['y']       # Containing continuation values for the period with option
    if n_option==1:
        xs1=xs[0,:]         # Oil price data
        xs2=xs[1,:]         # Variable operating cost data
    elif n_option>1:
        # When multiple periods contain options, then we should select a specific period first and then plot the results for that single period
        period_of_interest=0                # First period
        xs1=xs[period_of_interest][0,:]     # Oil price data 
        xs2=xs[period_of_interest][1,:]     # Variable operating cost data
        ys=ys[period_of_interest] 
    # Making regression plane (for 3D plot)
    xd1=np.arange(min(xs1),max(xs1),0.1)
    xd2=np.arange(min(xs2),max(xs2),0.1)
    # Making a mesh grid from xd1 and xd2
    xm1,xm2 = np.meshgrid(xd1, xd2)
    if n_option==1:
        reg=reg_model['model']
    elif n_option>1:
        reg=reg_model['model'][period_of_interest]
    yd=np.zeros(xm1.shape)
    for i in range(xm1.shape[0]):
        for j in range(xm1.shape[1]):
            xx=np.array([xm1[i,j],xm2[i,j]])
            # predicting Y data using the tuned regression model to generate the regression plane (3D plot)
            yd[i,j]=reg.predict(xx.reshape(1,-1))
    # Making regression line (for 2D plot)
    xd1=np.linspace(min(xs1),max(xs1),1000)
    xd2=np.linspace(min(xs2),max(xs2),1000)
    xd=np.concatenate((xd1.reshape(-1,1),xd2.reshape(-1,1)),axis=1)
    yd2D=reg.predict(xd)
    fig = plt.figure()
    if flag_3D:
        # 3D plot
        ax = Axes3D(fig)
        # Plotting scatter plot of continuation value vs. oil price and operating cost in 3D
        ax.scatter(xs1,xs2,ys) 
        ax.set_ylabel("Variable Operating Cost ($)")
        ax.set_zlabel("Continuation Value ($)")
        ax.plot_surface(xm1, xm2, yd, alpha=0.5, color=[0,1,0])
    else:
        # 2D plot
        ax = fig.add_subplot(111)
        # Plotting scatter plot of continuation value vs. oil price in 2D
        ax.scatter(xs1,ys)      
        ax.set_ylabel("Continuation Value ($)")
        ax.plot(xd1,yd2D,alpha=1.0,c='r',linewidth=3)
    ax.set_xlabel("Oil Price ($)")
    plt.show()
    ############################################################################