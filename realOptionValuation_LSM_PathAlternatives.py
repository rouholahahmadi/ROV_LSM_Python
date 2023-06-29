class ROV_with_LSM:
    """ This class solves any real option valuation problem using LSM technique."""
    # Oil field input parameters
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

    # Price Model input Parameters:
    # p0            start price at time 0
    # mu            annulazied drift parameter of GBM process
    # sigma         annulazied volatility parameter of GBM process
    # trialno       number of simulation trials or paths
    # n             granularity of time-period (also called dt)
    # T             total time in years (also called deltaT)
    # rev_rate      Reversion speed per annum (also called as eta)
    # mean_level    Long term equilibrium price, $
    # rf            risk-free rate
    # dr            Risk-adjusted discount rate
    # DR            risk-free discount rate coefficient (=1/(1+rf))

    def __init__(self,oilFieldProps,oilPriceProps,varOpCostProps,priceModelConfig,plot_option,price_model):
        # Extracting oil field properties
        period,reserve,prod_level,decline,op_cost,op_cost_rate,price,price_growth,fixed_cost,profit,init_invest=oilFieldProps
        # Extracting oil price properties
        p0_OP,mu_GBM_OP,sigma_GBM_OP,sigma_MR_OP,mean_level_OP,reversion_rate_OP=oilPriceProps
        # Extracting variable operating cost properties
        p0_VOC,mu_GBM_VOC,sigma_GBM_VOC,sigma_MR_VOC,mean_level_VOC,reversion_rate_VOC=varOpCostProps
        # Extracting price model configurations
        trialno,n,T,rf,dr=priceModelConfig

        # Oil field input parameters
        self.period=period
        self.reserve=reserve
        self.prod_level=prod_level
        self.decline=decline
        self.op_cost=op_cost
        self.op_cost_rate=op_cost_rate
        self.price=price
        self.price_growth=price_growth
        self.fixed_cost=fixed_cost
        self.profit=profit
        self.init_invest=init_invest
        # Price model input parameters
        # 1. Oil price model (OP)
        self.p0_OP=p0_OP
        self.mu_GBM_OP=mu_GBM_OP
        self.sigma_GBM_OP=sigma_GBM_OP
        self.sigma_MR_OP=sigma_MR_OP
        self.mean_level_OP=mean_level_OP
        self.reversion_rate_OP=reversion_rate_OP
        # 2. Variable operating cost model (VOC)
        self.p0_VOC=p0_VOC
        self.mu_GBM_VOC=mu_GBM_VOC
        self.sigma_GBM_VOC=sigma_GBM_VOC
        self.sigma_MR_VOC=sigma_MR_VOC
        self.mean_level_VOC=mean_level_VOC
        self.reversion_rate_VOC=reversion_rate_VOC
        # 3. Price model configuration
        self.trialno=trialno
        self.n=n
        self.T=T
        self.rf=rf
        self.dr=dr
        self.DR=1/(1+self.rf)
        self.step = int(self.T/self.n)                # Number of time steps
        # 4.Some settings
        self.plot_option=plot_option        # No plots for price models are created (it can take either 'Y' or 'N' values)
        self.price_model=price_model

    def LSM_Application(self,seed,option_dict,modelName,which_path):
        """This function allpies the LSM methodology to solve a real option valuation problem."""
        # Input parameters
        # seed                  Determines if random seed should be constant or changed randomly 
        # option_dict           List of all real options together with their properties in the form of a nested dictionary (key: Option Period, value: a dictionary with the option names as the keys and the option properties as their corresponding values)
        # modelName             Type of model used in linear regression (could be either "Polynomial" or "FeatureSelection")
        # which_path            Determines which paths should be considered for regression in LSM approach (it could be either 'in_the_money' or 'all')

        # Converting seed to a list so that it contains a separtae value for each trial for each uncertain variable
        if not isinstance(seed,(list)):
            seed_OP=[seed for i in range(self.trialno)]
            seed_VOC=[seed for i in range(self.trialno)]
            seed=[seed_OP,seed_VOC]
        self.seed=seed
        self.option_dict=option_dict
        self.modelName=modelName
        self.which_path=which_path

        # Importing required packages
        import pandas as pd
        import numpy as np
        import sys
        import os
        import random
        import matplotlib.pyplot as plt 
        from gekko import GEKKO
        from scipy import stats
        import importlib
        import copy
        import time
        import math
        # Appending the current working directory to the system path of python
        curr_path=os.getcwd()
        sys.path.append(curr_path)
        # Importing module containing class "PriceSimulation"
        from GBMsimLSMvaluation import PriceSimulation,GBM_Model
        from table_generation import table
        # Importing module containing class "PriceModel"
        from MeanReversionModel import PriceModel
        # Importing module containing class "LinearRegression"
        from regression import LinearRegression

        # Generating random stock prices using stochastic models and then solving for optimal policy using LSM technique
        ########################################################################
        if self.price_model=='GBM':
            # 1. Simulating market uncertainties using GBM approximation
            # 1.1. Oil price model (OP)
            # Initializing price list
            pt_GBM_list_OP=[]
            # Please note that self.seed[0] contains random seed values for oil price model
            for i in range(self.trialno):
                simdata_GBM=PriceModel(self.p0_OP,self.rf,self.dr,self.T,self.n,self.seed[0][i])
                pt_GBM_OP=simdata_GBM.GBM(self.mu_GBM_OP,self.sigma_GBM_OP)
                pt_GBM_list_OP.append(pt_GBM_OP)
            # Creating plot title for this model
            title_GBM_OP = f"{self.trialno} Simulated Oil Price Paths Using GBM Approximation (drift rate, $\\mu$={self.mu_GBM_OP}, Volatility, $\\sigma$={self.sigma_GBM_OP})"

            # 1.2. Variable operating cost model (VOC)
            # Initializing price list
            pt_GBM_list_VOC=[]
            # Please note that self.seed[1] contains random seed values for variable operating cost model
            for i in range(self.trialno):
                simdata_GBM=PriceModel(self.p0_VOC,self.rf,self.dr,self.T,self.n,self.seed[1][i])
                pt_GBM_VOC=simdata_GBM.GBM(self.mu_GBM_VOC,self.sigma_GBM_VOC)
                pt_GBM_list_VOC.append(pt_GBM_VOC)
            # Creating plot title for this model
            title_GBM_VOC = f"{self.trialno} Simulated Operating Cost Paths Using GBM Approximation (drift rate, $\\mu$={self.mu_GBM_VOC}, Volatility, $\\sigma$={self.sigma_GBM_VOC})"
            y_plot_VOC=pt_GBM_list_VOC
            ####################################################################
        elif self.price_model=='MR':
            # 2. Simulating market uncertainties using Mean-Reversion Model
            # 2.1. Oil price model (OP)
            # Initializing price list
            pt_MR_list_OP=[]
            for i in range(self.trialno):
                simdata_MR=PriceModel(self.p0_OP,self.rf,self.dr,self.T,self.n,self.seed[0][i])
                pt_MR_OP=simdata_MR.MeanReversion(self.sigma_MR_OP,self.reversion_rate_OP,self.mean_level_OP)
                pt_MR_list_OP.append(pt_MR_OP)
            # Creating plot title for this model
            title_MR_OP = f"{self.trialno} Simulated Stock Price Paths Using Mean Reversion Model (mean level={self.mean_level_OP}, $\\eta$={self.reversion_rate_OP}, $\\sigma$={self.sigma_MR_OP})"

            # 2.2. Variable operating cost model (VOC)
            # Initializing price list
            pt_MR_list_VOC=[]
            for i in range(self.trialno):
                simdata_MR=PriceModel(self.p0_VOC,self.rf,self.dr,self.T,self.n,self.seed[1][i])
                pt_MR_VOC=simdata_MR.MeanReversion(self.sigma_MR_VOC,self.reversion_rate_VOC,self.mean_level_VOC)
                pt_MR_list_VOC.append(pt_MR_VOC)
            # Creating plot title for this model
            title_MR_VOC = f"{self.trialno} Simulated Stock Price Paths Using Mean Reversion Model (mean level={self.mean_level_VOC}, $\\eta$={self.reversion_rate_VOC}, $\\sigma$={self.sigma_MR_VOC})"
            y_plot_VOC=pt_MR_list_VOC
        ######################################################################### Specifying plot properties and plotting all simulations on the same graph
        # Plotting simulated values for one of uncertain variables (here "oil price" is selected to be plotted)
        if self.plot_option=='Y':
            plt.xlabel('Time Step')
            plt.ylabel('Stock Prices')
            if self.price_model=='GBM':
                plt.title(title_GBM_OP)
                y_plot_OP=pt_GBM_list_OP
                for i in range(self.trialno):
                    plt.plot(y_plot_OP[i][0],linewidth=0.3)
            elif self.price_model=='MR':
                plt.title(title_MR_OP)
                y_plot_OP=pt_MR_list_OP
                for i in range(self.trialno):
                    plt.plot(y_plot_OP[i],linewidth=0.3)
                # Plotting mean level price as an indication
                ml=[self.mean_level_OP]*(self.step+1)
                plt.plot(ml,color='red',linewidth=2)
            # Plotting percentiles
            y=y_plot_OP
            y=np.array(y)
            y=y.transpose()
            # Estimating 10, 50 and 90 percentiles and also mean value for the simulated data at each time step
            p10=np.array([0]*(self.step+1))
            p50=np.array([0]*(self.step+1))
            p90=np.array([0]*(self.step+1))
            avg=np.array([0]*(self.step+1))
            for i in range(self.step+1):
                per=np.percentile(y[i][0],[10,50,90])
                mean=np.mean(y[i][0])
                p10[i]=per[0]
                p50[i]=per[1]
                p90[i]=per[2]
                avg[i]=mean
            # Plotting percentiles on the same graph
            plt.plot(p10,color='black',linewidth=2,label='P10')
            plt.plot(p50,color='black',linewidth=2,label='P50')
            plt.plot(p90,color='black',linewidth=2,label='P90')
            plt.plot(avg,color='blue',linewidth=2,label='Mean')
            plt.legend()
            plt.show()
        else:
            pass
        ########################################################################
        # 2. Generating table of data with the simulated values of oil prices and variable operating cost (This is equivalent to "Table 1" spreadsheet calculations in BDH Article, Page 80)
        data_sim={}         # Initializing an empty dictionary for storing the simulated table of data for each simulation trial (as generated by Monte Carlo approach)
        market_val_sim=[]   # Initializing an empty list for storing estimated market values of the project for each simulation trial (as generated by Monte Carlo approach)
        NPV_sim=[]          # Initializing an empty list for storing estimated project NPV for each simulation trial (as generated by Monte Carlo approach)
        for i in range(self.trialno):
            if self.price_model=='GBM':
                price_sim=pt_GBM_list_OP[i][0,1:].tolist()
                opCost_sim=pt_GBM_list_VOC[i][0,1:].tolist()
            elif self.price_model=='MR':
                price_sim=pt_MR_list_OP[i][1:]
                opCost_sim=pt_MR_list_VOC[i][1:]
            data_new,market_val_new,NPV_new=table(self.step,self.reserve,self.prod_level,self.decline,opCost_sim,self.op_cost_rate,price_sim,self.price_growth,self.fixed_cost,self.profit,self.dr,self.init_invest)
            # Storing table outputs for this simulation trial 
            data_sim[str(i)]=data_new
            market_val_sim.append(market_val_new)
            NPV_sim.append(NPV_new)
        ######################################################################### 3. Real option valuation using LSM approach
        # Extracting keys (containing years of option) from option dictionary
        option_year=list(self.option_dict.keys())
        option_year=[int(i) for i in option_year]
        n_option=len(option_year)       # Number of time periods containing option
        # Now we should record the NPV of cash flows after year T (the realized continuation value) and the oil price and variable operating costs in the same period T in each scenario:
        # Extracting net present value, oil price and variable operating cost data at the periods containing option
        # Initializing arrays for storing project values (pv), oil price (price) and variable opearting cost (opCost) for all simulation trials
        pv_optionYear=np.zeros((n_option,self.trialno))
        price_optionYear=np.zeros((n_option,self.trialno))
        opCost_optionYear=np.zeros((n_option,self.trialno))
        for i in range(n_option):
            # Net present value is discounted by risk-free rate to Time 0
            pv_optionYear[i,:]=[data_sim[str(j)][9,option_year[i]]/((1+self.rf)**(option_year[i])) for j in range(self.trialno)]
            # Oil price data
            price_optionYear[i,:]=[data_sim[str(j)][3,option_year[i]] for j in range(self.trialno)]
            # Variable operating cost data
            opCost_optionYear[i,:]=[data_sim[str(j)][2,option_year[i]] for j in range(self.trialno)]
        ########################################################################
        # ROV analysis using LSM technique
        # Description:
        # The estimated regression equation provides an estimate of the expected continuation value as a function of the year k state variables (i.e. years including options) and can be used to determine a near-optimal exercise policy.
        # For the BDH example: In any scenario, we calculate the estimated expected continuation value Y from the values of p (oil price) and c (operating cost) generated in that trial using the fitted regression equation. Noe we have 3 alternatives that should be decided on:
        # 1. If Y is less than $100 million, then you should divest in year 5. 
        # 2. If the estimated value of the partner’s share (Y/3) exceeds the cost to buy it out ($40 million), which is translated to when Y is greater than $120 million in this case, then you should buy out the partner. 
        # 3. In the other scenarios (i.e., when the estimated Y is between $100 and $120 million), you should continue without adjusting the partnership interests.
        ######################################################################### 4.LSM valuation 
        # Extracting price (pt_pr) and operating cost (pt_op) values from simulated table of data
        pt_pr=[data_sim[str(j)][3,:] for j in range(self.trialno)]
        pt_op=[data_sim[str(j)][2,:] for j in range(self.trialno)]
        path = range(self.trialno)      # path index initialization
        n_time = self.step-1            # The first time index is zero in coding
        # Generating stock price array for each state variable (including all uncertainties)
        stock_price1 = np.array(pt_pr)
        stock_price2 = np.array(pt_op)
        # Estimating size of cash flow matrix
        r = len(stock_price1)
        c = len(stock_price1[0])
        # Extracting estimations of pv of cash flows for all paths in all periods
        cash_flow=[data_sim[str(j)][8,:] for j in range(self.trialno)]   
        cash_flow=np.array(cash_flow)

        # 4.1. Estimating conditional expectation function
        # At each time step i, we should compare the payoff of the immediate
        # excersize against the expected payoff from continuation.  There are numerous alternatives regarding which paths should be onsidered in regressing procedure for more efficiency. 
        # Initializing an array for storing the project values (corresponding to the selected optimal policy) at each path along each period using the dynammic programming approach. This is equivalent to "cash flow matrix" in Longstaff & Schwartz notation 
        cash_flow_matrix = np.array([[0] * c] * r,'float64')
        # Initializing an empty list for storing the optimal exercise (i.e. the optimal decision alternative) at each period of time containing options.
        exercise=[]
        # Initializing empty lists for storing the following variables (The values of the following varaibles are set initially to [] so that there is no need to update them for those periods without any real option; however they have to be updated for periods including options)
        contVal, itmRatio, regMeasure, bestOrder, coeff, bestModels, bestScore, xData, yData = [[] for i in range(c)], [[] for i in range(c)], [[] for i in range(c)], [[] for i in range(c)], [[] for i in range(c)], [[] for i in range(c)], [[] for i in range(c)], [[] for i in range(c)], [[] for i in range(c)], 
        # contVal       Contination value for all periods
        # itmRatio      Ratio of ITM paths to all paths for all periods
        # regMeasure    Regression performance measure for all periods
        # bestOrder     Best polynomial order for tuning regression function for all periods
        # coeff         Coefficients of best regression equation for all periods
        # bestModels    Best regression models obtained for each period
        # bestScore     Correlation coefficients of best regression equation for each period
        # xData         X data used in making regression equation for each period
        # yData         X data used in making regression equation for each period        
        # Solving the problem using a rollback procedure, starting from the last period and then moving back to the first period
        for i in range(n_time,-1,-1):
            # Checking if the current period contains any options for exercising or not (if it contains, then we need to do a regression. otherwise, skip regression and go the next step)
            if i not in option_year:
                if (i == n_time):
                    # For the last time period, the project value is equal to the cash flow at that period since there is no time period after that
                    cash_flow_matrix[:,i] = cash_flow[:,i]
                else:
                    # For other time periods, the project value is equal to the discounted expected project value at the next period plus the amount of cash flow received at the end of that period
                    cash_flow_matrix[:,i] = cash_flow[:,i]+cash_flow_matrix[:,i+1]*self.DR
                # Initializing an array for storing the exercised option in each path for the current period  that contain real options (there are no options in the current period, so we replce exercise values by zero in the relevant array)
                ex_opt=[0]*self.trialno
            else:
                # Extracting pairs of key (option_name) and value (option properties) from option dictionary
                option_name=i
                option_property=self.option_dict[str(i)]
                nOption=len(option_property)        # Number of options in the current period
                count=0
                # Defining an interval for each option, so that if the project value at a specific period lies within that interval, then the corresponding option would be exercised at that period (this approach is equivalent to the solution presented by Smith (2005)for solving BDH example using LSM approach)
                for k,v in option_property.items():
                    count += 1
                    k=k.lower()
                    # Each option has two elements or values: (1) cost of exercising the option, (2) profit from exercising the option
                    if k in ['divest']:
                        # For divest option, the profit from exercising is simply a constant value
                        margin_low=v[0]
                        margin_high=v[1]
                        interval_divest=pd.Interval(left=margin_low, right=margin_high)
                    elif k in ['buyout','buy out','buy_out','buy-out']:
                        # For buyout option, the profit is not a constant, it is rather a fraction by which the amount of cash flows at the next time step grows up due to buying out the partnership
                        share_gain=v[1]/75
                        cost_buyout=v[0]
                        margin_low=cost_buyout/share_gain
                        # There is no limitation on the right margin for buyout (the higher the profit, the better to exercise buyout option)
                        margin_high=float('inf')
                        interval_buyout=pd.Interval(left=margin_low, right=margin_high)
                for k,v in option_property.items():
                    if k in ['continue']:
                        margin_low=interval_divest.right
                        margin_high=interval_buyout.left
                        interval_continue=pd.Interval(left=margin_low, right=margin_high)
                # Initializing an empty list for storing path numbers that will be used in tuning the regression function
                idx = []    
                # There are 5 different alternatives for selecting the types of paths    
                if (self.which_path=="alternative1"):
                    # Alternative1 means that both options should be in the money at the current period
                    for j in path:
                        option_divest=cash_flow[j,i]+interval_divest.right
                        option_buyout=cash_flow[j,i]+cash_flow_matrix[j,i+1]*(1+share_gain)*self.DR -cost_buyout
                        if (option_divest>0 and option_buyout>0):
                            idx.append(j)
                elif (self.which_path=="alternative2"):
                    # Alternative2 requires that the options to be in the money either at the current period or later on until the expiration date
                    for j in path:
                        option_divest=[]
                        option_buyout=[]
                        for kk in range(i,n_time+1):
                            option_divest.append(cash_flow[j,kk]+interval_divest.right)
                            if kk!=n_time:
                                option_buyout.append(cash_flow[j,kk]+cash_flow_matrix[j,kk+1]*(1+share_gain)*self.DR -cost_buyout)
                            else:
                                option_buyout.append(cash_flow[j,kk]-cost_buyout)
                        if (max(option_divest)>0 or max(option_buyout)>0):
                            idx.append(j)
                elif (self.which_path=="alternative3"):
                    # Alternative3 means that either of the options could be in the money at the current period
                    for j in path:
                        option_divest=cash_flow[j,i]+interval_divest.right
                        option_buyout=cash_flow[j,i]+cash_flow_matrix[j,i+1]*(1+share_gain)*self.DR -cost_buyout
                        if (option_divest>0 or option_buyout>0):
                            idx.append(j)
                elif (self.which_path=="alternative4"):
                    # Alternative4 means that only paths that have positive cash flows at the current time period are in-the-money and should be selected
                    for j in path:
                        if cash_flow[j,i] > 0 :
                            idx.append(j)
                elif (self.which_path=="alternative5"):
                    # Alternative5 means that all the paths are considered regardless of being in or out of the money
                    idx = np.arange(self.trialno).tolist()
                # Estimating the ratio of ITM paths to all paths (used just for reporting)
                itm_ratio=len(idx)/self.trialno
                print(f'Step T{i+1}')
                print('The ratio of paths considered for regression: ', itm_ratio)
                # Stock price at time step i-1 for the specified paths (stock price is composed of any number of uncertainties in the model, here "2" variables including oil price and operating cost)
                x=np.zeros((len(idx),2))
                for j in range(len(idx)):
                    x[j,:]=np.array([stock_price1[idx[j],i],stock_price2[idx[j],i]])  
                if len(x) == 0:
                    print(f"There is no in-the-money paths for time step {i}. Skipping to the next time step...")
                    # Since the cash_flow_matrix has been initialized with zero inputs, no need to re-assign zero values to the step i.
                else:
                    # If the current period does include any real options, just do the regression 
                    # Definition of y (independent variable)
                    if (i == n_time):
                        # cash flow at the last time step does not require any discounting
                        ym = np.array([cash_flow[idx,i]])    
                    else:
                        # discounted cash flow at time step i
                        ym=np.array([cash_flow_matrix[idx,i+1] * self.DR])
                    # Regressing y on x to get conditional expectation function for discounted cash flow (dcf)
                    try:
                        # Constructing data sets for regression
                        xm=x.transpose()        # X data for regression
                        cols = ["price","opCost","cash_flow"]
                        # Trying different linear regression models
                        test_split=0     # Fraction of testing data when splitting the dataset into test and train data
                        from regression import LinearRegression
                        # Making an instance of class LinearRegression (for running a (multivariate) regression relating the realized continuation values (ym) to the uncertain oil prices (xm1) and variable operating costs (xm2) at every period with options)
                        regModel=LinearRegression(xm,ym,cols,test_split)
                        if self.modelName.lower()=='kfold':
                            # Making linear regrerssion model with k-fold cross validation
                            model_kf,score_kf,mape_kf=regModel.kfoldCV(k=5,plot="False")
                            # Renaming outputs for more consistency
                            best_model,best_score,best_mape=model_kf,score_kf,mape_kf
                            best_order=1
                            # Getting the fitted regressor coefficients
                            coef=best_model.coef_
                        elif self.modelName.lower()=='gradientboosting':
                            # Making linear regression method with gradient boosting method
                            model_gb,score_gb,mape_gb=regModel.GradientBoosting(plot="False")
                            # Renaming outputs for more consistency
                            best_model,best_score,best_mape=model_gb,score_gb,mape_gb
                            best_order=1
                            # Getting the fitted regressor coefficients
                            coef=[] # No coefficients are available for this model
                        elif self.modelName.lower()=='polynomial':
                            # Making polynomial regression models 
                            # Selecting the best polynomial order for polynomial regression based on minimum mean absolute percentage error (mape)
                            order_pl=np.arange(1,10)
                            mape=[]
                            for order_i in range(len(order_pl)):
                                model_pl,score_pl,mape_pl=regModel.Polynomial(order=order_pl[order_i],plot="False")
                                mape.append(mape_pl)
                            idx_pl=np.where(mape==min(mape))
                            best_order=order_pl[idx_pl[0][0]]
                            # Re-building polynomial regression model using best-order
                            best_model,best_score,best_mape=regModel.Polynomial(order=best_order,plot="False")
                            # Getting the fitted regressor coefficients
                            coef=best_model['linear'].coef_
                        elif self.modelName.lower()=='featureselection':
                            # Finding the best regression model using AIC or mse criterion for feature selection
                            maxOrder=9
                            minOrder=1
                            bestModel, coef, xBest, mseNewList, scoreNewList, powerNewList=regModel.modelSelection(maxOrder,minOrder)
                            # Assessing the maximum order utilized in making the best fitted model
                            order=[]
                            for ii in range(len(powerNewList)):
                                order.append(sum(powerNewList[ii]))
                            best_order=max(order)
                            # Renaming outputs for more consistency
                            best_model,best_score,best_mape = bestModel, scoreNewList, mseNewList
                            print(f"step {i}")
                            # Making new features from xm
                            xArr=xm.transpose()
                            for ii in range(len(powerNewList)):
                                p1, p2 =powerNewList[ii][0], powerNewList[ii][1]
                                feature=xm.transpose()[:,0]**p1*xm.transpose()[:,1]**p2
                                # Adding this feature to the previous X dataset
                                xArr=np.c_[xArr,feature]
                            # Updating xm array with xArr
                            xm=xArr.transpose()
                        # Estimating "conditional expectation function" (expected payoff from continuation using the fitted regressor)
                        # Specifying whether all the data (including ITM and OOTM paths) are predicted using the regressed function, regradless if they had been used in tuning the regression equation or not (This is called Approach 2 in the powerpoint presentation)
                        flag_AllDataPredicted=False
                        if flag_AllDataPredicted:
                            idx=path
                            x=np.concatenate((stock_price1[:,i].reshape(-1,1),stock_price2[:,i].reshape(-1,1)),axis=1)  
                            continuation=best_model.predict(x)
                        else:
                            continuation=best_model.predict(xm.transpose())
                        # Initializing an array for storing the exercised option in each path for the current period that contains real options
                        ex_opt=[0]*self.trialno
                        # Comparing the payoff of immediate exercise with the expected payoff from continuation to choose the best decision in decision (option) node
                        countJ=-1
                        # Initializing empty lists for storing project values due to exercising buyout, divest and continuation options at the paths that are "excluded"
                        buyout_val, divest_val, continuation_val = [], [], []      
                        excluded=list(set(path)-set(idx))
                        # Defining a flag for "multi-option" problems (if more than one period contains real options)
                        flag_multiOption=n_option>1
                        for j in range(self.trialno):
                            # Checking if the current path has been in the included paths or not
                            if j in idx:
                                countJ += 1
                                # Estimating the value of divesting option (without considering the current period cash flow to make its definition consistent with the definition of continuation value)
                                option_divest=interval_divest.right
                                # Estimating the value of buyout option (without considering the current period cash flow to make its definition consistent with the definition of continuation value)
                                option_buyout=continuation[countJ]*(1+share_gain) -cost_buyout      # It does not need discounting anymore, since the continuation term (as embedded in option_buyout) has been discounted back once in its first-place definition prior to doing regression
                                # Selecting the decision with the maximum payoff
                                option_labels=list(option_property.keys())
                                option_val=[continuation[countJ],option_buyout,option_divest]
                                # Maximizing project values
                                best_option_val=max(option_val)
                                best_option_idx=option_val.index(best_option_val)
                                best_option=option_labels[best_option_idx]
                                # If it is best to continue the project without exercising any options at the current period, then we need to estimate the realized continuation value (rather than the regressed one)
                                if best_option.lower()=='continue':
                                    best_option_val=cash_flow[j,i]+cash_flow_matrix[j,i+1]*self.DR
                                elif best_option.lower() in ['buyout','buy out','buy_out','buy-out']:
                                    # If one of the options is decided to be exercised at the current period, then add the amount of cash flow recieved at the end of this period to the value option (since it was subtracted from the option value before comparing with the continuation value)
                                    # Taking care in the case of "multi-option" problem and checking if this option (buyout) has been exercised once more previously or not (Each option can be exercised only once). However it is possible to have a divest option exercised after a buyout option has exercised (that does not hold true vice versa, since there is no possiblity to have the project continued after a divest option has exercised)
                                    if flag_multiOption:
                                        countK=-1
                                        for k in range(n_time-1,i,-1):
                                            if exercise[countK][j]!='Divest':
                                                cash_flow_matrix[j,k]=cash_flow[j,k]+cash_flow_matrix[j,k+1]*self.DR
                                            countK -= 1
                                            if exercise[countK][j]=='Buyout':
                                                exercise[countK][j]='Continue'
                                    best_option_val=cash_flow[j,i]+cash_flow_matrix[j,i+1]*(1+share_gain)*self.DR-cost_buyout
                                elif best_option.lower()=='divest':
                                    best_option_val=cash_flow[j,i]+best_option_val
                                # Updating cash flow matrix
                                ex_opt[j]=best_option.title()
                                cash_flow_matrix[j,i]=best_option_val
                            else:
                                # Continuing project for excluded paths
                                cash_flow_matrix[j,i]=cash_flow[j,i]+cash_flow_matrix[j,i+1]*self.DR
                                # Doing additional investigation into the option values for those paths that are not considered during the regression (to see the effect of different path alternatives on the project value)
                                # Calculating buyout value (bv)
                                bv=cash_flow[j,i]+cash_flow_matrix[j,i+1]*(1+share_gain)*self.DR -cost_buyout
                                # Calculating divest value (dv)
                                dv=cash_flow[j,i]+interval_divest.right
                                # Calculating continuation value (cv)
                                cv=cash_flow_matrix[j,i]
                                # Storing bv, dv and cv
                                buyout_val.append(bv)
                                divest_val.append(dv)
                                continuation_val.append(cv)
                    except:
                        print(f"All discounted cash flows for time step {i} are zero. No regression is done at this step. Skipping to the next step...")
                        # Since the cash_flow_matrix has been initialized with zero inputs, no need to re-assign zero values to the step i.
                # Updating the following lists by storing the obtained results for the current period
                contVal[i]=continuation
                itmRatio[i]=itm_ratio
                regMeasure[i]=best_mape
                bestOrder[i]=best_order
                coeff[i]=coef
                bestModels[i]=best_model
                bestScore[i]=best_score
                xData[i]=xm
                yData[i]=ym
            # Appending ex_opt for the current period to the exercise list
            exercise.insert(0,ex_opt)
        # Finalizing cash flow matrix at the end of valuation for all time
        # horizons
        # Options could be exercised only once. Only earliest exrecise date should be considered and a value of zero is assigned to cash flows that occur after this time:
        # Generating a stopping_rule matrix based on the final cash flow matrix
        stopping_rule = np.array([[0] * c] * r)
        exercise=np.array(exercise)
        exercise=np.array(exercise.transpose())
        if option_year:
            for i in range(self.trialno):
                try:
                    # For buyout option, the next periods were checked against exercising the option while calculating the option value previously. However it is needed to check for divest option to see if there is any other period ahead that tends to exercise divest or not.
                    early_ex=min(np.where(exercise[i]=='Divest')[0])
                    cash_flow_matrix[i,(early_ex+1):]=0
                    # Updating stopping_rule matrix
                    stopping_rule[i,early_ex]=1
                except:
                    continue
        ########################################################################
        # Printing final results as simple tables with headings
        df_cashMatrix = pd.DataFrame(cash_flow_matrix)
        df_cashMatrix.columns = [f"T{str(i)}" for i in range(1,self.step+1)]
        df_cashFlow = pd.DataFrame(cash_flow)
        df_cashFlow.columns = [f"T{str(i)}" for i in range(1,self.step+1)]
        # Averaging NPV at Time 0 over all simulation paths
        npv_mean=np.mean(cash_flow_matrix[:,0])
        NPV=npv_mean*self.DR
        # Printing outputs
        print(f"The original cash flow matrix (without options) for this problem is as follows: (Only First 10 rows are displayed) \n {df_cashFlow[0:11]}\n")
        print(f"The optimal cash flow matrix (after exercising options) for this problem is as follows: (Only First 10 rows are displayed) \n {df_cashMatrix[0:11]}\n")
        print(f'NPV = ${NPV} Million')
        ######################################################################### Returning outputs
        if not option_year:
            reg_model=[]
            continuation=[]
        else:
            if n_option==1:
                # For problems with a single period containing options
                reg_model={'ITM_ratio':itm_ratio,'reg_measure':best_mape,'best_order':best_order,'coef':coef,'model':best_model,'score':best_score,'exercise@all_periods':exercise,'x':xm,'y':ym}
            else:
                # For problems with multiple periods containing options
                reg_model={'ITM_ratio':itmRatio,'reg_measure':regMeasure,'best_order':bestOrder,'coef':coeff,'model':bestModels,'score':bestScore,'exercise@all_periods':exercise,'x':xData,'y':yData}
                continuation=contVal
        return NPV,cash_flow,cash_flow_matrix,stopping_rule,continuation,reg_model