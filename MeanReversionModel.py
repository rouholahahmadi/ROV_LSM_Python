class PriceModel:
    """ This function estimates a path of future stock (oil) price using a number of stochastic models including GBM, Mean Reversion, and Two-factor price models. """

    def __init__(self,p0,rf,dr,T,n,seed="False"):
        """ This methods includes all the required parameters for simulating oil price using all available methods. """
        # Input parameters
        # p0                Initial price, $
        # rf                Risk-free interest rate (risk-neutral drift)
        # dr                Risk-adjusted discount rate (true drift)
        # n                 granularity of time-period (also called dt)
        # T                 total time in years (number of years)
        # seed              Random seed for random number generation; default value is False, indicating that it is not kept constant by the user
        self.p0=p0
        self.rf=rf
        self.dr=dr
        self.T=T
        self.n=n
        self.seed=seed
        # Importing required packages
        import numpy as np
        import pandas as pd
        import scipy.stats
        import scipy
        import sys
        import os
        import random
        import matplotlib.pyplot as plt 
        import importlib
        from tabulate import tabulate
        # Appending the current working directory to the system path of python
        curr_path=os.getcwd()
        sys.path.append(curr_path)
        from GBMsimLSMvaluation import GBM_Model
        from table_generation import table

        # Estimating risk premium
        self.risk_perm=self.dr-self.rf
        # Estimating number of total periods
        self.period=int(T/n)        
        # Creating list of all periods
        self.t = list(np.linspace(0, T, self.period+1))

    def GBM(self,mu,volat):
        """ This function simulates a single path of stock price using GBM approximation. """
        # This function is written for generating any number of trails (multiple stock price paths). To dedicate it for single path synthesis, the "trailno" parameter is assumed to be equal to 1 in the following calculations.
        # Input parameters
        # mu                Drift rate
        # volat             Annual volatility 
        self.mu_GBM=mu
        self.volat_GBM=volat               
        self.trialno=1
        # Description of equations
        # Weiner process
        #   w[i] = w[i-1]+(yi/np.sqrt(n_step)); yi: random Normal number
        # GBM process
        #   S(t) = S(0).exp{(mu-(sigma^2/2).t)+sigma.W(t)}
        # Importing required packages
        import random
        import numpy as np
        import matplotlib.pyplot as plt 
        import pandas as pd

        # Simulating stock price sequence by generating random normal numbers
        self.ds = np.linspace(0,self.T,num=self.period+1)	# time sequence including the starting time 0
        self.ds=np.delete(self.ds,0)    # Removing first element from ds array
        # Random number generation from a standard normal distribution
        if self.seed=="False":
            self.b = {str(i): [0] + np.random.standard_normal(size = int(self.period)) for i in range(1,self.trialno + 1)}	# random shocks for each time horizon
        else:
            phi=np.zeros((self.trialno,int(self.period)))
            if self.seed=="True":
                for i in range(self.trialno):
                    np.random.seed(i)
                    phi[i,:]=np.random.standard_normal(size = int(self.period)).tolist()
            elif isinstance(self.seed,(float,int)):
                for i in range(self.trialno):
                    np.random.seed(int(self.seed))
                    phi[i,:]=np.random.standard_normal(size = int(self.period)).tolist()
            else:
                print("No valid seed entered! Enter True, False or a number as the seed")
            self.b = {str(i): [0] + phi[i-1,:] for i in range(1,self.trialno + 1)}	# random shocks for each time horizon

        self.W = {str(i): self.b[str(i)].cumsum() * np.sqrt(self.n) for i in range(1,self.trialno + 1)}	# cumulative random shocks from start to each time horizon (Wiener process)
        self.drift = (self.mu_GBM - 0.5 * self.volat_GBM ** 2) * self.ds
        self.diffusion = {str(i): self.W[str(i)] * self.volat_GBM for i in range(1,self.trialno + 1)}
        self.factor = {str(i): np.exp(self.drift + self.diffusion[str(i)]) for i in range(1,self.trialno + 1)}
        self.price = {str(i): self.p0 * self.factor[str(i)] for i in range(1,self.trialno + 1)}
        self.p0_arr = np.array([[self.p0] * 1] * self.trialno)
        self.pt = np.hstack((self.p0_arr,np.array([self.price[str(i)] for i in range(1,self.trialno + 1)])))
        return self.pt

    def MeanReversion(self,volat,rev_rate,mean_level):
        """ This function simulates future oil price using a mean reversion model. """
        # The reference for this part:
        # http://marcoagd.usuarios.rdc.puc-rio.br/sim_stoc_proc.html#mc-mrd
        # Input parameters
        # volat             Annual volatility 
        # rev_rate          Reversion speed per annum (also called as eta)
        # mean_level        Long term equilibrium price, $
        self.volat_MR=volat
        self.eta=rev_rate
        self.mean_level=mean_level
        # Importing required packages
        import numpy as np

        # 1. Random number generation for all paths from a standard normal distribution
        if self.seed=="False":
            np.random.seed()
            phi=np.random.standard_normal(size = int(self.period)).tolist()
        else:
            if self.seed=="True":
                np.random.seed(1)
            elif isinstance(self.seed,(float,int)):
                np.random.seed(int(self.seed))
            else:
                print("No valid seed entered! Enter True, False or a number as the seed")
            # Generating random shocks for each time horizon
            phi=np.random.standard_normal(size = int(self.period)).tolist()
        phi.insert(0,0)
        # 2. Simulating oil prices using risk-neutral mean reversion
        xt=[0]*(self.period+1)
        xt[0]=np.log(self.p0)
        pt_MR=[0]*(self.period+1)
        pt_MR[0]=self.p0
        for i in range(self.period):
            x1=xt[i]*np.exp(-self.eta*self.n)
            x2=(np.log(self.mean_level)-(self.risk_perm/self.eta))*(1-np.exp(-self.eta*self.n))
            x3=self.volat_MR*np.sqrt((1-np.exp(-2*self.eta*self.n))/(2*self.eta))*phi[i+1]
            xt[i+1]=x1+x2+x3
            pt_MR[i+1]=np.exp(xt[i+1]-(0.5*((1-np.exp(-2*self.eta*self.t[i+1]))*((self.volat_MR**2)/(2*self.eta)))))
        return pt_MR

    def MeanReversion_Jump(self,volat,rev_rate,mean_level,sigma_jump,lambda_poisson):
        """ This function simulates the stcok prices using the Mean Reversion Model with Jumps. """
        # The references for this part are as follows:
        # 1. http://marcoagd.usuarios.rdc.puc-rio.br/sim_stoc_proc.html#mc-mrd
        # 2. Cartea, A., Figueroa, M., 2007. Pricing in Electricity Markets: A Mean Reverting Jump Diffusion Model with Seasonality, pp. 313-335.doi: https://doi.org/10.1080/13504860500117503 (refer to page 9 of the article)
        # Input parameters
        # volat             Annual volatility 
        # rev_rate          Reversion speed per annum (also called as eta)
        # mean_level        Long term equilibrium price, $
        # sigma_jump        Standard deviation of Jump size Poisson distribution
        # lambda_poisson    Mean number of arrivals (i.e. jumps) per unit time in Poisson process
        self.volat_MRJ=volat
        self.eta=rev_rate
        self.mean_level=mean_level
        self.lambdaP=lambda_poisson
        self.sigmaJ=sigma_jump
        # Importing required packages
        import numpy as np

        # 1. Random number generation for all paths from a standard normal distribution
        if self.seed=="False":
            np.random.seed()
            phi=np.random.standard_normal(size = int(self.period)).tolist()
        else:
            if self.seed=="True":
                np.random.seed(1)
            elif isinstance(self.seed,(float,int)):
                np.random.seed(int(self.seed))
            else:
                print("No valid seed entered! Enter True, False or a number as the seed")
            # Generating random shocks for each time horizon
            phi=np.random.standard_normal(size = int(self.period)).tolist()
        phi.insert(0,0)
        # 2. Drawing random samples from a poisson distribution with parameter lambdaP to see if any jumps occur in each time period or not
        rndPoisson = np.random.poisson(self.lambdaP, self.period)
        rndPoisson = rndPoisson.tolist()
        # 3. Simulating random variables of percentage change (jump size) in the stock price if the Poisson event occurs
        jump_up=np.log(2)
        jump_down=-jump_up
        mean_jump=-self.sigmaJ*2/2
        var_jump=self.sigmaJ*2
        # Drwaing random samples from normal distribution with mean "mean_jump" and variance "var_jump" (The jump size can be constant as well)
        rndJump = np.random.normal(mean_jump, np.sqrt(var_jump), self.period)
        #rndJump = np.exp(rndJump)         # Random jumps
        rndJump = rndJump.tolist()
        j=[0]*(self.period+1)
        for i in range(self.period):
            if (rndPoisson[i]>0):
                j[i+1]=rndJump[i]
        # 4. Estimating expected value of (phi^2) for using in the variance term of xt (E(X^2)=Var(X)+E(X)^2=Var(X)+mean^2)
        exp_val=var_jump+mean_jump**2
        # 5. Simulating oil prices using risk-neutral mean reversion with jump
        xt=[0]*(self.period+1)
        xt[0]=np.log(self.p0)
        pt_MR_jump=[0]*(self.period+1)
        pt_MR_jump[0]=self.p0
        for i in range(self.period):
            x1=xt[i]*np.exp(-self.eta*self.n)
            x2=(np.log(self.mean_level)-(self.risk_perm/self.eta))*(1-np.exp(-self.eta*self.n))
            x3=self.volat_MRJ*np.sqrt((1-np.exp(-2*self.eta*self.n))/(2*self.eta))*phi[i+1]
            x4=j[i+1]       # Jump component
            xt[i+1]=x1+x2+x3+x4
            # Estimating variance of xt
            var_x=(1-np.exp(-2*self.eta*self.t[i+1]))*((self.volat_MRJ**2+self.lambdaP*exp_val)/(2*self.eta))
            # Estimating next period price
            pt_MR_jump[i+1]=np.exp(xt[i+1]-(0.5*var_x))
        return pt_MR_jump

    def TwoFactorModel(self,mu,volat_GBM,volat_MR,rev_rate,mean_level):
        """ This function estimates the future prices using a two factor model including a short-term factor, as modeled by mean-reversion, and a long-term factor, as modeled by GBM. """
        # Please refer to the following reference for more details of the process
        # B Jafarizadeh, R Bratvold. 2012. Two-factor oil-price model and real option valuation: an example of oilfield abandonment. SPE Economics & Management 4 (03), 158-170.DOI: https://doi.org/10.2118/162862-PA
        # ln(St) = Sai_t + Kai_t (Sai: Long-term factor, Kai: Short-term factor)
        # Input parameters
        # mu                    Drift rate
        # volat_GBM             Annual volatility 
        # volat_MR              Annual volatility 
        # rev_rate              Reversion speed per annum (also called as eta)
        # mean_level            Long term equilibrium price, $
        self.mu_GBM=mu
        self.volat_GBM=volat_GBM
        self.volat_MR=volat_MR
        self.eta=rev_rate
        self.mean_level=mean_level
        # Importing required packages
        import numpy as np
        # 1. Simulating long-term factor using GBM approximation
        model=PriceModel(self.p0,self.rf,self.dr,self.T,self.n,self.seed)
        long_term=model.GBM(self.mu_GBM,self.volat_GBM)
        # 2. Simulating short-term factor using Mean Reversion model
        short_term=model.MeanReversion(self.volat_MR,self.eta,self.mean_level)
        # 3. Excluding p0 from short_term price as it is accounted by long_term price stochastically
        short_term=[short_term[i]-self.p0 for i in range(self.period+1)]
        # 4. Adding the long_term and short_term copmponents 
        pt_TF = long_term + short_term
        return pt_TF
