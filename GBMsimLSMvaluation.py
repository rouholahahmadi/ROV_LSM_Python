class GBM_Model:
    """This Program simulates market prices according to the Geometric Brownian Motion with the followoing properties"""
    def __init__(self,p0,mu,sigma,trialno,n,T,seed="False"):
        # Input Parameters:
        # p0        start price at time 0
        # mu        annulazied drift parameter of GBM process
        # sigma     annulazied volatility parameter of GBM process
        # trialno   number of simulation trials or paths
        # n         granularity of time-period (also called dt)
        # T         total time in years (also called deltaT)
        # seed      Random seed for random number generation
        self.p0 = p0
        self.mu = mu
        self.sigma = sigma
        self.trialno = trialno
        self.n = n
        self.T = T
        self.step = int(self.T /self.n)   # number of time steps
        self.seed=seed

    def GBM_Function(self,plot="True"):
        # This function generates any desired number of stock price paths using
        # GBM approach by simulation
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
        self.ds = np.linspace(0,self.T,num=self.step)	# time sequence including the starting time 0
        # random number generation from a standard normal distribution
        if self.seed=="False":
            self.b = {str(i): [0] + np.random.standard_normal(size = int(self.step)) for i in range(1,self.trialno + 1)}	# random shocks for each time horizon
        else:
            phi=np.zeros((self.trialno,int(self.step)))
            if self.seed=="True":
                for i in range(self.trialno):
                    np.random.seed(i)
                    phi[i,:]=np.random.standard_normal(size = int(self.step)).tolist()
            elif isinstance(self.seed,(float,int)):
                for i in range(self.trialno):
                    np.random.seed(i+int(self.seed))
                    phi[i,:]=np.random.standard_normal(size = int(self.step)).tolist()
            else:
                print("No valid seed entered! Enter True, False or a number as the seed")
            self.b = {str(i): [0] + phi[i-1,:] for i in range(1,self.trialno + 1)}	# random shocks for each time horizon

        self.W = {str(i): self.b[str(i)].cumsum() * np.sqrt(self.n) for i in range(1,self.trialno + 1)}	# cumulative random shocks from start to each time horizon (Wiener process)
        self.drift = (self.mu - 0.5 * self.sigma ** 2) * self.ds
        self.diffusion = {str(i): self.W[str(i)] * self.sigma for i in range(1,self.trialno + 1)}
        self.factor = {str(i): np.exp(self.drift + self.diffusion[str(i)]) for i in range(1,self.trialno + 1)}
        self.price = {str(i): self.p0 * self.factor[str(i)] for i in range(1,self.trialno + 1)}
        self.p0_arr = np.array([[self.p0] * 1] * self.trialno)
        self.pt = np.hstack((self.p0_arr,np.array([self.price[str(i)] for i in range(1,self.trialno + 1)])))

        # Ploting stock prices
        if (plot=="True"):
            plt.figure()
            plt.xlabel('Time Step')
            plt.ylabel('Stock Prices')
            title = f"{self.trialno} Simulated Stock Price Paths Using GBM(return={self.mu}, volatility={self.sigma})"
            plt.title(title)
            for i in range(self.trialno):
                plot_all = plt.plot(np.arange(0,self.step + 1),self.pt[i,:])
                plot_color1 = random.random()
                plot_color2 = random.random()
                plot_color3 = random.random()
                plt.setp(plot_all,linestyle='-',linewidth=0.5,color=(plot_color1,plot_color2,plot_color3))
            plt.show(block=False)
            #plt.pause(1)
            #plt.close()
        else:
            pass
        return self.pt

class PriceSimulation(GBM_Model):
    """This class tries to simulate stock prices using GBM model and then valuate American Put Options using LSM (Least Square Monte Carlo) approach."""
    
    def __init__(self,p0,mu,sigma,trialno,n,T,seed,strike,DR):
        # Input Parameters:
        # p0        start price at time 0
        # mu        annulazied drift parameter of GBM process
        # sigma     annulazied volatility parameter of GBM process
        # trialno   number of simulation trials or paths
        # n         granularity of time-period (also called dt)
        # T         total time in years (also called deltaT)
        # strike    strike price
        # DR        discount rate coefficient (=1/(1+r))
        super().__init__(p0,mu,sigma,trialno,n,T)
        self.strike = strike
        self.DR = DR
        
    def LSM_Valuation(self,power,which_path="in-the-money"):
        """This function implements American put option valuation using LSM approach"""
        # Input parameters
        # power                 Polynomial order for regression of discounted cash flows
        # which_path            Keyword determining which paths of stock price simulation shall be considered in regression process. It is set to be "in-the-money" by default that dictates only the paths that are in the money have to included in regression. If it gets a value like "all" or any other keywords, then all paths would be considered.

        # Importing required packages
        import numpy as np
        import matplotlib.pyplot as plt

        self.power=power
        self.which_path=which_path
        # Constructing GBM object using some input parameters
        gbm = GBM_Model(self.p0,self.mu,self.sigma,self.trialno,self.n,self.T,self.seed)
        self.pt = gbm.GBM_Function()
        self.step = int(self.T / self.n)        # time step size
        path = range(self.trialno)              # path index initialization
        n_time = self.step - 1
        print('ntime=',n_time)
        # Generating stock price table
        stock_price = self.pt

        # Estimating cash flow matrix
        r = len(stock_price)
        c = len(stock_price[0])

        cash_flow = np.array([[self.strike] * c] * r) - stock_price
        cash_flow[cash_flow < 0] = 0

        # Estimating conditional expectation function
        # At each time step i, we should compare the payoff of the intermediate
        # excersize against the expected payoff from continuation.  Only those
        # paths that are in the money (with positive cash flow) are onsidered
        # for more efficiency.
        cash_flow_matrix = np.array([[0] * c] * r)
        cash_flow_matrix = cash_flow_matrix.astype('float64')
        for i in range(n_time,1,-1):
            idx = []
            if (self.which_path=="in-the-money"):
                # Considering only the paths that are in the money
                for j in path:
                    # If the cash flow in the previous time step i-1 exceeds zero, then that period is in the money
                    if cash_flow[j,i - 1] > 0 :
                        idx.append(j)
            else:
                # Otherwise consider all the paths regardless of being in or out of the money
                idx = np.arange(self.trialno).tolist()
            idx.reverse()
            x = stock_price[idx,i - 1]  # Stock price at time step i-1 for the specified paths 
            if len(x) == 0:
                print(f"There is no path in the money for time step {i-1}. Skipping to the next step...")
                # Since the cash_flow_matrix has been initialized with zero
                # inputs, no need to re-assign zero values to the step i-1.
            else:
                if (i == n_time):
                    y = cash_flow[idx,i] * self.DR   # discounted cash flow at time step i
                else:
                    y = cash_flow_matrix[idx,i] * self.DR   # discounted cash flow at time step i
                # Regressing y on x to get conditional expectation function for discounted cash flow (dcf)
                try:
                    dcf = np.poly1d(np.polyfit(x, y, self.power))
                    print(f"step {i}")
                    # Estimating discounted cash flow (dcf) for the time step i
                    # as the continuation payoff
                    continuation = dcf(x)
                    # Extracting the immediate exercise for the time step i-1
                    immediate = [self.strike] * len(x) - x
                    print('continuation=',continuation,'immediate=',immediate)
                    # Comparing immediate exercise vs.  continuation exercise
                    # and finding the larger payoffs
                    for k in range(len(immediate) - 1,-1,-1):
                        index = idx[k]
                        if (immediate[k] >= continuation[k]):
                            cash_flow_matrix[index,i - 1] = cash_flow[index,i - 1]
                            cash_flow_matrix[index,i:] = 0
                        else:
                            if (i == n_time):
                                cash_flow_matrix[index,i] = cash_flow[index,i]
                            else:
                                continue
                except:
                    print(f"All discounted cash flows for time step {i} are zero. No regression is done at this step. Skipping to the next step...")
                    # Since the cash_flow_matrix has been initialized with zero
                    # inputs, no need to re-assign zero values to the step i.

        # Finalizing cash flow matrix at the end of valuating for all time
        # horizons
        cash_flow_matrix = np.delete(cash_flow_matrix,0,1)

        # Generating a stopping_rule matrix based on the final cash flow matrix
        stopping_rule = np.array([[0] * (c - 1)] * r)
        stopping_rule[cash_flow_matrix > 0] = 1
        return cash_flow_matrix,stopping_rule
