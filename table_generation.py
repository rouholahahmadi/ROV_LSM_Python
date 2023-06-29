def table(period,reserve,prod_level,decline,op_cost,op_cost_rate,price,price_growth,fixed_cost,profit,dr,init_invest):
    """ This function generates a table of data as required for solving the Example problem in BDH paper (Section 5)"""
    # Input parameters
    # period            Number of periods of production, years
    # reserve           Initial reserve, MMBD
    # decline           Yearly decline rate, fraction
    # prod_level        Initial yearly production level, MMBD
    # op_cost           Variable operational cost per barrel of oil at Year 0, $
    # op_cost_rate      Yearly growth rate for variable operational cost, fraction 
    # price             Oil price per barrel, $
    # price_growth      Yearly growth rate of oil price, fraction
    # fixed_cost        Yearly fixed cost, MM$
    # profit            Profit sharing rate, fraction
    # dr                Yearly discount rate, fraction
    # init_invest       Up=front investment, MM$

    # Importing required packages and modules
    import numpy as np
    import pandas as pd
    # Initializing table of data 
    data=np.zeros((11,period),dtype='float')
    # Assigning headers to rows and columns of table of data
    row_header=['Remaining reserves','Production level','Variable op cost rate','Oil price','Revenues','Production cost','Cash flow','Profit sharing','Net cash flows','PV of cash flows','Cash flow payout rate']
    column_header=[str(i) for i in range(1,period+1)]
    frame = pd.DataFrame(data, index=row_header, columns=column_header)

    # Updating table values
    # Row 1: Production level
    data[1,:]=[prod_level*(1-decline)**(i) for i in range(period)]
    # Row 0: Remaining reserves
    data[0,0]=reserve
    for i in range(1,period):
        data[0,i]=data[0,i-1]-data[1,i-1]
    # Row 2: Variable op cost rate
    # If the varaible "variable op cost" is a single number, then proceed with the calclations for the next row of data table; however if it is provided e.g. as a result of simulation, then just replace the empty row of data table with the already-available data
    if isinstance(op_cost,(int,float)):
        data[2,:]=[op_cost*(1+op_cost_rate)**(i) for i in range(1,period+1)]
    else:
        try:
            data[2,:]=op_cost
        except:
            print('The dimension of variable op cost data does not match the table row dimension\n')
    # Row 3: Oil price
    # If the varaible "oil price" is a single number, then proceed with the calclations for the next row of data table; however if it is provided e.g. as a result of simulation, then just replace the empty row of data table with the already-available data
    if isinstance(price,(int,float)):
        data[3,:]=[price*(1+price_growth)**(i) for i in range(1,period+1)]
    else:
        try:
            data[3,:]=price
        except:
            print('The dimension of oil price data does not match the table row dimension\n')
    # Row 4: Revenues = Production level * Oil price
    data[4,:]=data[3,:]*data[1,:]
    # Row 5: production cost = Production level * Variable op cost rate + Fixed cost
    data[5,:]=data[2,:]*data[1,:]+fixed_cost
    # Row 6: Cash flow = Revenues - Production cost
    data[6,:]=data[4,:]-data[5,:]
    # Row 7: Profit sharing = Revenues * Profit sharing rate
    data[7,:]=data[6,:]*profit
    # Row 8: Net cash flow = Cash flow - Profit sharing
    data[8,:]=data[6,:]-data[7,:]
    # Row 9: PV of cash flows (at each perid, t) = sum(NetCashFlow_i/((1+dr)^(i-t))), i=t,...,n
    for i in range(period-1,-1,-1):
        pv=[data[8,j]/((1+dr)**(j-i)) for j in range(period-1,i-1,-1)]
        data[9,i]=sum(pv)
    # Row 10: Cash flow payout rate = Net cash flows / PV of cash flows
    data[10,:]=data[8,:]/data[9,:]

    # Calculating the best estimate of the current market value of the project without options (base case). The Year 0 present value of the expected cash flows is calculated using the risk-adjusted discount rate.
    market_val=data[9,0]/(1+dr)
    # Estimating NPV of the project 
    NPV=market_val-init_invest
    return data,market_val,NPV
    