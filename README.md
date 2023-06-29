# ROV_LSM_Python
This code implements real options valuation using the Least Square Monte Carlo (LSM) approach for a typical oil production project called BDH as described in https://doi.org/10.1016/j.petrol.2022.111230.

Please take the following steps to run the LSM valuation code:
1.	Run the file “Main_LSM” to implement real option valuation of BDH example using LSM technique (Other files are separate classes that are employed by the main code).
The code is comprised of different parts as listed below:
1.1.	 Package importing
1.2.	 Input properties
1.2.1.	Oil field input parameters 
1.2.2.	Price model input parameters
1.2.3.	Model configuration
1.2.4.	Additional settings (including the type of stochastic price model (GBM or MR), type of regression  (K-fold cross validation, gradient boosting, polynomial, feature selection), path alternatives (Alternative 1 through Alternative 5)
1.2.5.	Integrating different input properties together
1.2.6.	Flag setting (optional)
1.3.	Real option valuation
1.3.1.	Option definition
1.3.2.	LSM valuation
1.3.2.1.	Seed setting
1.3.2.2.	Project valuation using LSM technique
1.3.2.3.	Returning and printing some useful outputs (Outputs are described carefully in the code)
1.3.2.4.	Scatter plot of Continuation Value vs. Uncertain Variables along with the regression plane (in 3D) or regression line (in 2D)
2.	To configure different properties of the model, change the relevant properties within the code. The current code, as it appears, runs the BDH example using the LSM technique when the Buyout and Divest options are included at Year 5. 
The code could be run with multiple option definitions at any number of periods (Multiple periods can contain multiple options.)
NB! This code can have only two uncertain variables including Oil Price and Variable Operating Cost. It should be modified if more number of uncertain variables are to be considered in the model. 

