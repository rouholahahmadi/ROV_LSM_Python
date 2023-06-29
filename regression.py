class LinearRegression:
    """ 
    This class offers a variety of linear regression tools for predicting a vector of Y values vs. an array of X values where X can have any desired number of features. The models considered in this calss are as follows:
    1. Linear regression with K-fold cross validation
    2. Linear regression with gradient boosting
    3. Polynomial regression 
    """
    # The references for algorithms are as follows
    # 1. https://towardsdatascience.com/machine-learning-with-python-regression-complete-tutorial-47268e546cea
    # 2. https://scikit-learn.org/stable/modules/linear_model.html
    
    # Importing required packages
    # for data
    import pandas as pd
    import numpy as np
    # for plotting
    import matplotlib.pyplot as plt
    import seaborn as sns
    # for statistical tests
    import scipy
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    # for machine learning
    from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression
    # for explainer
    from lime import lime_tabular

    def __init__(self,x,y,heading,test_split=0.25):
        # Input parameters
        # x             Array of independant variables, with size N*M, where N is the number of features and M is the number of observations  (The program then transposes the final design matrix so that the features conform to the columns)
        # y             Array of output data, with size 1*M
        # heading       Heading for x and y data columns
        # test_split    Fraction of testing data when splitting the dataset into test and train data
        self.x=x
        self.y=y
        self.cols=heading
        self.test_split=test_split
        # Importing required packages
        import pandas as pd
        import numpy as np
        from sklearn import model_selection
        # Preparing data sets for regression
        data=np.append(x,y,axis=0).transpose()
        self.dtf=pd.DataFrame(data, columns = self.cols)
        # Splitting data
        if self.test_split>0:
            self.dtf_train, self.dtf_test = model_selection.train_test_split(self.dtf,test_size=self.test_split)
            self.X_test = self.dtf_test.drop(self.cols[-1], axis=1).values
            self.y_test = self.dtf_test[self.cols[-1]].values
            self.X_train = self.dtf_train.drop(self.cols[-1], axis=1).values
            self.y_train = self.dtf_train[self.cols[-1]].values
        else:
            self.X_test = []
            self.y_test = []
            self.X_train = self.dtf.drop(self.cols[-1], axis=1).values
            self.y_train = self.dtf[self.cols[-1]].values
        self.feature_names = self.dtf.drop(self.cols[-1], axis=1).columns.tolist()

    def kfoldCV(self,k=5,plot="True"):
        """ This function designs a linear regression model using K-fold cross validation approach"""
        # Input parameters
        # k         Number of splits in k-fold CV method
        # plot      An option indicating if the results should be plotted or not
        self.k=k
        self.plot=plot
        # Importing required packages
        import numpy as np
        from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
        from sklearn.model_selection import TimeSeriesSplit
        import matplotlib.pyplot as plt

        # K-fold Cross Validation technique
        # Calling model
        model = linear_model.LinearRegression()
        # K fold validation
        scores = []     # Correlation coefficient
        mape = []       # Mean absolute percentage error
        cv = model_selection.KFold(n_splits=self.k, shuffle=True)
        fig = plt.figure()
        i = 1
        for train, test in cv.split(self.X_train, self.y_train):
            prediction = model.fit(self.X_train[train],self.y_train[train]).predict(self.X_train[test])
            true = self.y_train[test]
            # Estimating correlation coefficient
            score = metrics.r2_score(true, prediction)
            scores.append(score)
            # Estimating mape
            err=round(np.mean(np.abs((true-prediction)/prediction)), 2)
            mape.append(err)
            plt.scatter(prediction, true, lw=2, alpha=0.3, label='Fold %d (R2 = %0.2f)' % (i,score))
            i = i+1
        score_mean=np.mean(scores)
        mape_mean=np.mean(mape)
        print('Average R2 = ',score_mean)
        if self.plot=="True":
            plt.plot([min(self.y_train),max(self.y_train)], [min(self.y_train),max(self.y_train)], linestyle='--', lw=2, color='black')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('K-Fold Validation')
            plt.legend()
            plt.show()
        return model, score_mean, mape_mean

    def GradientBoosting(self,plot="True"):
        """ This function designs a gradient boosting regressor for linear regression."""
        # Input parameters
        # plot      An option indicating if the results should be plotted or not
        self.plot=plot
        # Importing required packages
        # for data
        import numpy as np
        import pandas as pd
        # for plotting
        import matplotlib.pyplot as plt
        import seaborn as sns
        # for statistical tests
        import scipy
        import statsmodels.formula.api as smf
        import statsmodels.api as sm
        # for machine learning
        from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
        # for explainer
        from lime import lime_tabular

        # Gradient Boosting method
        # Calling model
        model = ensemble.GradientBoostingRegressor()
        # Assessing variable importance
        model.fit(self.X_train,self.y_train)
        importances = model.feature_importances_
        # Puting in a pandas dtf
        dtf_importances = pd.DataFrame({"IMPORTANCE":importances, "VARIABLE":self.feature_names}).sort_values("IMPORTANCE", ascending=False)
        dtf_importances['cumsum'] =  dtf_importances['IMPORTANCE'].cumsum(axis=0)
        dtf_importances = dtf_importances.set_index("VARIABLE")
            
        # Plotting results
        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False)
        fig.suptitle("Features Importance", fontsize=20)
        ax[0].title.set_text('variables')
        dtf_importances[["IMPORTANCE"]].sort_values(by="IMPORTANCE").plot(kind="barh", legend=False, ax=ax[0]).grid(axis="x")
        ax[0].set(ylabel="")
        ax[1].title.set_text('cumulative')
        dtf_importances[["cumsum"]].plot(kind="line", linewidth=4, legend=False, ax=ax[1])
        ax[1].set(xlabel="", xticks=np.arange(len(dtf_importances)), xticklabels=dtf_importances.index)
        plt.xticks(rotation=70)
        plt.grid(axis='both')
        plt.show()
        # Modeling
        # Fitting model using train data
        model.fit(self.X_train, self.y_train)
        # Predicting with test or train data
        if not self.X_test:
            predicted = model.predict(self.X_train)
            true = self.y_train
        else:
            predicted = model.predict(self.X_test)
            true = self.y_test
        # Estimating correlation coefficient 
        score = metrics.r2_score(true, predicted) 
        # Estimating mape
        mape=round(np.mean(np.abs((true-predicted)/predicted)), 2)
        print('Average R2 = ',score)

        if self.plot=="True":
            # Kpi
            print("R2 (explained variance):", round(metrics.r2_score(self.y_test, predicted), 2))
            print("Mean Absolute Perc Error (Σ(|y-pred|/y)/n):", round(np.mean(np.abs((self.y_test-predicted)/predicted)), 2))
            print("Mean Absolute Error (Σ|y-pred|/n):", "{:,.0f}".format(metrics.mean_absolute_error(self.y_test, predicted)))
            print("Root Mean Squared Error (sqrt(Σ(y-pred)^2/n)):", "{:,.0f}".format(np.sqrt(metrics.mean_squared_error(self.y_test, predicted))))
            # Assessing residuals
            residuals = self.y_test - predicted
            max_error = max(residuals) if abs(max(residuals)) > abs(min(residuals)) else min(residuals)
            max_idx = list(residuals).index(max(residuals)) if abs(max(residuals)) > abs(min(residuals)) else list(residuals).index(min(residuals))
            max_true, max_pred = self.y_test[max_idx], predicted[max_idx]
            print("Max Error:", "{:,.0f}".format(max_error))

            # Plotting predicted vs true
            fig, ax = plt.subplots(nrows=1, ncols=2)
            from statsmodels.graphics.api import abline_plot
            ax[0].scatter(predicted, self.y_test, color="black")
            abline_plot(intercept=0, slope=1, color="red", ax=ax[0])
            ax[0].vlines(x=max_pred, ymin=max_true, ymax=max_true-max_error, color='red', linestyle='--', alpha=0.7, label="max error")
            ax[0].grid(True)
            ax[0].set(xlabel="Predicted", ylabel="True", title="Predicted vs True")
            ax[0].legend()
                
            # Plotting predicted vs residuals
            ax[1].scatter(predicted, residuals, color="red")
            ax[1].vlines(x=max_pred, ymin=0, ymax=max_error, color='black', linestyle='--', alpha=0.7, label="max error")
            ax[1].grid(True)
            ax[1].set(xlabel="Predicted", ylabel="Residuals", title="Predicted vs Residuals")
            ax[1].hlines(y=0, xmin=np.min(predicted), xmax=np.max(predicted))
            ax[1].legend()
            plt.show()

            # Plotting distribution of the residuals and see if it looks approximately normal:
            fig, ax = plt.subplots()
            sns.distplot(residuals, color="red", hist=True, kde=True, kde_kws={"shade":True}, ax=ax)
            ax.grid(True)
            ax.set(yticks=[], yticklabels=[], title="Residuals distribution")
            plt.show()
        return model, score, mape

    def Polynomial(self,order,plot="True"):
        """ This function designs a polynomial multivariate regression."""
        # Input parameters
        # order     Order of polynomial
        # plot      An option indicating if the results should be plotted or not
        self.order=order
        self.plot=plot
        # Importing required packages
        # for data
        import numpy as np
        # for plotting
        import matplotlib.pyplot as plt
        import seaborn as sns
        # for machine learning
        from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LinearRegression
        # Generating a model of polynomial features
        # Polynomial regression could be fulfilled by transforming the x data into an expanded space of more number of features (for single variable type and order "n" it returns [1,x,x**2,...,x**n])
        # This sort of preprocessing can be streamlined with the Pipeline tools. A single object representing a simple polynomial regression can be created and used as follows:
        model = Pipeline([('poly', PolynomialFeatures(degree=self.order)),('linear', LinearRegression(fit_intercept=False))])
        # Fitting to a polynomial of order "n"
        model = model.fit(self.X_train, self.y_train.reshape(-1,1))
        model.named_steps['linear'].coef_
        # Model valuation
        if not self.X_test:
            predicted = model.predict(self.X_train)
            true = self.y_train
        else:
            predicted = model.predict(self.X_test)
            true = self.y_test
        score = metrics.r2_score(true, predicted)    
        # Estimating MAPE
        mape=round(np.mean(np.abs((true-predicted)/predicted)), 2)
        print("Mean Absolute Perc Error (Σ(|y-pred|/y)/n):", mape)
        
        if self.plot=="True":
            # Assessing residuals
            residuals = self.y_test - predicted
            max_error = max(residuals) if abs(max(residuals)) > abs(min(residuals)) else min(residuals)
            max_idx = list(residuals).index(max(residuals)) if abs(max(residuals)) > abs(min(residuals)) else list(residuals).index(min(residuals))
            max_true, max_pred = self.y_test[max_idx], predicted[max_idx]
            print("Max Error:", "{:,.0f}".format(max_error))

            # Plotting predicted vs true
            fig, ax = plt.subplots(nrows=1, ncols=2)
            from statsmodels.graphics.api import abline_plot
            ax[0].scatter(predicted, self.y_test, color="black")
            abline_plot(intercept=0, slope=1, color="red", ax=ax[0])
            ax[0].vlines(x=max_pred, ymin=max_true, ymax=max_true-max_error, color='red', linestyle='--', alpha=0.7, label="max error")
            ax[0].grid(True)
            ax[0].set(xlabel="Predicted", ylabel="True", title="Predicted vs True")
            ax[0].legend()
                
            # Plotting predicted vs residuals
            ax[1].scatter(predicted, residuals, color="red")
            ax[1].vlines(x=max_pred, ymin=0, ymax=max_error, color='black', linestyle='--', alpha=0.7, label="max error")
            ax[1].grid(True)
            ax[1].set(xlabel="Predicted", ylabel="Residuals", title="Predicted vs Residuals")
            ax[1].hlines(y=0, xmin=np.min(predicted), xmax=np.max(predicted))
            ax[1].legend()
            plt.show()

            # Plotting distribution of the residuals and see if it looks approximately normal:
            fig, ax = plt.subplots()
            sns.distplot(residuals, color="red", hist=True, kde=True, kde_kws={"shade":True}, ax=ax)
            ax.grid(True)
            ax.set(yticks=[], yticklabels=[], title="Residuals distribution")
            plt.show()
        return model, score, mape

    def modelSelection(self,maxOrder,minOrder):
        """ This program attempts to select a model with the lowest value of MSE or AIC measures. """
        # Input parameters
        # maxOrder          Maximum polynomial order for model selection
        # minOrder          Minimum polynomial order for model selection (can take negative values)

        # Reference for more concepts
        # https://machinelearningmastery.com/probabilistic-model-selection-measures/
        # for machine learning
        from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition, datasets
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LinearRegression
        import numpy as np
        # The AIC calculation for an ordinary least squares linear regression model can be calculated as follows (taken from “A New Look At The Statistical Identification Model“,  1974.): AIC = n * LL + 2 * k
        # Where n is the number of examples in the training dataset, LL is the log-likelihood for the model using the natural logarithm (e.g. the log of the MSE), and k is the number of parameters in the model.
        # calculate aic for regression
        def calculate_aic(n, mse, num_params):
            """ This function estimates the AIC criterion for an ordinary regression. """
            # Input parameters
            # n             Number of examples in the training dataset
            # mse           Mean squared error
            # num_params    Number of parameters in the model
            import numpy as np
            aic = n * np.log(mse) + 2 * num_params
            return aic
        # Making an instance of class LinearRegression (for running a (multivariate) regression relating the realized continuation values (y) to the uncertain oil prices (x1) and variable operating costs (x2) at every period with options)
        model = LinearRegression()
        # Making a simple linear regression model (first order polynomial):
        # Model fitting
        model.fit(self.X_train, self.y_train)
        # number of parameters
        num_params = len(model.coef_) + 1
        # Predicting the training set
        if not np.any(self.X_test):
            predicted = model.predict(self.X_train)
            true = self.y_train
        else:
            predicted = model.predict(self.X_test)
            true = self.y_test
        # calculating mean squared error
        mse = metrics.mean_squared_error(true, predicted)
        # Calculating correlation coefficient
        score = metrics.r2_score(true, predicted) 
        print('mse, score = ',mse,score)
        # calculating aic
        aicOld = calculate_aic(len(true), mse, num_params)
        # We prefer to use mse as the regression performance criteria. So we use "mse" as "aic" in the following notations interchangeably.
        aicNew=mse
        flag=True
        xBest_train=self.X_train
        xBest_test=self.X_test
        c=-1
        aicNewList=[aicNew]
        scoreNewList=[score]
        powerNewList=[]  # list of power of polynomials for first order polynomial which is the basic model prior to adding other components to the regression equation
        # Defining a function for breaking an integer (order of polynomial) into n (number of input features in X_train) sub-integers
        def intBreak(intVal,n):
            """ This function breaks an integer down to a set of n non-negative integers (each less than or equal to that) so that the sum of all sub-integers equals the main integer. """
            # Input parameters
            # intVal               Integer
            # n                 Desired number of sub-integers
            # Importing required package
            from itertools import product
            # Inigtialzing list of integers with zero values
            path=[]
            lists=[]
            for i in range(n):
                lists.append([j for j in range(intVal+1)])
            for items in product(*lists):
                if (sum(items)==intVal):
                    path.append(items)
            return path
        # Adding more features to the initial feature set by gradually making powers and cross terms
        while flag:
            c += 1
            aicList=[]
            f_train=[]      # List of feature values
            f_test=[]
            power=[]        # Power of features
            scoreList=[]    # Correlation coefficient values
            aicOld=aicNew
            # Extracting number of input features
            nf=self.X_train.shape[1]
            for i in range(minOrder,maxOrder+1):
                # Breaking down current order into nf number of integers to generate cross products between the original features
                path=intBreak(i,nf)
                for j in range(len(path)):
                    # Making new feature: (X1^j1)*(X2^j2)*...*(X_nf^j_nf)
                    feature_train=self.X_train[:,0]**path[j][0]
                    for k in range(1,nf):
                        feature_train=feature_train*(self.X_train[:,k]**path[j][k])
                    # Adding this feature to the previous X dataset
                    xNew_train=np.c_[xBest_train,feature_train]
                    if np.any(self.X_test):
                        feature_test=self.X_test[:,0]**path[j][0]
                        for k in range(1,nf):
                            feature_test=feature_test*(self.X_test[:,k]**path[j][k])
                        xNew_test=np.c_[xBest_test,feature_test]
                    # Model fitting
                    model.fit(xNew_train, self.y_train)
                    # number of parameters
                    num_params = len(model.coef_) + 1
                    # Predicting the training set
                    if not np.any(self.X_test):
                        yhat = model.predict(xNew_train)
                        y = self.y_train
                    else:
                        yhat = model.predict(xNew_test)
                        y = self.y_test
                    # calculating the error
                    mse = metrics.mean_squared_error(y, yhat)
                    # calculating the aic
                    aic = calculate_aic(len(y), mse, num_params)
                    # Calculate r2
                    score = metrics.r2_score(y, yhat) 
                    aicList.append(mse)
                    scoreList.append(score)
                    f_train.append(feature_train)
                    if np.any(self.X_test):
                        f_test.append(feature_test)
                    power.append(path[j])
            aicNew=min(aicList)
            aicNewIndex=aicList.index(aicNew)
            featureNew_train=f_train[aicNewIndex]
            if np.any(self.X_test):
                featureNew_test=f_test[aicNewIndex]
            powerNew=power[aicNewIndex]
            scoreNew=scoreList[aicNewIndex]
            flag=((aicOld-aicNew)/aicOld)>1e-4
            if flag:
                xBest_train=np.c_[xBest_train,featureNew_train]
                if np.any(self.X_test):
                    xBest_test=np.c_[xBest_test,featureNew_test]
                aicNewList.append(aicNew)
                scoreNewList.append(scoreNew)
                powerNewList.append(powerNew)
        # Making final model once more again:
        bestModel=model.fit(xBest_train, self.y_train)
        coef=model.coef_
        return bestModel, coef, xBest_train, aicNewList, scoreNewList, powerNewList