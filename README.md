# MNAR_exp


1. Run ```create_data.ipynb```: load raw data, do the normalization, and do the datasplit of each data.
   datalist : [
      
      "banknote":https://archive.ics.uci.edu/dataset/267/banknote+authentication
      1372 * 5  Classification
       1. variance of Wavelet Transformed image (continuous) 
       2. skewness of Wavelet Transformed image (continuous)
       3. curtosis of Wavelet Transformed image (continuous)
       4. entropy of image (continuous)
       5. class (integer) 
      
      "concrete_compression":https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
      Regression
      Number of instances 	1030
      Number of Attributes	9
      Attribute breakdown	8 quantitative input variables, and 1 quantitative output variable
      
      "wine_quality_white":  https://archive.ics.uci.edu/dataset/186/wine+quality
      "wine_quality_red":  Regression
      Number of instances 	1599 red , 4898 white
      Number of Attributes	12
      Attribute breakdown	11 quantitative input variables, and 1 quantitative output variable


      "california":https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
      Regression
      Samples total 20640 
      feature 9 + 1
      
      "climate_model_crashes":https://archive.ics.uci.edu/dataset/252/climate+model+simulation+crashes 
      Classification (Discrete Variable Col1, Col2 Mix Type)
      Column 1: Latin hypercube study ID (study 1 to study 3)

      Column 2: simulation ID (run 1 to run 180)

      Columns 3-20: values of 18 climate model parameters scaled in the interval [0, 1]

      Column 21: simulation outcome (0 = failure, 1 = success)


      "connectionist_bench_sonar":https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks
      Classification 
      Instances 208, Features 60



      "qsar_biodegradation":https://archive.ics.uci.edu/dataset/254/qsar+biodegradation 
      1055 instances
      41 molecular descriptors and 1 experimental class:
      
      "yeast":https://archive.ics.uci.edu/dataset/110/yeast
      1484 * 8+1, Classification

      "yacht_hydrodynamics":https://archive.ics.uci.edu/dataset/243/yacht+hydrodynamics
      Regression, 308 * 6+1
            ]

2. Run```create_missing.ipynb```
   Create Missing Masks, include missing rate

3. Run ```create_visualization.ipynb```
   Create Missing Mech ScatterPlot, Missing Rate Plot, Missing Distribution Plot

4. Run models
   Under Model folders, each model will create a train and test data
   ```mean.ipynb``` Mean Value imputation from Sklearn

5. Run Evaluation
   RMSE
   Downstream
   