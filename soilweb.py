"""this code load the data containing lab results of different soil samples across Hungary,
they classification labels (that is whether a sample is an outlier or correct value). Then it does
preprocessing, feature selections,  restructures the data and run an anomaly detection  based geolocation of sampled points
and chemical properties of the sample analyzed in the laboratory. Finally the model is used to predict whether a sample is an outlier (erroneous) or True-value (correct value)
validation dataset.
"""


#Load important libraries for data loading and visualization
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


# Import the most important machine learning libraries for our modeling part
import scikitplot as skplt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import  GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import pickle5 as pickle
from pickle import dump
from pickle import load


# Load our data for training the model
# The class column in the data represent the target variable----with  value 1 representing correct samples (inliers)
# and the value 0 representing outliers
# The data was cleaned, and preprocessed for the modeling part
df_complete=pd.read_csv("~/df_complete.csv")

# the data has the following columns
# columns=["pH_KCl",
#          "KA",
#          "Ossz_so",
#          "CaCO3",
#          "Humusz",
#          "Mg",
#          "Na",
#          "Zn",
#          "Cu",
#          "Mn",
#          "SO4",
#          "x","y", "class"]


#Create arrays for model development: We chose to use the numpy arrays
# because they are faster to manipulate and process than the pandas dataframe


dt_array=df_complete.values
x=dt_array[:,0:13]# the values may change in the future as the data features increase
y=dt_array[:,13]# this represent the position of the class column----containing our target variable

#Create a training and validation set to evaluate models
#Set the seed to reproduce the results
# And specify the evaluation metric

number_folds=10
seed=7
scoring='f1'
validation_size=0.20
x_train, x_val, y_train,y_val=train_test_split(x,y,test_size=validation_size, random_state=seed)


# Base algorithms
models=[]
models.append(('LR',LogisticRegression()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))


# Evaluate models performance and plot the results as a figure and then save
# the figure in a specific folder
results=[]
names=[]
for name, model in models:
    kfold=KFold(n_splits=number_folds, random_state=seed,shuffle=True)
    cv_results=cross_val_score(model,x_train,y_train,cv=kfold, scoring=scoring, n_jobs=10)
    results.append(cv_results)
    names.append(name)
    print(name,'=', round(cv_results.mean(),2),'±',round(cv_results.std(),2))


# Visualize and compare base Algorithms
sns.set_style("whitegrid")
fig_base=plt.figure()
fig_base.suptitle('Comparison of Base Models')
ax = fig_base.add_subplot(111)
plt.boxplot(results,labels=names, showmeans=True)
ax.set_xticklabels(names)
plt.ylabel(" F1 score of the model")
plt.xlabel("Tested models")
fig_base.savefig("/home/drdc/Documents/drdc/open_projects/soilweb/4_results/Base_models.png",dpi=300)

# Train ensemble models
scoring='f1'
ensembles=[]

ensembles.append(('GBM',GradientBoostingClassifier()))
ensembles.append(('RF',RandomForestClassifier()))
ensembles.append(('ET',ExtraTreesClassifier()))

results_ens=[]
names_ens=[]
for name, model in ensembles:
    kfold=KFold(n_splits=number_folds, random_state=seed,shuffle=True)
    cv_results=cross_val_score(model,x_train,y_train,cv=kfold, scoring=scoring, n_jobs=10)
    results_ens.append(cv_results)
    names_ens.append(name)
    print(name,'=', round(cv_results.mean(),2),'±',round(cv_results.std(),2))

# Visualize and Compare Algorithms
fig_ens=plt.figure()
fig_ens.suptitle('Comparison of Ensemble models')
ax = fig_ens.add_subplot(111)
plt.boxplot(results_ens,labels=names_ens, showmeans=True)
ax.set_xticklabels(names_ens)
plt.ylabel(" F1 score of the model")
plt.xlabel("Tested models")
fig_ens.savefig("/home/drdc/Documents/drdc/open_projects/soilweb/4_results/Ensemble_models.png",dpi=300)


# Finetune the best model: ET
seed=7
param_grid = dict(n_estimators=np.array(np.arange(100,500,100)),
                  criterion=["gini"],min_samples_split=np.arange(1,5,1), max_features=np.arange(1,5,1))
model = ExtraTreesClassifier(random_state=seed, n_jobs=14)
kfold = KFold(n_splits=number_folds, random_state=seed, shuffle=True)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(x_train, y_train)
print("Best: %f using %s" % (round(grid_result.best_score_,2), grid_result.best_params_))


#Finalize the ET model
final_ET=ExtraTreesClassifier(random_state=seed,
                              criterion=grid_result.best_params_["criterion"],
                               max_features=grid_result.best_params_["max_features"],
                                 min_samples_split=grid_result.best_params_["min_samples_split"],
                                 n_estimators=grid_result.best_params_["n_estimators"])

final_ET_model=final_ET.fit(x_train, y_train)

# Run the evaluation on the training data
predict_train=final_ET.predict(x_train)
eval_on_training_data=classification_report(y_train, predict_train)
result_on_training_data=open("/home/drdc/Documents/drdc/open_projects/soilweb/4_results/report_on_training_data.txt", "w")
result_on_training_data.write(eval_on_training_data)
result_on_training_data.close()

#Make predictions on validation dataset
# And report classification report

predictions_ET= final_ET_model.predict(x_val)

#Validation_results=confusion_matrix(y_val, predictions_ET)
model_report=classification_report(y_val, predictions_ET)
model_result=open("/home/drdc/Documents/drdc/open_projects/soilweb/4_results/classification_report.txt", "w")
model_result.write(model_report)
model_result.close()

skplt.metrics.plot_confusion_matrix(y_val, predictions_ET, normalize=True, cmap="BrBG")

#Save the model locally for future use
# Load the important libraries for model saving
pickle.dump(final_ET, open("/home/drdc/Documents/drdc/open_projects/soilweb/3_code/ET_final_model.sav", "wb"))


# We can then load the the model and make prediction on unseen dataset

ET_model=load(open("/home/drdc/Documents/drdc/open_projects/soilweb/3_code/ET_final_model.sav", "rb"))
ET_model

#create a new dataframe for output based the validation test
col_names=['pH_KCl', 
           'KA', 
           'Ossz_so', 
           'CaCO3',
            'Humusz', 
           'Mg',
            'Na', 
            'Zn',
             'Cu',
             'Mn', 
             'SO4', 
             'x', 
             'y']

test_df=pd.DataFrame(x_val,columns=col_names)

test_df["label"]=predictions_ET
test_df=test_df.replace({"label":{0:"Possible-outlier", 1:"True-value"}})
test_df.head()

test_df.to_csv("/home/drdc/Documents/drdc/open_projects/soilweb/2_processed/test_labels.csv", index=False)
# create a figure to visualize the data
# Note that there are many reasons why a sample would be classified as an outlier 
sns.set_style("whitegrid")
fig = plt.subplots(figsize=(8, 8))

sns.scatterplot(y="y", x="x",
                hue="label", 
                palette="viridis",
                sizes=(1, 100), linewidth=0,
                data=test_df)
plt.ylabel("Latitude", size=15)
plt.xlabel("Longitide",size=15)
plt.legend(title="")  
plt.show()
