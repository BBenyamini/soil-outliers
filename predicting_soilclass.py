# import modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
import openpyxl

#machine learning libraries

import scikitplot as skplt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import  GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance

plt.style.use("bmh")


#load the data for model training
labdata_oldclass_merge=pd.read_csv("/home/drdc/Documents/drdc/open_projects/soilweb/2_processed/original_classes.csv")
labdata_newclass_merge=pd.read_csv("/home/drdc/Documents/drdc/open_projects/soilweb/2_processed/new_classes.csv")
df_complete=pd.read_csv("/home/drdc/Documents/drdc/open_projects/soilweb/2_processed/df_complete.csv")
validation_set_topsoil=pd.read_csv("/home/drdc/Documents/drdc/open_projects/soilweb/2_processed/validation-set-topsoil.csv")

## # Create test and evaluation metrics


x_Termohely=labdata_oldclass_merge.iloc[:,1:12]# the values may change in the future as the data features increase
y_Termohely=labdata_oldclass_merge.iloc[:,15]#

# Create test and evaluation metrics
num_folds=10
seed=7
scoring='accuracy'
val_size=0.20
seed=7
x_train_Termohely, x_val_Termohely, y_train_Termohely,y_val_Termohely=train_test_split(x_Termohely,y_Termohely,test_size=val_size, random_state=seed)

# compare algorithms
algorithms_Termohely=[]
scoring="accuracy"

algorithms_Termohely.append(('GBM',GradientBoostingClassifier()))
algorithms_Termohely.append(('RF',RandomForestClassifier()))
algorithms_Termohely.append(('ET',ExtraTreesClassifier()))

results_Termohely=[]
names_Termohely=[]
for name, algorithm in algorithms_Termohely:
    kfold=KFold(n_splits=num_folds, random_state=seed,shuffle=True)
    cv_results=cross_val_score(algorithm,x_train_Termohely,y_train_Termohely,cv=kfold, scoring=scoring, n_jobs=-1)
    results_Termohely.append(cv_results)
    names_Termohely.append(name)

# visualize and compare Algorithms
plt.style.use("bmh")
fig=plt.figure()
fig.suptitle('Comparison of ensemble models')
ax = fig.add_subplot(111)
plt.boxplot(results_Termohely,labels=names_Termohely, showmeans=True)
ax.set_xticklabels(names_Termohely)
plt.show();


# combine the three ensembles to develop a voting classifier for Termohely

Termohely_estimators = [
    ("RF", RandomForestClassifier(random_state=seed)),
    ("ET", ExtraTreesClassifier(random_state=seed)),
    ("GBM", GradientBoostingClassifier(random_state=seed))]

voting_clf_Termohely = VotingClassifier(Termohely_estimators)

voting_clf_Termohely.fit(x_train_Termohely, y_train_Termohely)


#Make predictions on validation dataset of

predictions_Termohely= voting_clf_Termohely.predict(x_val_Termohely)
print(voting_clf_Termohely.score(x_val_Termohely, y_val_Termohely))
print(confusion_matrix(y_val_Termohely, predictions_Termohely))
Termohely_eval=classification_report(y_val_Termohely, predictions_Termohely)

result_on_Termohely=open("/home/drdc/Documents/drdc/open_projects/soilweb/4_results/test/report_on_Termohely.csv", "w")
result_on_Termohely.write(Termohely_eval)
result_on_Termohely.close()

skplt.metrics.plot_confusion_matrix(y_val_Termohely, predictions_Termohely, normalize=True, cmap="mako_r")
plt.xticks(rotation=45, ha="right")
plt.show();

# Get feature importance
def calculate_feature_importance(voting_clf, weights):
    """ This Function calculates feature importance of Voting Classifier """

    feature_importance = {}
    for estimator in voting_clf_Termohely.estimators_:
        feature_importance[str(estimator)] = estimator.feature_importances_

    feature_scores = [0]*len(list(feature_importance.values())[0])
    for id, imp_score in enumerate(feature_importance.values()):
        imp_score_with_weight = imp_score*weights[id]
        feature_scores = list(np.add(feature_scores, list(imp_score_with_weight)))
    return feature_scores

#create a dataframe of the features and their corresponding fe_scores
Termohely_df = pd.DataFrame()
Termohely_df['Feature'] = x_train_Termohely.columns
Termohely_df['Feature Importance'] = calculate_feature_importance(voting_clf_Termohely, [1, 1, 2])

#plot features feature_importances_
sns.barplot(data=Termohely_df.sort_values(by="Feature Importance", ascending=False), y="Feature",
 x="Feature Importance",color="#5ab4ac")
plt.title(" Voting classifier Feature Importances")
plt.ylabel("")
plt.xlabel("")
plt.show();


permit_Termohely= permutation_importance(
    voting_clf_Termohely, x_val_Termohely, y_val_Termohely, n_repeats=10, random_state=7, n_jobs=20
)

sorted_importances_idx = permit_Termohely.importances_mean.argsort()
importances = pd.DataFrame(
    permit_Termohely.importances[sorted_importances_idx].T,
    columns=x_Termohely.columns[sorted_importances_idx],
)
ax = importances.plot.box(vert=False, whis=10)
ax.set_title("Permutation Importances (For validation set)")
ax.axvline(x=0, color="k", linestyle="--")
ax.set_xlabel("Decrease in accuracy score")
ax.figure.tight_layout()
plt.show();


# combine the three ensembles to develop a voting classifier for Fotipus
x_Fotipus=labdata_oldclass_merge.iloc[:,1:12]
y_Fotipus=labdata_oldclass_merge.iloc[:,14]
x_train_Fotipus, x_val_Fotipus, y_train_Fotipus,y_val_Fotipus=train_test_split(x_Fotipus,y_Fotipus,test_size=val_size, random_state=seed)


# combine the three ensembles to develop a voting classifier for Fotipus

Fotipus_estimators = [
    ("RF", RandomForestClassifier(random_state=seed)),
    ("ET", ExtraTreesClassifier(random_state=seed)),
    ("GBM", GradientBoostingClassifier(random_state=seed))]

voting_clf_Fotipus = VotingClassifier(Fotipus_estimators)

voting_clf_Fotipus.fit(x_train_Fotipus, y_train_Fotipus)


#Make predictions on validation dataset

predictions_Fotipus= voting_clf_Fotipus.predict(x_val_Fotipus)
print(voting_clf_Fotipus.score(x_val_Fotipus, y_val_Fotipus))
print(confusion_matrix(y_val_Fotipus, predictions_Fotipus))
Fotipus_eval=classification_report(y_val_Fotipus, predictions_Fotipus)

result_on_Fotipus=open("/home/drdc/Documents/drdc/open_projects/soilweb/4_results/test/report_on_Fotipus.csv", "w")
result_on_Fotipus.write(Fotipus_eval)
result_on_Fotipus.close()

#plot the confusion confusion_matrix
skplt.metrics.plot_confusion_matrix(y_val_Fotipus, predictions_Fotipus, normalize=True, cmap="copper_r")
plt.xticks(rotation=45, ha="right");
plt.show();

#look at the feature importance
Fotipus_df = pd.DataFrame()
Fotipus_df['Feature'] = x_train_Fotipus.columns
Fotipus_df['Feature Importance'] = calculate_feature_importance(voting_clf_Fotipus, [1, 1, 1])


sns.barplot(data=Fotipus_df.sort_values(by="Feature Importance", ascending=False), y="Feature",
 x="Feature Importance",color="#176B87")

plt.title(" Voting classifier Feature Importances")
plt.ylabel("")
plt.xlabel("")
plt.show();


permit_Fotipus = permutation_importance(
    voting_clf_Fotipus, x_val_Fotipus, y_val_Fotipus, n_repeats=10, random_state=7, n_jobs=20
)

sorted_importances_idx = permit_Fotipus.importances_mean.argsort()
importances = pd.DataFrame(
    permit_Fotipus.importances[sorted_importances_idx].T,
    columns=x_Fotipus.columns[sorted_importances_idx],
)
ax = importances.plot.box(vert=False, whis=10)
ax.set_title("Permutation Importances (For validation set)")
ax.axvline(x=0, color="k", linestyle="--")
ax.set_xlabel("Decrease in accuracy score")
ax.figure.tight_layout()
plt.show();

# combine the three ensemblesble to develop a voting classifier for new classes
x_newclass=labdata_newclass_merge.iloc[:,2:12]
y_newclass=labdata_newclass_merge.iloc[:,1]
x_train_newclass, x_val_newclass, y_train_newclass,y_val_newclass=train_test_split(x_newclass,y_newclass,test_size=val_size, random_state=seed)

newclass_estimators = [
    ("RF", RandomForestClassifier(random_state=seed)),
    ("ET", ExtraTreesClassifier(random_state=seed)),
    ("GBM", GradientBoostingClassifier(random_state=seed))]

voting_clf_newclass = VotingClassifier(newclass_estimators)

voting_clf_newclass.fit(x_train_newclass, y_train_newclass)

#Make predictions on validation dataset

predictions_newclass= voting_clf_newclass.predict(x_val_newclass)
print(voting_clf_newclass.score(x_val_newclass, y_val_newclass))
print(confusion_matrix(y_val_newclass, predictions_newclass))
newclass_eval=classification_report(y_val_newclass, predictions_newclass)

#plot the confusion confusion_matrix
skplt.metrics.plot_confusion_matrix(y_val_newclass, predictions_newclass, normalize=True, cmap="terrain_r")
plt.xticks(rotation=45, ha="right");
plt.show();

result_on_newclass=open("/home/drdc/Documents/drdc/open_projects/soilweb/4_results/test/report_on_newclass.csv", "w")
result_on_newclass.write(newclass_eval)
result_on_newclass.close()


newclass_df = pd.DataFrame()
newclass_df['Feature'] = x_train_newclass.columns
newclass_df['Feature Importance'] = calculate_feature_importance(voting_clf_newclass, [1, 1, 1])


sns.barplot(data=newclass_df.sort_values(by="Feature Importance", ascending=False), y="Feature",
 x="Feature Importance",color="#CDC2AE")

plt.title(" Voting classifier Feature Importances")
plt.ylabel("")
plt.xlabel("")
plt.show();


#final prediction on laboratory data
#rename variables to harmonize names of the predictors
df_complete.rename(columns={"Ossz_so":"Osszso"}, inplace=True)


#specify important columns
columns=["pH_KCl",
         "KA",
         "Osszso",
         "CaCO3",
         "Humusz",
         "Mg",
         "Na",
         "Zn",
         "Cu",
         "Mn",
         "SO4",
         "x","y"]

def final_predictions(df, model1, model2, model3):
    """
    the function runs predictions on new data and return a dataframe with all predicted classes
    """
    df["Fotipus_pred"]=model1.predict(df[columns])
    df["Termohely_pred"]=model2.predict(df[columns])
    df["Newclass_pred"]=model3.predict(df[columns])
    return df

#Note. you can change the final directories
df_complete_pred=final_predictions(df_complete, voting_clf_Fotipus, voting_clf_Termohely, voting_clf_newclass)
df_complete_pred.to_csv("/home/drdc/Documents/drdc/open_projects/soilweb/2_processed/df_complete_pred_class.csv", index=False)

validation_set_topsoil=pd.read_csv("/home/drdc/Documents/drdc/open_projects/soilweb/2_processed/validation-set-topsoil.csv")
validation_set_topsoil_pred=final_predictions(validation_set_topsoil, voting_clf_Fotipus, voting_clf_Termohely, voting_clf_newclass)

validation_set_topsoil_pred.to_csv("/home/drdc/Documents/drdc/open_projects/soilweb/2_processed/df_validation_pred_class.csv", index=False)
