import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import statsmodels.api as sm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
import os
import shap

if __name__ == "__main__":

    if(len(sys.argv)<4):
        print("ERROR! Usage: python scriptName.py fileCSV targetN modelloML\n")
              
        sys.exit(1)
    nome_script, pathCSV, targId, mlModel = sys.argv

    targetId = int(targId)

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"

    back = ['Artistic', 'Scientific']
    pos = 1
    if (pathCSV == 'datasetArtisticBackgroundFINAL.csv'):
        pos = 0


    dataset = pd.read_csv(pathCSV, sep=';')

    index_target= dataset.iloc[:,-7:]
    list_ind_t = index_target.columns.values.tolist()
    targetN = list_ind_t[targetId]

    X = dataset[['timeDuration', 'nMovements', 'movementsDifficulty', 'AItechnique', 'robotSpeech',    'acrobaticMovements', 'movementsRepetition', 'musicGenre', 'movementsTransitionsDuration', 'humanMovements', 'balance', 'speed', 'bodyPartsCombination', 'musicBPM', 'sameStartEndPositionPlace', 'headMovement', 'armsMovement', 'handsMovement', 'legsMovement', 'feetMovement']]
    y = dataset[targetN]

    categorical_features = ['AItechnique', 'musicGenre']
    categorical_transformer = Pipeline(steps=[
                                          ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                          ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    numeric_features = ['timeDuration', 'nMovements', 'movementsDifficulty', 'robotSpeech',    'acrobaticMovements', 'movementsRepetition', 'movementsTransitionsDuration', 'humanMovements', 'balance', 'speed', 'bodyPartsCombination', 'musicBPM', 'sameStartEndPositionPlace', 'headMovement', 'armsMovement', 'handsMovement', 'legsMovement', 'feetMovement']
    numeric_transformer = Pipeline(steps=[
                                      ('imputer', SimpleImputer(strategy='median')),
                                      ('scaler', RobustScaler())])

    preprocessor = ColumnTransformer(
                                 transformers=[
                                               ('num', numeric_transformer, numeric_features),
                                               ('cat', categorical_transformer, categorical_features)])
    model_reg = ['lr',
                'dt',
                'rf',
                'mlp']

    param_lr = [{'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}]

    param_dt = [{'max_depth': [5, 10, 15, 20, 25, 30]}]

    param_mlp = [{'hidden_layer_sizes': [(5,), (10,), (15,)],
                 'activation': ['tanh', 'relu'],
                 'solver': ['sgd', 'adam'],
                 'alpha': [0.0001, 0.05],
                 'learning_rate': ['constant','adaptive'],}]

    param_rf = [{'bootstrap': [True, False],
                 'max_depth': [10, 20, 30],
                 'max_features': ['auto', 'sqrt'],
                 'min_samples_leaf': [1, 2, 4],
                 'min_samples_split': [2],}]

    models_regression = {
        'lr': {'name': 'Linear Regression',
               'estimator': LinearRegression(),
               'param': param_lr,
              },
        'dt': {'name': 'Decision Tree',
            'estimator': DecisionTreeRegressor(random_state=42),
            'param': param_dt,
              },
        'rf': {'name': 'Random Forest',
               'estimator': RandomForestRegressor(random_state=42),
               'param': param_rf,
              },
        'mlp': {'name': 'Multi Linear Perceptron',
                'estimator': MLPRegressor(random_state=42),
                'param': param_mlp
               },
    }

    mod_grid = GridSearchCV(models_regression[mlModel]['estimator'], models_regression[mlModel]['param'], cv=5, return_train_score = False, scoring='neg_mean_squared_error', n_jobs = 8)

    data_train, data_test, target_train, target_test = train_test_split(
                                                                    X, y, test_size=0.2, random_state=42)
    model = Pipeline(steps=[('preprocessor', preprocessor),
                ('regressor', mod_grid)])


    _ = model.fit(data_train, target_train)

    target_pred = model.predict(data_test)



#################### PLOT ##############################

    plt.scatter(target_test, target_pred, color="black")
    plt.savefig('Results-%s/Results-%s/%s/Plot/scatter.png' %(back[pos], mlModel,targetN))
    plt.clf()
    plt.cla()
    plt.close()
    plt.hist(target_test - target_pred)
    plt.savefig('Results-%s/Results-%s/%s/Plot/hist.png'  %(back[pos], mlModel,targetN))
    plt.clf()
    plt.cla()
    plt.close()

######### FEATURE SCORES ###########

    feature_cat_names = model['preprocessor'].transformers_[1][1]['onehot'].get_feature_names(categorical_features)

    l= feature_cat_names.tolist()
    ltot = numeric_features + l

    importance = []

    if (mlModel=='lr'):
              importance = mod_grid.best_estimator_.coef_
              coefs = pd.DataFrame(mod_grid.best_estimator_.coef_,
                                   columns=["Coefficients"],
                                   index= ltot)
                                   
    elif (mlModel=='dt' or mlModel=='rf'):
              importance = mod_grid.best_estimator_.feature_importances_
              coefs = pd.DataFrame(mod_grid.best_estimator_.feature_importances_,
                                   columns=["Coefficients"],
                                   index= ltot)
                                  
    else:
              c = [None] * len(ltot)
              l = mod_grid.best_estimator_.coefs_[0]
              n_l = mod_grid.best_params_['hidden_layer_sizes'][0]
              for i in range(len(ltot)):
                c[i] = l[i][n_l-1]
              importance = c
              coefs = pd.DataFrame(c,
                                   columns=["Coefficients"],
                                   index= ltot)

    # plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    plt.savefig('Results-%s/Results-%s/%s/Plot/bar.png' %(back[pos], mlModel,targetN))
    plt.clf()
    plt.cla()
    plt.close()
    
########SHAP#########

    music_categories ={'Rock':0,'Classic':1,'Pop':2,'R&B':3,'Country':4,'Electronic':5,'Jazz':6,'Folk':7,'Indie':8,'Metal':9,'Rap':10}
    X['musicGenre']=X.musicGenre.apply(lambda x: music_categories[x])
    AItechnique_categories ={'searchStartegy':0,'planning':1,'constraints':2}
    X['AItechnique']=X.AItechnique.apply(lambda x: AItechnique_categories[x])
    
    model = models_regression[mlModel]['estimator']
    model.set_params(**mod_grid.best_params_)
    data_train1, data_test1, target_train1, target_test1 = train_test_split(
    X, y, test_size=0.2, random_state=42)
    model.fit(data_train1,target_train1)
    
    X100 = shap.utils.sample(X, 100)
    explainer = shap.Explainer(model.predict,X)
    shap_values = explainer(X)

    
    shap.plots.beeswarm(shap_values, show=False)
    plt.savefig('Results-%s/Results-%s/%s/Plot/shap.jpg' %(back[pos], mlModel,targetN), bbox_inches='tight')




################ WRITE RES IN A TXT #################################

    original_stdout = sys.stdout
    with open('Results-%s/Results-%s/%s/res.txt' %(back[pos], mlModel,targetN), 'w') as f:
        sys.stdout = f
        print('\n--------------------- Model errors and report:-------------------------')
        print('Mean Absolute Error:', metrics.mean_absolute_error(target_test, target_pred))
        print('Mean Squared Error:', metrics.mean_squared_error(target_test, target_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(target_test, target_pred)))
        print('Feauture Scores: \n')
        print(coefs)

        print('\nR2 score:' , metrics.r2_score(target_test, target_pred))

        print('\nBest Parameters used: ', mod_grid.best_params_)

    sys.stdout = original_stdout
    print('Results saved')


