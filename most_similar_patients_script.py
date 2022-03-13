# Wysocki 19/05/2021

import sys
import pandas as pd

module_path = r'C:\Users\d07321ow\Google Drive\Cytokine\COVID19\Cancer_calculator\CORONET_python'
if module_path not in sys.path:
    sys.path.append(module_path)
import CORONET_functions as f

import importlib
importlib.reload(f)


# define input patient data. Example:
x = {'Age': 75,
     'Total no. comorbidities': 1,
     'Performance status':2,
     'NEWS2': 4,
     'Platelets': 400,
     'Albumin': 49,
     'CRP': 25,
     'Neutrophil': 2,
     'Lymphocyte': 10,
     }

# define all paths
path_to_data_scaled = r'C:\Users\d07321ow\Google Drive\Cytokine\COVID19\Cancer_calculator\CORONET_KNN\data_standard_scaled.xlsx'
path_to_data_masked = r'C:\Users\d07321ow\Google Drive\Cytokine\COVID19\Cancer_calculator\CORONET_KNN\data_masked.xlsx'
path_to_scaler = r'C:\Users\d07321ow\Documents\GitHub\CORONET_dev\scaler.pkl'
path_to_exlainer = r'C:\Users\d07321ow\Documents\GitHub\CORONET_dev\CORONET_V2_explainer.pkl'

#----------------------------------------------------------------------------------------------------------------------------------

# load scaled dataset used in Ball query
df_scaled = pd.read_excel(path_to_data_scaled, index_col=0, engine='openpyxl')

# load masked data used to present similar patients' data to the user
df_masked = pd.read_excel(path_to_data_masked,index_col=0, engine='openpyxl')

# load explainer
explainer = f.load_predictive_model(path_to_exlainer)

# load scaler
scaler = f.load_scaler(path_to_scaler)


# get shap weights
x_trans = f.transform_x_values(x)
shap_weigths = f.get_local_shap_importances(x_trans, explainer)

# transform the whole X dataset according to shap weights
X_tranformed = f.transform_X_using_weigths(df_scaled, shap_weigths)

# get BallTree
kdt_BallTree = f.create_distance_BallTree(X_tranformed)


# transform x input patient according to shap weights
x_to_query = f.prepare_input_x_to_KNearest(x, scaler, shap_weigths)


# find K nearest
df_nearest = f.find_K_nearest(df_masked, x_to_query, kdt_BallTree, k = 5)


# final table to present to the user
df_to_show = f.prepate_df_to_show(x_trans, df_nearest)
#----------------------------------------------------------------------------------------------------------------------------------

print(df_to_show)

