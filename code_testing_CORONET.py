

# 26/04/2021 
# Script uses functions from CORONET_functions to check if all of them run correctly
# 
# it should generate prediction and explanation for a patient
# script tests the main function "predict_and_explain"


import sys
module_path = '...' # path to file with functions
if module_path not in sys.path:
    sys.path.append(module_path)
import CORONET_functions as f
import importlib
importlib.reload(f)


model_path = '.../CORONET_model.pkl' # path to file with Random Forest model
explained_path = '.../CORONET_explainer.pkl' # path to file with SHAP explainer


model = f.load_predictive_model(model_path)

explainer = f.load_predictive_model(explained_path)




# Define example patient

x = {'Age': 51,
     'Total no. comorbidities': 5,
     'Performance status':4,
     'NEWS2': 3,
     'Platelets': 30,
     'Albumin': 40,
     'CRP': 150,
     'Neutrophil': 7.7,
     'Lymphocyte': 0.5,
     }



path_to_save_plots = '...'


# # Using main function 'predict and explain'


prediction, explanation = f.predict_and_explain(x, model, explainer, plot_expl_barplot = True, path_to_save_plots=path_to_save_plots)

print('\n\nPREDICTION: \n', prediction)

print('\n\nEXPLANATION: \n', explanation)

print('DONE')

