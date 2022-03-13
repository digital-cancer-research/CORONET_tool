
# 26/04/2021 
# Script uses functions from CORONET_functions to check if all of them run correctly
# 
# it should generate prediction and explanation for a patient
# script investigates individual functions - if they produce correct output


import sys
module_path = '...'
if module_path not in sys.path:
    sys.path.append(module_path)
import CORONET_functions as f
import pandas as pd
import importlib
importlib.reload(f)


model_path = '.../CORONET_model.pkl'
explained_path = '.../CORONET_explainer.pkl'


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

x_trans = f.transform_x_values(x)

print('\n\n input x: \n\n', x)
print('\n\n x transformed: \n\n', x_trans)


admission_threshold = 0.9
severe_condition_threshold = 2.3
output = f.get_prediction_for_x(x_trans, model, admission_threshold, severe_condition_threshold)
print('\n\nPrediction: \n\n',output)



explanation = f.get_shap_values_for_x(x_trans, explainer, sort_explanation=True)
print('\n\n Explanation (shap_values):\n\n',explanation)



f.plot_local_explanation_shap(explanation, x_trans, path_to_save_plots)


f.generate_colorbar(admission_threshold, severe_condition_threshold, path_to_save_plots)


df = pd.read_csv(r'C:\Users\d07321ow\Documents\GitHub\CORONET_dev\predictions_all.csv')


f.generate_plot_all_patients(df, path_to_save_plots)



print('DONE')

