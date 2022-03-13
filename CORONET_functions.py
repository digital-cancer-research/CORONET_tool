import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

"""
List of functions:

predict_and_explain - main function that generates prediction and explanation


load_predictive_model
load_predictive_model_using_pickle
load_explainer

transform_x_values
get_prediction_for_x
get_shap_values_for_x

plot_local_explanation_shap
generate_colorbar
generate_plot_all_patients
calculate_NEWS2

"""

def predict_and_explain(x, model, explainer, plot_expl_barplot = True, path_to_save_plots=''):
    """
    Predicts and explains the prediction using pre-trained model and explainer.
    It outputs two dictionaries: 1) prediction, 2) explanation
    Optionally, it can plot and save a barplot with shap values explaining the prediction.

    Before running the function, the model and the explainer should be loaded.

    Contains admission_threshold and severe_condition_threshold which are specified inside this function.

    Parameters
    ----------
    x : dict
    A dictionary with keys = features, values = patient's parameters values.
    Dictionary format:

    {'NEWS2': value,
    'CRP': value,
    'Albumin': value,
    'Age': value,
    'Platelets': value,
    'Neutrophil': value,
    'Lymphocyte': value,
    'Performance status: value',
    'Total no. comorbidities': value
    }


    model : sklearn predictive model

    explainer : shap.TreeExplainer object

    plot_expl_barplot : bool - default True, if False the function does not generate the barplot with the explanation

    path_to_save_plots : str - directory to save png file with the figure

    Returns
    ------
    prediction : dict
    a dictionary with keys: 'Predicted_score' and 'Recommendation'

    explanation : dict
    a dictionary with shap values for each feature sorted by absolute value


    """
    admission_threshold = 1.0
    severe_condition_threshold = 2.3

    x_trans = transform_x_values(x)

    prediction = get_prediction_for_x(x_trans, model, admission_threshold, severe_condition_threshold)

    explanation = get_shap_values_for_x(x_trans, explainer, sort_explanation=True)

    if plot_expl_barplot:
        plot_local_explanation_shap(explanation, x_trans, path_to_save_plots)

    return prediction, explanation


def load_predictive_model(file_path):
    """
     Loads predictive model stored in a .pkl file from 'file_path'
     The model is a Random Forest model trained using sklearn library and saved to .pkl file using joblib library (using joblib.dump command)
     required libraries:
      joblib

    :param file_path:
    :return:
    -----------
    model
    """

    model = joblib.load(file_path)

    return model


def load_predictive_model_using_pickle(file_path):
    """
     Loads predictive model stored in a .pkl file from 'file_path'
     The model is a Random Forest model trained using sklearn library and saved to .pkl or to .sav file using pickle library
        Commands used for saving the model:
            pickle_file = ...\CORONET_model\RF_model_pickle.pkl'
            pickle_file = ...\CORONET_model\RF_model_pickle.sav'
            pickle.dump(coronet_RF_model, open(pickle_file, 'wb'))

      (using pickle.dumps command)
     required libraries:
      pickle

    :param file_path:
    :return:
    -----------
    model
    """

    model = pickle.load(open(file_path, 'rb'))

    return model


def load_explainer(file_path):
    """
     Loads explainer stored in a .pkl file from 'file_path'
    The explainer is a TreeExplainer from SHAP library (https://github.com/slundberg/shap)
    created using function shap.TreeExplainer(model), where 'model' is the predictive model used in CORONET
    and saved to .pkl file using joblib library (using joblib.dump command)
    required libraries:
       joblib

    :param file_path:
    :return:
    """


    explainer = joblib.load(file_path)

    return explainer

def load_explainer_using_pickle(file_path):
    """
     Loads explainer stored in a .pkl file from 'file_path'
    The explainer is a TreeExplainer from SHAP library (https://github.com/slundberg/shap)
    created using function shap.TreeExplainer(model), where 'model' is the predictive model used in CORONET
    and saved to .pkl file using pickle library
    required libraries:
       pickle

    :param file_path:
    :return:
    """


    explainer = pickle.load(open(file_path, 'rb'))

    return explainer


def transform_x_values(x):
    """
    Calculates NLR (Neutrophil:Lymphocyte Ratio)
    If Lymphocyte < 0.1, then Lymhpcyte is set to 0.1 before calculating the NLR

    Parameters:
    ----------
    param x : dict
    x : dict
    A dictionary with keys = features, values = patient's parameters values.
    Dictionary format:

    {'NEWS2': value,
    'CRP': value,
    'Albumin': value,
    'Age': value,
    'Platelets': value,
    'Neutrophil': value,
    'Lymphocyte': value,
    'Performance status: value',
    'Total no. comorbidities': value
    }

    Return:
    ------
    x_transformed : dict
    A dictionary with keys = features, values = patient's parameters values.
    Dictionary format:

    {'NEWS2': value,
    'CRP': value,
    'Albumin': value,
    'Age': value,
    'Platelets': value,
    'Neutrophil': value,
    'Lymphocyte': value,
    'Performance status: value',
    'Total no. comorbidities': value
     'NLR': value}
    """
    x_transformed = x.copy()


    if x_transformed['Lymphocyte'] < 0.1:
        x_transformed['Lymphocyte'] = 0.1

    x_transformed['NLR'] = x_transformed['Neutrophil']/x_transformed['Lymphocyte']


    # Reorder dict after adding NLR - this is
    desired_order_list = ['NEWS2',
                          'CRP',
                          'Albumin',
                          'Age',
                          'Platelets',
                          'Neutrophil',
                          'Performance status',
                          'Lymphocyte',
                          'Total no. comorbidities',
                          'NLR']

    x_transformed = {k: x_transformed[k] for k in desired_order_list}

    return x_transformed


def get_prediction_for_x(x, model, admission_threshold, severe_condition_threshold):
    """
    Calculates the score and assigns recommendation based on given thresholds.
    Uses transformed x (with calculated NLR) and predictive model and calculates the score (range 0.0-3.0),
    It also outputs a string with a recommendation from the list of three:
    - 'consider discharge'
    - 'consider admission'
    - 'high risk of severe condition'


    Parameters:
    -----------
    x : dict
    A dictionary with keys = features, values = patient's parameters values.
    Dictionary format:

    {'NEWS2': value,
    'CRP': value,
    'Albumin': value,
    'Age': value,
    'Platelets': value,
    'Neutrophil': value,
    'Lymphocyte': value,
    'Performance status: value',
    'Total no. comorbidities': value
     'NLR': value}


    model : sklearn predictive model

    admission_threshold : float
    A threshold defined by the researcher.
    Above this value all recommendation will be 'consider admission' or 'high risk of severe condition'.
    Below this value all recommendation will be 'consider discharge'.

    severe_condition_threshold : float
    A threshold defined by the researcher.
    Above this value all recommendation will be 'high risk of severe condition'.
    Below this value all recommendation will be 'consider discharge' or 'consider admission'.

    Return
    ------
    prediction : dict
    a dictionary with predicted score (str, the score rounded to 2 decimals) and textual recommendation (str).
    Dictionary format (example values):
    {'Predicted_score': '0.95',
     'Recommendation': 'consider discharge'}

    """

    x_to_model = np.array(list(x.values())).reshape(1, -1)

    predicted_score = np.round(model.predict(x_to_model)[0], 2)


    recommendations = ['consider discharge', 'consider admission', 'high risk of severe condition']

    if predicted_score < admission_threshold:
        recommendation = recommendations[0]
    elif predicted_score > severe_condition_threshold:
        recommendation = recommendations[2]
    else:
        recommendation = recommendations[1]

    # convert to string with 2 decimal places (for consitency in showing the coronet score to the user)
    predicted_score = f'{predicted_score:.2f}'

    prediction = {'Predicted_score': predicted_score, 'Recommendation': recommendation}

    return prediction


def get_shap_values_for_x(x, explainer, sort_explanation=True):
    """
    Computes shapley values of local explanation for 'x'.
    Uses 'explainer' which is an explainer object from shap library.
    Generated 'explanation' can sorted (default) or in the same order as 'x'.

    Parameters:
    ----------
    x : dict
    A dictionary with keys = features, values = patient's parameters values. CRP and NLR values should be transformed.
    Dictionary format:

    {'NEWS2': value,
    'CRP': value,
    'Albumin': value,
    'Age': value,
    'Platelets': value,
    'Neutrophil': value,
    'Lymphocyte': value,
    'Performance status: value',
    'Total no. comorbidities': value
     'NLR': value}

    explainer : shap.Explainer object

    sort_explanation : bool
    default True, if False the keys of explanation dict will be in the same order as x.
    If True, the explanation dict will be sorted by absolute value of shap value (the highest - most important - are at the bottom)

    Return:
    -------
    explanation : dict
    a dictionary with shap values for each feature sorted by absolute value (sorting is optional but default True)
    Dictionary format:
    {'NEWS2': shap_value,
    'CRP': shap_value,
    'Albumin': shap_value,
    'Age': shap_value,
    'Platelets': shap_value,
    'Neutrophil': shap_value,
    'Lymphocyte': shap_value,
    'Performance status: shap_value',
    'Total no. comorbidities': shap_value
     'NLR': shap_value}


    """
    x_to_model = np.array(list(x.values()))

    features = list(x.keys())

    shap_values = np.round(explainer.shap_values(x_to_model), 4)

    explanation = {}

    for i, feature in enumerate(features):
        explanation[feature] = shap_values[i]

    if sort_explanation:
        explanation = {k: v for k, v in sorted(explanation.items(), key=lambda item: np.abs(item[1]), reverse=False)}

    return explanation


def plot_local_explanation_shap(shap_dict, x, path_to_save, example_no=0):
    """
    Plot a red-green barplot showing the contribution of each feature to the prediction. 
    The contribution is equal to shap value for given feature. 
    Negative shap values contribute to the 'consider discharge' recommendation and are represented as green bars on the left side of the plot.
    Positive shap values contribute to the 'consider admission' or 'high risk of severe condition' recommendation and are represented as red bars on the right side of the plot.
    
    Next to the bars a value of given parameter is shown in a textbox.
    
    Important: bar width corresponds to the shap value, not to the parameter value displayed in the textbox.

    Saves the figure as 'local_explanation_shap.png'

    Parameters:
    -----------
    shap_dict : : dict
    a dictionary with shap values for each feature sorted by absolute value (sorting is optional but default True)
    Dictionary format:
    {'NEWS2': shap_value,
    'CRP': shap_value,
    'Albumin': shap_value,
    'Age': shap_value,
    'Platelets': shap_value,
    'Neutrophil': shap_value,
    'Lymphocyte': shap_value,
    'Performance status: shap_value',
    'Total no. comorbidities': shap_value
     'NLR': shap_value}
    
     
    x : dict  
    A dictionary with keys = features, values = patient's parameters values.
    Dictionary format:

    {'NEWS2': value,
    'CRP': value,
    'Albumin': value,
    'Age': value,
    'Platelets': value,
    'Neutrophil': value,
    'Lymphocyte': value,
    'Performance status: value',
    'Total no. comorbidities': value
     'NLR': value}
     
     
    path_to_save : str
     Directory where the png file with figure should be saved.

    Returns:
    -------
    

    """


    # sort shap_values dictionary by absolute value
    shap_dict_sorted = shap_dict#{k: v for k, v in sorted(shap_dict.items(), key=lambda item: np.abs(item[1]), reverse=False)}

    fig, ax = plt.subplots(figsize=(13, 6))

    features = list(shap_dict_sorted.keys())

    values = list(shap_dict_sorted.values())

    # plot barplot
    bars = ax.barh(width=values, y=features, linewidth=1, edgecolor='black')

    # assing bar colors (red for features voting for 'admission', green for features voting for 'discharge')
    for j, bar in enumerate(bars):
        if values[j] < 0:
            bar.set_color('green')
        else:
            bar.set_color('red')
        bar.set_edgecolor('k')

    ax.set_xticklabels([None])

    # add arrows at the top
    props = dict(boxstyle="larrow,pad=0.3", facecolor='white', alpha=1)
    text = 'DISCHARGE'
    ax.text(0.45, 1.05, text, bbox=props, transform=ax.transAxes, va='bottom', ha='right', fontsize=15)
    props = dict(boxstyle="rarrow,pad=0.3", facecolor='white', alpha=1)
    text = 'ADMISSION'
    ax.text(0.55, 1.05, text, bbox=props, transform=ax.transAxes, va='bottom', ha='left', fontsize=15)
    ax.set_yticklabels([None])

    # add bars description (feature name and its value, i.e. real value, not the shap value)
    for m in range(len(values)):
        parameter = features[m]
        shap_value = values[m]

        if parameter == 'NLR':
            text = 'Neutrophil:Lymphocyte Ratio' + ' = ' + str(np.round(x['NLR'], 1))

        elif parameter == 'CRP':
            # RF model uses transformed CRP, but to show the CRP value on the plot, we need to refer to initial 'x' instead of 'x_transformed'
            unit = 'mg/L'
            text = 'C-reactive protein' + ' (' + unit + ') = ' + str(np.round(x['CRP'], 1))
        elif parameter == 'Albumin':
            unit = 'g/L'
            text = parameter + ' (' + unit + ') = ' + str(np.round(x[parameter], 0))
        elif parameter == 'Lymphocyte':
            unit = 'x10^9/L'
            text = parameter + ' (' + unit + ') = ' + str(np.round(x[parameter], 1))
        elif parameter == 'Neutrophil':
            unit = 'x10^9/L'
            text = parameter + ' (' + unit + ') = ' + str(np.round(x[parameter], 1))
        elif parameter == 'Platelets':
            unit = 'x10^9/L'
            text = parameter + ' (' + unit + ') = ' + str(np.round(x[parameter], 0))
        else:
            text = parameter + ' = ' + str(np.int(np.round(x[parameter], 0)))

        if shap_value > 0:
            ha = 'left'
        else:
            ha = 'right'

        ax.text(shap_value + 0.008 * np.sign(shap_value), m, text, ha=ha, va='center', fontsize=18)

    #ax.set_xlim([-np.abs(values).max() - 0.1, np.abs(values).max() + .1])
    ax.set_xlim([-.65, .65])

    # remove axes lines
    ax.set_frame_on(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)

    plt.subplots_adjust(top=1.1)
    plt.tight_layout()
    #path_to_save = os.path.join(path_to_save, 'local_explanation_shap.png')
    path_to_save = os.path.join(path_to_save, 'local_explanation_shap_example_{}.png'.format(example_no))
    plt.savefig(path_to_save, dpi=400)


def generate_colorbar(admission_threshold, severe_condition_threshold, path_to_save):
    """
    Generates and saves a colorbar with a gradient colors green-->yellow-->red-->black.
    The color transition values are defined by 'admission_threshold 'severe_condition_threshold:
    - the center of the yellow field is defined by  'admission threshold' value
    - the center of the red field is defined by  'severe_condition_threshold' value

    Saves the figure as 'colorbar_03score.png'

    Parameters:
    -----------
    admission_threshold : float
    Defined by the researcher.

    severe_condition_threshold : float
    Defined by the researcher.

    path_to_save : str
    Directory to save the figure with the colorbar

    Return
    ------
    """


    thresh1 = admission_threshold / 3
    thresh2 = severe_condition_threshold / 3

    nodes = [0, thresh1, thresh2, 1.0]

    colors = ["green", "yellow", "red", "black"]

    cmap = LinearSegmentedColormap.from_list("", list(zip(nodes, colors)))

    gradient = np.linspace(0, 3, 300).reshape(1, -1)

    fig, ax = plt.subplots(figsize=(15, 3))
    ax.imshow(gradient, extent=[-0.0, 3, -1, 1], aspect='auto', cmap=cmap)  # 'RdYlGn_r')
    ax.set_yticks([])

    ax.tick_params(axis='both', which='major', size=15)
    plt.xticks(fontsize=14)
    plt.tight_layout()

    path_to_save = os.path.join(path_to_save, 'colorbar_03score.png')
    plt.savefig(path_to_save + '', dpi=300)


def generate_plot_all_patients(df, path_to_save):
    """
    Generates a dot plot (i.e. swarmplot from seaborn library) for all the patients from the training set.
    The score for each patient is calculated according to LOOCV - model trained on all samples except one.
    Each dot represents a score predicted by the model for given patient. Dots are colored by the true outcome.
    It serves as an explanation 'where my patient is in the whole cohort in terms of predicted score'

    Saves the figure as 'plot_all_patients.png'
    Saves the figure as 'plot_all_patients_separated.png'

    Parameters:
    -----------

    df : DataFrame
    Dataframe with 3 columns:
    index,  y_pred, y_true, constant
    0,      0.633,      0,     1
    1,      0.635,      0,     1
    2,      0.651,      0,     1
    3,      0.652,      0,     1
    ...     ...         ...    ...

    path_to_save : str

    Returns
    -------


    """
    colors = ['green', 'gold', 'red', 'black']

    sns.set_palette(sns.color_palette(colors))

    fig, ax = plt.subplots(figsize=(13, 6))
    g = sns.swarmplot(x='y_pred', y='constant', hue='y_true', data=df, ax=ax, zorder=1, orient='h', size=7)
    handles, labels = g.get_legend_handles_labels()
    g.legend(handles, ['Discharged', 'Admitted', 'Required O2', 'Death due to COVID'], title='Outcome', fontsize=12,
             framealpha=1, bbox_to_anchor=(1.01, 0.5), loc=6, borderaxespad=0., edgecolor='k')

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([1, 2, 3])
    ax.set_ylabel(None)
    ax.set_xlabel(None)

    ax.set_yticks([])
    ax.set_xlim([0.1, 3])
    plt.tight_layout()
    path_to_save_joined = os.path.join(path_to_save, 'plot_all_patients.png')
    plt.savefig(path_to_save_joined, dpi=300)


    fig, ax = plt.subplots(figsize=(13, 7))
    g = sns.swarmplot(x='y_pred', y='y_true', hue='y_true', data=df, ax=ax, zorder=1, orient='h', size=7)
    handles, labels = g.get_legend_handles_labels()
    g.legend(handles, ['Discharged', 'Admitted', 'Required O2', 'Death due to COVID'], title='Outcome',
             fontsize=12,
             framealpha=1, bbox_to_anchor=(1.01, 0.5), loc=6, borderaxespad=0., edgecolor='k')

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([1, 2, 3])
    ax.set_ylabel(None)
    ax.set_xlabel(None)

    ax.set_yticks([])
    ax.set_xlim([0.1, 3])
    plt.tight_layout()
    path_to_save_separated = os.path.join(path_to_save, 'plot_all_patients_separated.png')
    plt.savefig(path_to_save_separated, dpi=300)


def calculate_NEWS2(x):
    '''
    Function to calculate NEWS2 score

    https://www.mdcalc.com/national-early-warning-score-news-2

    INPUT:
    dictionary
    x = {'Respiratory Rate (bpm)': int,
        'Hypercapnic respiratory failure': str, -string 'Yes' or 'No'
        'SpO2 (%)': int,
        'Supplemental O2': str, -string 'Yes' or 'No'
        'Systolic BP (mmHg)': int,
        'Heart Rate (bpm)':int
         'Consciousness': 'str,   -string 'Yes' or 'No'
         'Temperature (degrees of C)': float, - decimal

         }


    OUTPUT:

    NEWS2_score - int
    '''

    rr_score = 0
    sat_score = 0
    supp_o2_score = 0
    Systolic_BP_score = 0
    Heart_rate_score = 0
    Consciousness_score = 0
    Temperature_score = 0


    # Respiratory Rate (bpm)
    if (11 >= x['Respiratory Rate (bpm)']) & (x['Respiratory Rate (bpm)'] >= 9):
        rr_score = 1
    elif x['Respiratory Rate (bpm)'] <= 8:
        rr_score = 3
    elif (24 >= x['Respiratory Rate (bpm)']) & (x['Respiratory Rate (bpm)'] >= 21):
        rr_score = 2
    elif x['Respiratory Rate (bpm)'] >= 25:
        rr_score = 3

    # SpO2
    if x['Hypercapnic respiratory failure'] == 'No':
        # SpO2 (%)
        if x['SpO2 (%)'] <= 91:
            sat_score = 3
        elif (92 <= x['SpO2 (%)']) & (x['SpO2 (%)'] <= 93):
            sat_score = 2
        elif (94 <= x['SpO2 (%)']) & (x['SpO2 (%)'] <= 95):
            sat_score = 1
    elif x['Hypercapnic respiratory failure'] == 'Yes':
        # SpO2 (%)
        if x['SpO2 (%)'] <= 83:
            sat_score = 3
        elif (84 <= x['SpO2 (%)']) & (x['SpO2 (%)'] <= 85):
            sat_score = 2
        elif (86 <= x['SpO2 (%)']) & (x['SpO2 (%)'] <= 87):
            sat_score = 1
        elif (93 <= x['SpO2 (%)']) & (x['SpO2 (%)'] <= 94):
            sat_score = 1
        elif (95<= x['SpO2 (%)']) & (x['SpO2 (%)'] <= 96):
            sat_score = 2
        elif x['SpO2 (%)'] >= 97:
            sat_score = 3


    # Supplemental O2
    if x['Supplemental O2'] == 'Yes':
        supp_o2_score = 2

    # Systolic BP (mmHg)
    if x['Systolic BP (mmHg)'] <= 90:
        Systolic_BP_score = 3
    elif (91 <= x['Systolic BP (mmHg)']) & (x['Systolic BP (mmHg)'] <= 100):
        Systolic_BP_score = 2
    elif (101 <= x['Systolic BP (mmHg)']) & (x['Systolic BP (mmHg)'] <= 110):
        Systolic_BP_score = 1
    elif x['Systolic BP (mmHg)'] >= 220:
        Systolic_BP_score = 3

    # Consciousness
    if x['Consciousness'] == 'No':
        Consciousness_score = 3

    # Temperature
    if x['Temperature (degrees of C)'] <= 35:
        Temperature_score = 3
    elif (35.1 <= x['Temperature (degrees of C)']) & (x['Temperature (degrees of C)'] <= 36):
        Temperature_score = 1
    elif (38.1 <= x['Temperature (degrees of C)']) & (x['Temperature (degrees of C)'] <= 39):
        Temperature_score = 1
    elif x['Temperature (degrees of C)'] >= 39.1:
        Temperature_score = 2

    # Temperature
    if x['Heart Rate (bpm)'] <= 40:
        Heart_rate_score = 3
    elif (41 <= x['Heart Rate (bpm)']) & (x['Heart Rate (bpm)'] <= 50):
        Heart_rate_score = 1
    elif (91 <= x['Heart Rate (bpm)']) & (x['Heart Rate (bpm)'] <= 110):
        Heart_rate_score = 1
    elif (111 <= x['Heart Rate (bpm)']) & (x['Heart Rate (bpm)'] <= 130):
        Heart_rate_score = 2
    elif x['Heart Rate (bpm)'] >= 131:
        Heart_rate_score = 3

    NEWS2_score = rr_score + sat_score + supp_o2_score + Systolic_BP_score + Consciousness_score + Temperature_score + Heart_rate_score
    #print(rr_score, sat_score, Systolic_BP_score, Consciousness_score, Temperature_score, Heart_rate_score)

    return NEWS2_score

#------------------------------------------------------------------------------------------------------------------------

# Functions for finding N nearest patients from the dataset X based on input x
def load_scaler(file_path):
    """
     Loads StandardScaler stored in a .pkl file from 'file_path'

     required libraries:
      joblib

    :param:
    file_path:

    :return:
    -----------
    scaler
    """

    scaler = joblib.load(file_path)

    return scaler

def prepare_input_x_to_KNearest(x, scaler, shap_weights, cols_from_shap_weights=None):
    '''

    :param:
    x: dict

    scaler: scaler object from sklearn StandardScaler

    cols_from_shap_weights: optional, list of strings; used when defining custom search in the future

    shap_weigths: DataFrame, with absolute shap values normalized (all values divided by max)

    :return:
    x_to_query: dict, with scaled values using scaler and multiplied by shap weights
    '''
    cols_in_scaler = ['NEWS2',
                     'CRP',
                     'Albumin',
                     'Age',
                     'Platelets',
                     'Neutrophil_log',
                     'Lymphocyte_log',
                     'Performance status',
                     'Total no. comorbidities',
                     'NLR_log'
                     ]

    if cols_from_shap_weights==None:
        cols_from_shap_weights =  cols_in_scaler
    else:
        pass

    x_trans = transform_x_values(x)

    x_trans['Neutrophil_log'] = np.log(x_trans['Neutrophil'] + 1)
    x_trans['Lymphocyte_log'] = np.log(x_trans['Lymphocyte'] + 1)
    x_trans['NLR_log'] = np.log(x_trans['NLR'] + 0.01)

    x_trans_df = pd.DataFrame.from_dict([x_trans])[cols_in_scaler]

    # scale x
    x_to_query = scaler.transform(x_trans_df.values)
    x_to_query = pd.DataFrame(x_to_query, columns = cols_in_scaler)[cols_from_shap_weights]

    # apply shap weights to scaled x
    x_to_query = x_to_query*shap_weights
    x_to_query =x_to_query[shap_weights.columns]

    return x_to_query


def get_local_shap_importances(x_trans, explainer, cols_to_KNN=None):
    """

    :param:
    x_trans: dict, patient data already transformed by transform_x_values function

    explainer: shap explainer object

    cols_to_KNN: list of strings, optional

    :return:
    shap_weigths: DataFrame with absolute shap values normalized (all values divided by max)
    """
    if cols_to_KNN==None:
        cols_to_KNN =  ['NEWS2',
                         'CRP',
                         'Albumin',
                         'Age',
                         'Platelets',
                         'Neutrophil',
                         'Lymphocyte',
                        'Performance status',
                        'Total no. comorbidities',
                        'NLR'
                        ]
    else:
        pass

    shap_weigths = get_shap_values_for_x(x_trans, explainer, sort_explanation=False)
    shap_weigths = pd.DataFrame.from_dict([shap_weigths])
    shap_weigths = shap_weigths.abs()

    shap_weigths = shap_weigths[cols_to_KNN]
    shap_weigths = shap_weigths / shap_weigths.max(axis=1).values[0]

    if 'Neutrophil' in shap_weigths.columns:
        shap_weigths['Neutrophil_log'] = shap_weigths['Neutrophil']
        shap_weigths = shap_weigths.drop(columns='Neutrophil')

    if 'Lymphocyte' in shap_weigths.columns:
        shap_weigths['Lymphocyte_log'] = shap_weigths['Lymphocyte']
        shap_weigths = shap_weigths.drop(columns = 'Lymphocyte')

    if 'NLR' in shap_weigths.columns:
        shap_weigths['NLR_log'] = shap_weigths['NLR']
        shap_weigths = shap_weigths.drop(columns = 'NLR')

    return shap_weigths


def transform_X_using_weigths(df_scaled, shap_weigths):
    """

    :param:
    df_scaled: DataFrame, dataset with scaled values of 10 parameters

    shap_weigths: DataFrame, with absolute shap values normalized (all values divided by max)

    :return:
    X_weigthed: DataFrame, dataset with scaled values multiplied by shap weights
    """
    cols_to_KNN = shap_weigths.columns
    X = df_scaled[cols_to_KNN].values

    X_weigthed = X * shap_weigths.values

    X_weigthed = pd.DataFrame(X_weigthed, columns=cols_to_KNN)

    return X_weigthed

from sklearn.neighbors import BallTree
def create_distance_BallTree(X):
    kdt = BallTree(X, leaf_size=30, metric='euclidean')

    return kdt


def find_K_nearest(df_masked, x, kdt_BallTree, k=5, cols_to_show=None, index_filtered = None):
    """

    :param:

    df_masked: DataFrame, dataset with masked patient data used to present details to the user

    x: dict, your patient data, prepared by using function 'prepare_input_x_to_KNearest'

    kdt_BallTree: BallTree object

    k: int, number of similar patients to look for

    cols_to_show: list of strings, optional

    index_filtered: optional, for the future...

    :return:
    df_k_nearest: DataFrame, with k columns, rows represent patients' parameters
    """

    k_nearest_index = kdt_BallTree.query(x, k=100, return_distance=False)[0]

    if cols_to_show is None:
        cols_to_show = ['Outcome',
                'Admitted/Discharged',
                'Required O2',
                'Death due to COVID19',
                'CORONET score',
                        'CORONET recommendation',
                'Biological Sex','Age',
                'NEWS2','CRP','Albumin', 'Platelets',
                'Lymphocyte', 'Neutrophil', 'NLR',  'LDH',
                'Consciousness',
       'Respiratory Rate',
       'Oxygen saturation (SAT)',
       'Chemotherapy', 'Immunotherapy', 'Targeted therapy', 'Radiotherapy',
       'Total no. comorbidities', 'Performance status',
       'Treatment intent',
       'Early/advanced stage', 'Cancer type', 'Solid cancer stage']

    #k_nearest_index = list(set.difference(set(np.where(index_filtered==True)[0]), set(k_nearest_index)))[:k]
    if index_filtered is None:
        k_nearest_index = k_nearest_index[:k]
    elif index_filtered.sum()>0:
        index_filtered=list(np.where(index_filtered==True)[0])
        k_nearest_index = [i for i in k_nearest_index if i in index_filtered][:k]

    df_k_nearest = df_masked.loc[k_nearest_index, cols_to_show]
    df_k_nearest = df_k_nearest.T

    return df_k_nearest


def prepate_df_to_show(x_trans, df_nearest):
    """

    :param:
    x_trans:

         dict, patient data already transformed by transform_x_values function
    df_nearest:
        DataFrame, with k columns, rows represent patients' parameters

    :return:
    df_to_show: DataFrame, x_trans and df_nearest concatenated and with ordered index
    """
    x_to_nearest = pd.DataFrame.from_dict([x_trans]).T
    x_to_nearest = x_to_nearest.rename(columns={0: 'Your patient'})

    df_to_show = pd.concat((x_to_nearest, df_nearest), axis=1)
    df_to_show = df_to_show.fillna('-')
    df_to_show = df_to_show.loc[df_nearest.index, :]
    df_to_show.columns = ['Your patient','1st','2nd','3rd','4th','5th']


    return df_to_show

def mask_patients_in_df(df):
    df_masked = df.copy()

    # mask age, e.g.: 65 to '60s'
    df_masked['Age'] = ((df_masked['Age'] / 10).apply(np.floor) * 10).astype('int').astype('str') + 's'
    df_masked.loc[df_masked['Age']=='0s', 'Age'] = '<10'
    #
    crp_list = [list(np.arange(0, 10, 2)),list(np.arange(10, 100, 5)), list(np.arange(100, 1200, 20))]
    crp_list = [item for sublist in crp_list for item in sublist]
    df_masked['CRP'] = group_parameter(df['CRP'], groups_ranges=crp_list)

    df_masked['Platelets'] = group_parameter(df['Platelets'], groups_ranges=list(np.arange(0, 700, 10)))
    df_masked['Albumin'] = group_parameter(df['Albumin'], groups_ranges=list(np.arange(0, 120, 5)))

    neutro_list = [list(np.arange(0, 5, 0.5)),list(np.arange(5, 20, 1)), list(np.arange(20, 150, 2))]
    neutro_list = [float(item) for sublist in neutro_list for item in sublist]
    df_masked['Neutrophil'] = group_parameter(df['Neutrophil'], groups_ranges=neutro_list)

    lympho_list = [list(np.arange(0, 5, 0.5)),list(np.arange(5, 20, 1)), list(np.arange(20, 102, 2))]
    lympho_list = [item for sublist in lympho_list for item in sublist]
    df_masked['Lymphocyte'] = group_parameter(df['Lymphocyte'], groups_ranges=lympho_list)

    df_masked['NLR'] = group_parameter(df['NLR'], groups_ranges=list(np.arange(0, 120, 2)))

    df_masked['LDH'] = group_parameter(df['LDH'], groups_ranges=list(np.arange(0, 5000, 25)))

    df_masked['respiratory rate'] = group_parameter(df['respiratory rate'], groups_ranges=list(np.arange(0, 120, 2)))
    df_masked['SAT'] = group_parameter(df['SAT'], groups_ranges=list(np.arange(70, 106, 2)))

    yes_no_columns = ['Chemotherapy', 'Immunotherapy', 'Targetted_therapy', 'Radiotherapy','haematological_cancer']

    for col in yes_no_columns:
        df_masked.loc[df_masked[col] == 0, col] = 'No'
        df_masked.loc[df_masked[col] == 1, col] = 'Yes'

    df_masked.loc[df_masked['Treatment_intent'] == 1, 'Treatment_intent'] = 'Curative'
    df_masked.loc[df_masked['Treatment_intent'] == 2, 'Treatment_intent'] = 'Palliative'

    df_masked['Outcome'] = ''
    df_masked.loc[df_masked['coronet_true_score'] == 0, 'Outcome'] = 'Discharged'
    df_masked.loc[df_masked['coronet_true_score'] == 1, 'Outcome'] = 'Admitted'
    df_masked.loc[df_masked['coronet_true_score'] == 2, 'Outcome'] = 'Admitted+O2'
    df_masked.loc[df_masked['coronet_true_score'] == 3, 'Outcome'] = 'Admitted+O2+died'

    df_masked.loc[df_masked['Total_nb_comorbidities']>4, 'Total_nb_comorbidities'] = '>4'

    df_masked['Cancer type'] = '-'
    df_masked.loc[df_masked['Cancer_stage'].isin([1,2,3,4]), 'Cancer type'] = 'Solid'
    df_masked.loc[df_masked['haematological_cancer']=='Yes', 'Cancer type'] = 'Haematological'

    df_masked.loc[df_masked['Cancer type'] == 'Haematological', 'Cancer_stage'] = '-'
    df_masked.loc[df_masked['Cancer_stage']==0, 'Cancer_stage'] = '-'


    df_masked.loc[df_masked['GCS_below15'] == 0, 'GCS_below15'] = 'fully conscious'
    df_masked.loc[df_masked['GCS_below15'] == 1, 'GCS_below15'] = 'GCS<15'


    # rename columns
    df_masked= df_masked.rename(columns = {'Sex':'Biological Sex',
                                'Targetted_therapy':'Targeted therapy',
                                'Treatment_intent':'Treatment intent',
                                'Total_nb_comorbidities':'Total no. comorbidities',
                                'Early_advanced_stage':'Early/advanced stage',
                                'Death_day_since_admission':'Death day since admission',
                                'respiratory rate': 'Respiratory Rate',
                                'SAT':'Oxygen saturation (SAT)',
                                'Cancer_stage':'Solid cancer stage',
                                'haematological_cancer':'Haematological cancer',
                                '4C':'4C score',
                                           'GCS_below15':'Consciousness',
                                           'Performance_status':'Performance status'
                                })

    df_masked['Admitted/Discharged'] = 'Discharged'
    df_masked['Required O2'] = 'No'
    df_masked['Death due to COVID19'] = 'No'

    df_masked.loc[df_masked['coronet_true_score'] >0 , 'Admitted/Discharged'] = 'Admitted'
    df_masked.loc[df_masked['coronet_true_score'] >=2 ,'Required O2'] = 'Yes'
    df_masked.loc[df_masked['coronet_true_score'] ==3 ,'Death due to COVID19'] = 'Yes'

    df_masked = df_masked.fillna('-')

    return df_masked






def group_parameter(series, groups_ranges=[0, 5, 15, 50, 150, 1000]):
    series_of_labels = series.copy()
    label = np.nan
    for i in range(len(groups_ranges) - 1):

        range_start = groups_ranges[i]
        range_end = groups_ranges[i + 1]

        index_ = (series >= range_start) & (series < range_end)

        if i == 0:
            label = '<' + str(range_end)
        elif i == (len(groups_ranges) - 2):
            label = '>' + str(range_start)
        else:
            label = str(range_start) + '-' + str(range_end)

        series_of_labels[index_] = label

    return series_of_labels


def filter_cancer_type(df, haem_cancer=True, solid_cancer=True, missing_cancer=True):

    if haem_cancer:
        index_haem = df['Cancer type'] == 'Haematological'
    else:
        index_haem = df['Cancer type'].astype('bool') * False

    if solid_cancer:
        index_solid = df['Cancer type'] == 'Solid'
    else:
        index_solid = df['Cancer type'].astype('bool') * False

    if missing_cancer:
        index_missing = df['Cancer type'] == '-'
    else:
        index_missing = df['Cancer type'].astype('bool') * False

    index_ = index_haem | index_solid | index_missing

    return index_














