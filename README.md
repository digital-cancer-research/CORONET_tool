![coronet_front_logo.PNG](/coronet_front_logo.PNG)

------

CORONET is an online tool to support decisions regarding hospital admissions or discharge in cancer patients presenting with symptoms of COVID-19 and the likely severity of illness. It is based on real world patient data.

The tool is available at:
https://coronet.manchester.ac.uk/

This repository contains code to develop a CORONET model used in the tool to generate recommendation.

Detailed description of the process of developing CORONET can be found in our publication:

*Establishment of CORONET; COVID-19 Risk in Oncology Evaluation Tool to identify cancer patients at low versus high risk of severe complications of COVID-19 infection upon presentation to hospital*

https://www.medrxiv.org/content/10.1101/2020.11.30.20239095v1

To learn more about how the score was calculated and to see a **global explanation** for the model and **local explanations** for individual patients in the training cohort, visit:
[CORONET_explain.ipynb](notebooks/CORONET_explain.ipynb)

The code used for the model development can be found here:
[CORONET_model_dev.ipynb](notebooks/CORONET_model_dev.ipynb)

The code on which the tool is based on, is available here:
[CORONET_code_for_tool.py](code/CORONET_code_for_tool.py)
