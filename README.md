![coronet_front_logo.PNG](/coronet_front_logo.PNG)

------

CORONET is an online tool to support decisions regarding hospital admissions or discharge in cancer patients presenting with symptoms of COVID-19 and the likely severity of illness. It is based on real world patient data.

The tool is available at:
[CORONET_tool](https://coronet.manchester.ac.uk/)

This repository contains code to run a CORONET model used in the tool to generate recommendation:
- [CORONET_functions.py](https://github.com/digital-ECMT/CORONET_tool/blob/main/CORONET_functions.py) - all functions for data processing, prediction, explanation and finding similar patients in the dataset
- [code_testing_CORONET.py](https://github.com/digital-ECMT/CORONET_tool/blob/main/code_testing_CORONET.py) - a script to test the main function 'predict_and_explain'
- [code_testing_CORONET_functions.py](https://github.com/digital-ECMT/CORONET_tool/blob/main/code_testing_CORONET_functions.py) - a script to test individual functions
- [most_similar_patients_script.py](https://github.com/digital-ECMT/CORONET_tool/blob/main/most_similar_patients_script.py) - a script to test finding similar patients


## Cite
Detailed description of the process of developing CORONET can be found in our publications:



- [2021 ASCO Annual Meeting I - Meeting Abstract](https://ascopubs.org/doi/10.1200/JCO.2021.39.15_suppl.1505)

  _CORONET; COVID-19 in Oncology evaluatiON Tool: Use of machine learning to inform management of COVID-19 in patients with cancer._
Rebecca Lee, Oskar Wysocki, Cong Zhou, Antonio Calles, Leonie Eastlake, Sarju Ganatra, Michelle Harrison, Laura Horsley, Prerana Huddar, Khurum Khan, Hayley Mckenzie, Carlo Palmieri, Jacobo Rogado Revuelta, Anne Thomas, Caroline Wilson, Tim Cooksley, Caroline Dive, Andre Freitas, Anne Caroline Armstrong, and CORONET Consortium
Journal of Clinical Oncology 2021 39:15_suppl, 1505-1505; DOI: 10.1200/JCO.2021.39.15_suppl.1505

- [_Establishment of CORONET; COVID-19 Risk in Oncology Evaluation Tool to identify cancer patients at low versus high risk of severe complications of COVID-19 infection upon presentation to hospital_](https://ascopubs.org/doi/10.1200/JCO.2021.39.15_suppl.1505)
  R.J. Lee, C. Zhou, O. Wysocki, R. Shotton, A. Tivey, L. Lever, J. Woodcock, A. Angelakas, T. Aung, K. Banfill, M. Baxter, T. Bhogal, H. Boyce, E. Copson, E. Dickens, L. Eastlake, H. Frost, F. Gomes, D.M Graham, C. Hague, M. Harrison, L. Horsley, P. Huddar, Z. Hudson, S. Khan, U. T. Khan, A. Maynard, H. McKenzie, T. Robinson, M. Rowe, Anne Thomas, Lance Turtle, R. Sheehan, A. Stockdale, J. Weaver, S. Williams, C. Wilson, R. Hoskins, J. Stevenson, P. Fitzpatrick, C. Palmieri, D. Landers, T Cooksley, C. Dive, A. Freitas, A. C. Armstrong;
medRxiv 2020.11.30.20239095; doi: https://doi.org/10.1101/2020.11.30.20239095


## Model evolution
- version 1 available at [medRxiv](https://www.medrxiv.org/content/10.1101/2020.11.30.20239095v1)
- version 2 published in JCO Clinical Cancer Informatics
