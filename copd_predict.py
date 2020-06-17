import pandas as pd	
import pickle as p
import numpy as np	
					   

def predict_helath(patient_input):
	wellbeing_df = pd.read_csv('AllAssessment-updated - Copy.csv')
	input_cols = wellbeing_df.drop(['Gender','Age (Yrs)','Patient Name','NHS ID','Status','Performed By','Performed Date'], 
								   axis = 1).columns
	training_input_data = wellbeing_df[input_cols]
	mapped_values = {'Breathing today': {'Bad': 0,
                                        'Excellent': 1,
                                        'Fair': 2,
                                        'Good': 3,
                                        'Very Bad': 4},
                    'Breathing affecting daily activity': {'A Little': 0,
                                                           'Fair Amount': 1,
                                                           'Much': 2,
                                                           'Not At All': 3,
                                                           'Very Much': 4},
                    'Breathing affecting physical activity': {'A Little': 0,
                                                              'Fair Amount': 1,
                                                              'Much': 2,
                                                              'Not At All': 3,
                                                              'Very Much': 4},
                    'Cough today': {'Better than usual': 0,
                                    'Much better than usual': 1,
                                    'Much worse than usual': 2,
                                    'NO': 3,
                                    'Usual': 4,
                                    'Worse than usual': 5},
                    'Daily Sputum production': {'None': 0,
                                                'Up to 15mls (1 tablespoon)': 1,
                                                'Up to 30mls (1 eggcup)': 2,
                                                'Up to 5mls (1 teaspoon)': 3,
                                                'Up to or 50mls (1 cup or more)': 4},
                    'Colour of Sputum': {'Dark green': 0,
                                         'Watery Clear Transparent': 1,
                                         'Watery Cloudy Colourless': 2,
                                         'Yellowish': 3,
                                         'Yellowish green': 4},
                    'How much tired today?': {'Better than usual': 0,
                                              'Much better than usual': 1,
                                              'Much worse than usual': 2,
                                              'NO': 3,
                                              'Usual': 4,
                                              'Worse than usual': 5},
                    'Ankle swelling today?': {'Better than usual': 0,
                                              'Much better than usual': 1,
                                              'Much worse than usual': 2,
                                              'NO': 3,
                                              'Usual': 4,
                                              'Worse than usual': 5},
                    'Chest Pain today?': {'NO': 0, 'YES': 1},
                    'Heartburn today?': {'NO': 0, 'YES': 1},
                    'Exacerbation of COPD': {'NO': 0, 'YES': 1},
                    'Receiving treatment for COPD flare-up': {'NO': 0, 'YES': 1}}
	
	lst = []
	for k,v in eval(patient_input).items():
		lst.append(mapped_values[k][v])
	
	modelfile = "C:/Users/itzai/PycharmProjects/Project/COPD/final_prediction_rfc.pickle"
	model = p.load(open(modelfile, 'rb'))
	return {'prediction': np.array2string(model.predict([lst]))}