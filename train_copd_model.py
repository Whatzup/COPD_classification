import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
import pickle

le = LabelEncoder()

wellbeing_df = pd.read_csv('AllAssessment-updated - Copy.csv')
df_for_le = pd.read_csv('AllAssessment-updated - Copy.csv')	


input_cols = wellbeing_df.drop(['Gender','Age (Yrs)','Patient Name','NHS ID','Status','Performed By','Performed Date'], 
                               axis = 1).columns
							   
training_input_data = wellbeing_df[input_cols]
target_var = wellbeing_df.Status

le_inputs= training_input_data.apply(le.fit_transform)
x_train, x_test, y_train, y_test = train_test_split(le_inputs, target_var, test_size = 0.3)

from sklearn.tree import DecisionTreeClassifier
dtc_model = DecisionTreeClassifier()
dtc_model.fit(x_train, y_train)
dtc_model.score(x_test, y_test)*100
#dtc_model.score(x_train, y_train)*100

from sklearn.ensemble import RandomForestClassifier
rfc_model = RandomForestClassifier(n_estimators=100)
rfc_model.fit(x_train, y_train)
rfc_model.score(x_test, y_test)*100
#rfc_model.score(x_train, y_train)*100

if dtc_model.score(x_test, y_test)*100 > rfc_model.score(x_test, y_test)*100:
	pickle.dump(rfc_model, open('final_prediction_dtc.pickle', 'wb'))
else:
	pickle.dump(dtc_model, open('final_prediction_rfc.pickle', 'wb'))