import pandas as pd
import numpy as np
import time
import copy
import datetime
from sklearn import linear_model
from sklearn.cross_validation import cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from imblearn.under_sampling import RandomUnderSampler

print 'test started...'

#######data load#######
course_id = 'KNU/101/2014_T2'
user_id = 20
#sm = pd.read_csv('data\courseware_studentmodule.csv')
#users = pd.read_csv('data\users.csv')
#cert = pd.read_csv('data\certificates_generatedcertificate.csv')
targetColumns = pd.read_csv('data\dummy\ds_x0.csv').columns.tolist()[:-1]
#targetColumns = sm[(sm.module_type == 'problem') & (sm.course_id == course_id)].module_id.unique()

#sm = sm[sm.module_type == 'problem']
#sm = sm.rename(columns = {'student_id':'user_id'})
#sm.grade = map(lambda x: x if x != '\\b' else 0,sm.grade.tolist())
#cert = cert[cert.course_id == course_id]
#registered = cert.user_id.unique()
#users = users[users.id.isin(registered)]
########

##__getProblemResultFeatures
#start_time = time.time()
#modules = sm.module_id.unique().tolist()
#result = []
#for module in modules:
#	first = sm[sm.module_id == module].created.min()#
#	result.append({'name':module,'date':first})
#labels = map(lambda x: x['name'],sorted(result,key=lambda x: x['date']))
#rebased = sm.pivot(index='user_id',columns='module_id',values='grade')
#rebased = rebased[labels].fillna(0)
#res_features =  rebased
#print '__getProblemResultFeatures finished in ' + str(time.time() - start_time)
##

result = []
main_start_time = time.time()
for i in range(0,17):
	name = 'data\dummy\ds_x{0}.csv'.format(i*5)
	baseFrame = pd.read_csv(name)
	
	#undersampling
	#rus = RandomUnderSampler(return_indices=True)
	#X_resampled, y_resampled, idx_resampled = rus.fit_sample(baseFrame[targetColumns], baseFrame.status)
	#input = X_resampled
	#output = y_resampled
	input = baseFrame[targetColumns]
	output = baseFrame.status
	
	#prediction test
	models = [GaussianNB(),KNeighborsClassifier(),linear_model.LogisticRegression()]
	row = []
	for m in models:
		model = GaussianNB()
		start_time = time.time()
		predicted = cross_val_predict(model, input, output)
		row.append(time.time() - start_time)
		
		print 'cross_validation_timing on name "{0}" is {1}'.format(name, row[-1])
		
		score = float(np.sum(predicted == output))/len(input)
		start_time = time.time()
		model.fit(input,output)
		row.append(time.time() - start_time)
		print 'fit_timing on name "{0}" is {1}'.format(name, row[-1])
		
		#ols = res_features.columns.tolist()[:last]
		start_time = time.time()
		answer = model.predict(baseFrame[targetColumns].ix[user_id].tolist())
		row.append(time.time() - start_time)
		print 'predict_timing on name "{0}" is {1}'.format(name, row[-1])
	result.append(row)
	
	print str(i*5) + 'is done after ' + str(time.time() - main_start_time) + ' seconds after start'

df = pd.DataFrame(result)
df.to_csv('data\\result.csv',header=False,index=False,sep='\t')
print 'test finished!'
#GaussianNB
#cross_validation_timing = [2.1360001564,12.0580000877,26.1510000229,57.9549999237,132.995000124,195.130000114]
#fit_timing = [0.651000022888,3.40100002289,7.73900008202,21.4050002098,48.4249999523,67.4809999466]
#predict_timing = [0.588999986649,0.615000009537,2.70199990273,2.86199998856,3.00099992752,3.16200017929]
#--------------
