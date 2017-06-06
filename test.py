import pandas as pd
import numpy as np
import time
import copy
import datetime
from sklearn import linear_model
from sklearn.cross_validation import cross_val_predict
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import SGDClassifier
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import AllKNN
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import RepeatedEditedNearestNeighbours

if __name__ == "__main__":
	print 'test started...'
	course_id = 'KNU/101/2014_T2'
	user_id = 20
	targetColumns = pd.read_csv('data\dummy\ds_x0.csv').columns.tolist()[:-1]
	
	result = []
	main_start_time = time.time()
	for i in range(0,28):
		name = 'data\dummy\ds_x{0}.csv'.format(i*5)
		baseFrame = pd.read_csv(name)
		
		#undersampling
		#rus = RandomUnderSampler(return_indices=True)
		#X_resampled, y_resampled, idx_resampled = rus.fit_sample(baseFrame[targetColumns], baseFrame.status)
		#input = X_resampled
		#output = y_resampled
		input = baseFrame[targetColumns]
		output = baseFrame.status
		#sampler = RandomUnderSampler()
		start_time = time.time()
		#input, output = sampler.fit_sample(input, output)
		#print 'sampling has taken in {0}'.format(time.time() - start_time)
		
		#prediction test
		#models = [GaussianNB(),GaussianNB(),linear_model.LogisticRegression(n_jobs=-1)]
		models = [GaussianNB(),linear_model.SGDClassifier(),linear_model.LogisticRegression()]
		row = []
		for m in models:
			model = m
			start_time = time.time()
			predicted = cross_val_score(model, input, output, cv=10)
			row.append(time.time() - start_time)
			
			print 'cross_validation_timing on name "{0}" is {1}'.format(name, row[-1])
			
			#score = float(np.sum(predicted == output))/len(input)
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
