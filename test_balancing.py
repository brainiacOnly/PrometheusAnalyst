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
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE

if __name__ == "__main__":
	print 'test started...'
	course_id = 'KNU/101/2014_T2'
	user_id = 20
	targetColumns = pd.read_csv('data\dummy\ds_x0.csv').columns.tolist()[:-1]
	
	result = []
	main_start_time = time.time()
	for i in range(0,4):
		name = 'data\dummy\ds_x{0}.csv'.format(i*5)
		baseFrame = pd.read_csv(name)
		
		input = baseFrame[targetColumns]
		output = baseFrame.status
		
		#models = [RandomUnderSampler(), RandomOverSampler(), TomekLinks(),NeighbourhoodCleaningRule(),
        #    SMOTE(kind='regular'), SMOTE(kind='borderline1'), 
        #    SMOTE(kind='borderline2'), SMOTE(kind='svm'),
        #   NearMiss(version=3)]
		models = [RandomUnderSampler(), RandomOverSampler(),NeighbourhoodCleaningRule(n_jobs=-1),SMOTE(kind='regular',n_jobs=-1)]
		row = []
		for m in models:
			start_time = time.time()
			X, y = m.fit_sample(input, output)
			row.append(time.time() - start_time)
			
			print 'cross_validation_timing on name "{0}" is {1} for sampler {2}'.format(name, row[-1], m)
			
		result.append(row)
		
		print str(i*5) + 'is done after ' + str(time.time() - main_start_time) + ' seconds after start'

	df = pd.DataFrame(result)
	df.to_csv('data\\result_balancing.csv',header=False,index=False,sep='\t')
	print 'test finished!'
