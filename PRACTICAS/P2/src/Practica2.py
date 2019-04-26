# -*- coding: utf-8 -*-
"""
Práctica 1: APC
Estudiante: JJavier Alonso Ramos

"""

from scipy.io import arff
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from time import time
import sys
from prettytable import PrettyTable
import matplotlib.pyplot as plt

###################################################
### Funciones para la codificación de etiquetas ###
###################################################
byte2string = lambda x: x.decode('utf-8')

#######################################
### Función para leer archivos arff ###
#######################################
def read_arff(name):
	path = '../data/'+name+'.arff'
	file, meta = arff.loadarff(path)

	df_data = pd.DataFrame(file)	# Transformamos file (los datos) en un data-frame
	data = df_data.values			# Transformamos este data-frame en una matriz para que sea más manejable

	return data, meta

##############################################################################
### Devuelve las etiquetas (clases) de los elementos del conjunto de datos ###
##############################################################################
def get_tags(data):
	tags = []
	for _data in data:
		tags.append(byte2string(_data[-1]))
	tags = np.asarray(tags)

	return tags 

def get_only_data(data):
	return (data[:,0:-1]).astype(float)

###########################################
### k-NN ; leave-one-out when necessary ###
###########################################
def k_NN(data_training, tags_training, w, data_test = None, tags_test = None, is_training = True):
	w_prim = np.copy( w )
	w_prim[w_prim < 0.2] = 0.0
	eliminated = w_prim[w_prim < 0.2].shape[0]
	hit = 0
	hit_rate = 0.0

	data_training_mod = (data_training*w_prim)[:, w_prim > 0.2]

	tree = KDTree(data_training_mod)
	if is_training:
		nearest_ind = tree.query(data_training_mod, k=2, return_distance=False)[:,1]
		hit_rate = np.mean( tags_training[nearest_ind] == tags_training )
	else:
		data_test_mod = (data_test*w_prim)[:, w_prim > 0.2]
		nearest_ind = tree.query(data_test_mod, k=1, return_distance=False)
		for i in range(nearest_ind.shape[0]):
			if tags_training[nearest_ind[i]] == tags_test[i]:
				hit += 1

		hit_rate = hit/data_test_mod.shape[0]


	reduction_rate = eliminated/w.shape[0]

	f = (hit_rate + reduction_rate)* 0.5

	return f, hit_rate, reduction_rate

############
### 1-NN ###
############
def _1_NN(data_training, tags_training, data_test, tags_test):
	hit = 0

	tree = KDTree(data_training)
	nearest_ind = tree.query(data_test, k=1, return_distance=False)
	for i in range(nearest_ind.shape[0]):
		if tags_training[nearest_ind[i]] == tags_test[i]:
			hit += 1

	hit_rate = hit/data_test.shape[0]
	reduction_rate = 0.0

	f = (hit_rate + reduction_rate)* 0.5

	return f, hit_rate, reduction_rate

########################
### Algoritmo Greedy ###
########################
def relief(data, tags):
####################################### BUCLES ####################################################
	"""
	w = np.zeros(data.shape[1])
	closest_enemy_id = -4
	closest_friend_id = -4

	for i in range(data.shape[0]):
		enemy_distance = 999
		friend_distance = 999
		for j in range(data.shape[0]):
			if i != j:
				current_distance = np.linalg.norm(data[i] - data[j])

				if tags[i] == tags[j] and current_distance < friend_distance:
					friend_distance = current_distance
					closest_friend_id = j
				elif tags[i] != tags[j] and current_distance < enemy_distance:
					enemy_distance = current_distance
					closest_enemy_id = j

		w = w + np.abs(data[i] - data[closest_enemy_id]) - np.abs(data[i] - data[closest_friend_id])
	"""
######################################### KDTree ##################################################
	
	w = np.zeros(data.shape[1])
	closest_enemy_id = -4
	closest_friend_id = -4
	ally_found = False
	enemy_found = False

	tree = KDTree(data)
	nearest_ind = tree.query(data, k=data.shape[0], return_distance=False)[:,1:]

	for i in range(nearest_ind.shape[0]):
		for j in range(nearest_ind.shape[1]):
			if not ally_found and tags[i] == tags[ nearest_ind[i,j] ]:
				ally_found = True
				closest_friend_id = nearest_ind[i,j]
			elif not enemy_found and tags[i] != tags[ nearest_ind[i,j] ]:
				enemy_found = True
				closest_enemy_id = nearest_ind[i,j]
			if ally_found and enemy_found:
				break
		ally_found = enemy_found = False
		w = w + np.abs(data[i] - data[closest_enemy_id]) - np.abs(data[i] - data[closest_friend_id])
	
###########################################################################################

	w_max = np.max(w)
	w[ w < 0.0] = 0.0
	w /= w_max

	
	# Comentado para no retrasar la ejecucion del algoritmo
	"""
	for i in range(len(w)):
		plt.bar(i,w[i])
	plt.show()
	"""

	return w

###################
### Local Seach ###
###################
def local_search(data, tags):
	w = np.random.uniform(0.0,1.0,data.shape[1])
	max_eval = 15000
	max_neighbors = 20*data.shape[1]
	n_eval = 0
	n_neighbors = 0
	variance = 0.3
	mean = 0.0
	class_prev, h, r = k_NN(data, tags, w)

	while n_eval < max_eval and n_neighbors < max_neighbors:
		for i in range(w.shape[0]):
			n_eval += 1
			prev = w[i]
			w[i] = np.clip(w[i] + np.random.normal(mean, variance), 0, 1)
			class_mod, h, r = k_NN(data, tags, w)

			if(class_mod > class_prev):
				n_neighbors = 0
				class_prev = class_mod
				break
			else:
				w[i] = prev
				n_neighbors += 1

	"""
	for i in range(len(w)):
		plt.bar(i,w[i])
	plt.show()
	"""

	return w

###############
### AGG-BLX ###
###############
def agg_blx():
	

###########################################################################################
###··································### MAIN ###·······································###
###########################################################################################

if len(sys.argv) == 2:
	np.random.seed(int(sys.argv[1]))
else:
	np.random.seed(1)

archivos = ['colposcopy', 'texture', 'ionosphere']
for archivo in archivos:
	data, meta = read_arff(archivo)
	tags = get_tags(data)
	_data = get_only_data(data)

	scaler = MinMaxScaler()
	scaler.fit(_data)
	_data = scaler.transform(_data)

	partition = 0
	mean_test_greedy_hr = 0.0
	mean_test_greedy_rr = 0.0
	mean_test_greedy_f = 0.0
	mean_test_greedy_t = 0.0
	mean_test_LS_hr = 0.0
	mean_test_LS_rr = 0.0
	mean_test_LS_f = 0.0
	mean_test_LS_t = 0.0
	mean_test_1nn_hr = 0.0
	mean_test_1nn_rr = 0.0
	mean_test_1nn_f = 0.0

	table_greedy = PrettyTable(['Partición', '%_clas', '%_red', 'Agr.', 'T'])
	table_ls = PrettyTable(['Partición', '%_clas', '%_red', 'Agr.', 'T'])
	table_1nn = PrettyTable(['Partición', '%_clas', '%_red', 'Agr.', 'T'])
	mean_table = PrettyTable(['Algorithm', '%_clas', '%_red', 'Agr.'])

	skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
	for train_index, test_index in skf.split(_data, tags):
		x_train, x_test = _data[train_index], _data[test_index]
		y_train, y_test = tags[train_index], tags[test_index]
		partition += 1

		print('\n\n Partición ', partition)

		###·· GREEDY ··###
		print('··· Calculando pesos por medio de Relief ···')
		ini_time = time()
		w_g = relief(x_train, y_train)
		fin_time = time()
		dif_time_g = fin_time - ini_time
		mean_test_greedy_t	+= dif_time_g

		###·· LOCAL SEARCH ··###
		print('··· Calculando pesos por medio de Local Search ···')
		ini_time = time()
		w_ls = local_search(x_train, y_train)
		fin_time = time()
		dif_time_ls = fin_time - ini_time
		mean_test_LS_t += dif_time_ls

		###·· TEST GREEDY ··###
		print('··· Evaluando Relief ···')
		f, hr, rr = k_NN(x_train, y_train, w_g, x_test, y_test, False)
		mean_test_greedy_f = mean_test_greedy_f + f
		mean_test_greedy_hr = mean_test_greedy_hr + hr
		mean_test_greedy_rr = mean_test_greedy_rr + rr

		table_greedy.add_row([partition, 100*hr, 100*rr, f*100, dif_time_g])

		###·· TEST LOCAL SEARCH ··###
		print('··· Evaluando Local Search ···')
		f, hr, rr = k_NN(x_train, y_train, w_ls, x_test, y_test, False)
		mean_test_LS_f = mean_test_LS_f + f
		mean_test_LS_hr = mean_test_LS_hr + hr
		mean_test_LS_rr = mean_test_LS_rr + rr

		table_ls.add_row([partition, 100*hr, 100*rr, f*100, dif_time_ls])

		###·· TEST 1-NN ··###
		print('··· Evaluando 1-NN ···')
		f, hr, rr = _1_NN(x_train, y_train, x_test, y_test)
		mean_test_1nn_f = mean_test_1nn_f + f
		mean_test_1nn_hr = mean_test_1nn_hr + hr
		mean_test_1nn_rr = mean_test_1nn_rr + rr

		table_1nn.add_row([partition, 100*hr, 100*rr, f*100, 0])

	table_1nn.add_row(['Media', 100*mean_test_1nn_hr/5, 100*mean_test_1nn_rr/5, 100*mean_test_1nn_f/5, 0])
	table_greedy.add_row(['Media', 100*mean_test_greedy_hr/5, 100*mean_test_greedy_rr/5, 100*mean_test_greedy_f/5, mean_test_greedy_t/5])
	table_ls.add_row(['Media', 100*mean_test_LS_hr/5, 100*mean_test_LS_rr/5, 100*mean_test_LS_f/5, mean_test_LS_t/5])

	mean_table.add_row(['1-NN', 100*mean_test_1nn_hr/5, 100*mean_test_1nn_rr/5, 100*mean_test_1nn_f/5])
	mean_table.add_row(['RELIEF', 100*mean_test_greedy_hr/5, 100*mean_test_greedy_rr/5, 100*mean_test_greedy_f/5])
	mean_table.add_row(['LOCAL SEARCH', 100*mean_test_LS_hr/5, 100*mean_test_LS_rr/5, 100*mean_test_LS_f/5])

	print(table_1nn.get_string(title='Resultados 1-NN - '+archivo))
	print(table_greedy.get_string(title='Resultados RELIEF - '+archivo))
	print(table_ls.get_string(title='Resultados LOCAL SEARCH - '+archivo))
	print(mean_table.get_string(title='Media de resultados - '+archivo))