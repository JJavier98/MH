# -*- coding: utf-8 -*-
"""
Práctica 1: APC
Estudiante: JJavier Alonso Ramos

"""

from scipy.io import arff
import numpy as np
import pandas as pd
import sys # Leer parámetros de entrada
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree
from sklearn.model_selection import StratifiedKFold
from time import time
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler

np.random.seed(1)

###################################################
### Funciones para la codificación de etiquetas ###
###################################################
byte2string = lambda x: x.decode('utf-8')
string2int = lambda x: int(x)

###########################
### Función a maximizar ###
###########################
F = lambda hr, rr, alpha: alpha*hr + (1-alpha)*rr 

#######################################
### Función para leer archivos arff ###
#######################################
def read_arff(name):
	path = '../data/'+name+'.arff'
	file, meta = arff.loadarff(path)

	df_data = pd.DataFrame(file)	# Transformamos file (los datos) en un data-frame
	data = df_data.values			# Transformamos este data-frame en una matriz para que sea más manejable
	#df_meta = pd.DataFrame(meta)	# Transformamos meta en un data-frame
	#meta = df_meta.values			# Transformamos este data-frame en una matriz para que sea más manejable

	return data, meta

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

	f = F(hit_rate, reduction_rate, 0.5)

	return f, hit_rate, reduction_rate

########################
### Algoritmo Greedy ###
########################
def relief(data, tags):
####################################### BUCLES ####################################################
	
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
	
######################################### KDTree ##################################################
	"""
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
			if not enemy_found and tags[i] != tags[ nearest_ind[i,j] ]:
				enemy_found = True
				closest_enemy_id = nearest_ind[i,j]
			if ally_found and enemy_found:
				break
		ally_found = enemy_found = False
		w = w + np.abs(data[i] - data[closest_enemy_id]) - np.abs(data[i] - data[closest_friend_id])
	"""
###########################################################################################

	w_max = np.max(w)
	w[ w < 0.0] = 0.0
	w /= w_max

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

	while n_eval < max_eval and n_neighbors < max_neighbors:
		for i in range(w.shape[0]):
			n_eval += 1
			prev = w[i]
			class_prev = k_NN(data, tags, w)
			w[i] = np.clip(w[i] + np.random.normal(mean, variance), 0, 1)
			class_mod = k_NN(data, tags, w)

			if(class_mod > class_prev):
				n_neighbors = 0
				break
			else:
				w[i] = prev
				n_neighbors += 1

	return w

###########################################################################################
###··································### MAIN ###·······································###
###########################################################################################

# NOMBRE ARCHIVO
if len(sys.argv) != 2:
	# Indicar como segundo parámetro el nombre del fichero con el que se va a trabajar
	archivo = input('Indique path/archivo.arff que desea abrir: \n')
else:
	# Leemos el fichero y lo cargamos en 'archivo'
	archivo = sys.argv[1]

# ABRIR ARCHIVO
data, meta = read_arff(archivo)

# OBTENER LAS ETIQUETAS
tags = []
for i in data:
	tags.append(byte2string(i[-1]))
tags = np.asarray(tags)


_data = (data[:,0:-1]).astype(float)
_class = tags
it = 0
mean_test_greedy_hr = 0.0
mean_test_LS_hr = 0.0
mean_test_greedy_rr = 0.0
mean_test_LS_rr = 0.0
mean_test_greedy_f = 0.0
mean_test_LS_f = 0.0

scaler = MinMaxScaler()
scaler.fit(_data)
_data = scaler.transform(_data)

skf = StratifiedKFold(n_splits=5, random_state=3, shuffle=True)

for train_index, test_index in skf.split(_data, _class):
	x_train, x_test = _data[train_index], _data[test_index]
	y_train, y_test = _class[train_index], _class[test_index]

	print('Muestra: ' + str(it))
	it += 1

	w_g = relief(x_train, y_train)
	f, hr, rr = k_NN(x_train, y_train, w_g, x_test, y_test, False)
	mean_test_greedy_f = mean_test_greedy_f + f
	mean_test_greedy_hr = mean_test_greedy_hr + hr
	mean_test_greedy_rr = mean_test_greedy_rr + rr
	print('\nGreedy:')
	print('F:', f, '   Hit-rate:', hr, '   Reduction-rate:', rr,'\n\n')

	w_ls = local_search(x_train, y_train)
	f, hr, rr = k_NN(x_train, y_train, w_ls, x_test, y_test, False)
	mean_test_LS_f = mean_test_LS_f + f
	mean_test_LS_hr = mean_test_LS_hr + hr
	mean_test_LS_rr = mean_test_LS_rr + rr
	print('\nLocal Search:')
	print('F:', f, '   Hit-rate:', hr, '   Reduction-rate:', rr,'\n\n')

print('\n\nTEST_GREEDY')
print('F:', 100*mean_test_greedy_f/5, '%')
print('Hit-rate:', 100*mean_test_greedy_hr/5, '%')
print('Reduction-rate:', 100*mean_test_greedy_rr/5, '%')

print('\n\nTEST_LS')
print('F:', 100*mean_test_LS_f/5, '%')
print('Hit-rate:', 100*mean_test_LS_hr/5, '%')
print('Reduction-rate:', 100*mean_test_LS_rr/5, '%')