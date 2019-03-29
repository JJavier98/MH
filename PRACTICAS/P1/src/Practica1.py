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
def read_arff(path):
	file, meta = arff.loadarff(path)

	df_data = pd.DataFrame(file)	# Transformamos file (los datos) en un data-frame
	data = df_data.values			# Transformamos este data-frame en una matriz para que sea más manejable
	#df_meta = pd.DataFrame(meta)	# Transformamos meta en un data-frame
	#meta = df_meta.values			# Transformamos este data-frame en una matriz para que sea más manejable

	return data, meta

#########################################################################################################
### Función que devuelve cuántos tipos distintos de etiquetas hay y cuántos elementos hay de cada una ###
#########################################################################################################
def get_tags_class(tags):
	num_element_in_class = {}
	tags_class = []
	for w in tags:
		if w in num_element_in_class:
			num_element_in_class[w] = num_element_in_class[w]+1
		else:
			num_element_in_class[w] = 1
			tags_class.append(w)

	return num_element_in_class, tags_class

##########################################################################################
### Función para dividir el conjunto de datos en 5 subconjuntos de mismas proporciones ###
##########################################################################################
def _5_fold_cross_validation(data, tags, tags_class, num_element_in_class):
	sub_parts = []
	proportionalities = []

	for i in range( len(tags_class) ):
		# subconjuntos dependiendo de la etiqueta
		np.random.shuffle(data)
		sub_parts.append(data[ tags == tags_class[i] ])
		# 20% de elementos de cada subconjunto
		proportionalities.append( int(0.2*num_element_in_class[tags_class[i]]) )

	sub_parts = np.array(sub_parts)
	proportionalities = np.array(proportionalities)

	conjunto_1 = sub_parts[0][0:proportionalities[0]]
	conjunto_2 = sub_parts[0][proportionalities[0]:proportionalities[0]*2]
	conjunto_3 = sub_parts[0][proportionalities[0]*2:proportionalities[0]*3]
	conjunto_4 = sub_parts[0][proportionalities[0]*3:proportionalities[0]*4]
	conjunto_5 = sub_parts[0][proportionalities[0]*4:]

	for i in range(1,sub_parts.shape[0]):
		conjunto_1 = np.r_[ conjunto_1, sub_parts[i][0:proportionalities[i]] ]
		conjunto_2 = np.r_[ conjunto_2, sub_parts[i][proportionalities[i]:proportionalities[i]*2] ]
		conjunto_3 = np.r_[ conjunto_3, sub_parts[i][proportionalities[i]*2:proportionalities[i]*3] ]
		conjunto_4 = np.r_[ conjunto_4, sub_parts[i][proportionalities[i]*3:proportionalities[i]*4] ]
		conjunto_5 = np.r_[ conjunto_5, sub_parts[i][proportionalities[i]*4:] ]
	"""
	np.random.shuffle(conjunto_1)
	np.random.shuffle(conjunto_2)
	np.random.shuffle(conjunto_3)
	np.random.shuffle(conjunto_4)
	np.random.shuffle(conjunto_5)
	"""

	conjuntos = np.array([conjunto_1, conjunto_2, conjunto_3, conjunto_4, conjunto_5])

	return conjuntos

##################
### k-NN ; k=2 ###
##################
def k_NN_leave_one_out(data, tags, w):
	w_prim = np.copy( w )
	w_prim[w < 0.2] = 0.0
	eliminated = w[w < 0.2].shape[0]
	right = 0

	data_mod = (data*w_prim)[:, w > 0.2] # Puedes hacer (data * w)[:, w > 0.2]
	tree = KDTree(data_mod)
	nearest_ind = tree.query(data_mod, k=2, return_distance=False)[:,1]

	hit_rate = np.mean( tags[nearest_ind] == tags )
	reduction_rate = eliminated/w.shape[0]

	f = F(hit_rate, reduction_rate, 0.5)

	return f

##################
### k-NN ; k=1 ###
##################
def k_NN(data, tags, w):
	w_prim = np.copy( w )
	w_prim[w < 0.2] = 0.0
	eliminated = w[w < 0.2].shape[0]
	right = 0

	data_mod = (data*w_prim)[:, w > 0.2] # Puedes hacer (data * w)[:, w > 0.2]
	tree = KDTree(data_mod)
	nearest_ind = tree.query(data_mod, k=1, return_distance=False)

	hit_rate = np.mean( tags[nearest_ind] == tags )
	reduction_rate = eliminated/w.shape[0]

	f = F(hit_rate, reduction_rate, 0.5)

	return f

########################
### Algoritmo Greedy ###
########################
def relief(data, tags, tags_class):
	#########################
	### greedy con bucles ###
	#########################
	
	num_data = np.copy( data.shape[0] )
	num_attributes = np.copy( data.shape[1] )
	w = np.copy( np.zeros(num_attributes) )
	closest_enemy = np.copy( np.zeros_like(w) )
	enemy_distance = 999
	closest_friend = np.copy( np.zeros_like(w) )
	friend_distance = 999

	for i in range(num_data):
		for j in range(num_data):
			if i != j:
				current_distance = np.copy( np.linalg.norm(data[i] - data[j]) )

				if tags[i] == tags[j] and current_distance < friend_distance:
					friend_distance = np.copy( current_distance )
					closest_friend = np.copy( data[j] )
				elif tags[i] != tags[j] and current_distance < enemy_distance:
					enemy_distance = np.copy( current_distance )
					closest_enemy = np.copy( data[j] )

		w = np.copy( w + np.abs(data[i] - closest_enemy) - np.abs(data[i] - closest_friend) )

	"""
	#########################
	### greedy con KDTree ###
	#########################

	num_data = data.shape[0]
	num_attributes = data.shape[1]
	w = np.zeros(num_attributes)
	equipo1 = tags[0]
	iteraciones = np.zeros_like( len(tags_class) )

	for i in range(num_data):

		allies  = data[ tags == tags[i] ]
		enemies	= data[ tags != tags[i] ]

		row = enemies.shape[0]
		col = enemies.shape[1]

		enemies = np.append(data[i],enemies)
		enemies = np.reshape(enemies,(row+1,col))


		ally_tree	= cKDTree(allies)
		enemy_tree	= cKDTree(enemies)


		nearest_dist_ally, nearest_ind_ally = ally_tree.query(allies, k=2)
		nearest_dist_enemy, nearest_ind_enemy = enemy_tree.query(enemies, k=2)


		indice = 0
		for j in range( len(tags_class) ):
			if tags[i] == tags_class[j]:
				indice = iteraciones[j]
				iteraciones[j] = iteraciones[j]+1
		
		w = w + abs(data[i] - data[nearest_ind_enemy[0,1]]) - abs(data[i] - data[nearest_ind_ally[indice,1]])
"""
	w_max = np.copy( np.max(w) )

	for i in range(w.shape[0]):
		if w[i] < 0.0:
			w[i] = 0.0
		else:
			w[i] = np.copy( w[i] / w_max )

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
			class_prev = k_NN_leave_one_out(data, tags, w)
			w[i] = np.clip(w[i] + np.random.normal(mean, variance), 0, 1)
			class_mod = k_NN_leave_one_out(data, tags, w)

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

# OBTENEMOS CUANTOS CLASES  DISTINTAS DE ETIQUETAS HAY Y CUANTOS ELEMENTOS DE CADA UNA
num_element_in_class, tags_class = get_tags_class(tags)


# SEPARAMOS LOS DATOS EN 5 CONJUNTOS MANTENIENDO LA PROPORCION DE ETIQUETAS IGUAL QUE EN EL CONJUNTO ORIGINAL
conjuntos = _5_fold_cross_validation(data, tags, tags_class, num_element_in_class)

mean_train_greedy = 0.0
mean_test_greedy = 0.0
mean_train_LS = 0.0
mean_test_LS = 0.0

skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
_data = data[:,0:-1]
_class = tags
it = 0
for train_index, test_index in skf.split(_data, _class):
    x_train, x_test = _data[train_index], _data[test_index]
    y_train, y_test = _class[train_index], _class[test_index]

    print('Muestra: ' + str(it))
    it += 1

    w_g = relief(x_train, y_train, tags_class)
    f = k_NN(x_test, y_test, w_g)
    mean_test_greedy = mean_test_greedy + f

    w_ls = local_search(x_train, y_train)
    f = k_NN(x_test, y_test, w_ls)
    mean_test_LS = mean_test_LS + f

"""
for i in range(0,5):

	data_train = np.r_[conjuntos[(i+1)%5], conjuntos[(i+2)%5], conjuntos[(i+3)%5], conjuntos[(i+4)%5]]
	data_test = conjuntos[i]

	time_ini = time()
	w = relief(data_train[:,0:-1], data_train[:,-1], tags_class)
	time_end = time()
	#print(time_end - time_ini)

	f = k_NN(data_train[:,0:-1], data_train[:,-1], w)

	mean_train_greedy = mean_train_greedy + f

	f = k_NN(data_test[:,0:-1], data_test[:,-1], w)

	mean_test_greedy = mean_test_greedy + f
"""
print('TRAIN')
print(100*mean_train_greedy/5)
print('\n\nTEST')
print(100*mean_test_greedy/5)

print('TRAIN')
print(100*mean_train_LS/5)
print('\n\nTEST')
print(100*mean_test_LS/5)