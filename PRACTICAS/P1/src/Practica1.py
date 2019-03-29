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
from time import time

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
	df_meta = pd.DataFrame(meta)	# Transformamos meta en un data-frame
	meta = df_meta.values			# Transformamos este data-frame en una matriz para que sea más manejable

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
	sub1 = data[tags == tags[0]]
	sub2 = data[tags != tags[0]]

	porc_atrb_1 = int(0.2*num_element_in_class[tags_class[0]])
	porc_atrb_2 = int(0.2*num_element_in_class[tags_class[1]])

	conjunto_1 = np.r_[ sub1[0:porc_atrb_1], sub2[0:porc_atrb_2] ]
	conjunto_2 = np.r_[ sub1[porc_atrb_1:porc_atrb_1*2], sub2[porc_atrb_2:porc_atrb_2*2] ]
	conjunto_3 = np.r_[ sub1[porc_atrb_1*2:porc_atrb_1*3], sub2[porc_atrb_2*2:porc_atrb_2*3] ]
	conjunto_4 = np.r_[ sub1[porc_atrb_1*3:porc_atrb_1*4], sub2[porc_atrb_2*3:porc_atrb_2*4] ]
	conjunto_5 = np.r_[ sub1[porc_atrb_1*4:], sub2[porc_atrb_2*4:] ]

	conjuntos = np.array([conjunto_1, conjunto_2, conjunto_3, conjunto_4, conjunto_5])

	return conjuntos

##################
### k-NN ; k=1 ###
##################
def k_NN(data, tags, w):
	eliminated = 0
	right = 0

	for i in range(w.shape[0]-1, -1, -1):
		if w[i] == 0.0:
			data = np.delete(data, i, 1)
			w 	 = np.delete(w, i)
			eliminated = eliminated +1

	data_mod = data*w
	tree = cKDTree(data_mod)
	nearest_dist, nearest_ind = tree.query(data_mod, k=2)

	for i in range(data.shape[0]):
		calculated_tag = tags[ nearest_ind[i,1] ]
		if tags[i] == calculated_tag:
			right = right+1

	hit_rate = 100*right/data.shape[0]
	reduction_rate = 100*eliminated/data.shape[0]

	f = F(hit_rate, reduction_rate, 0.5)

	return f, hit_rate, reduction_rate

########################
### Algoritmo Greedy ###
########################
def relief(data, tags, tags_class):
	#########################
	### greedy con bucles ###
	#########################

	num_data = data.shape[0]
	num_attributes = data.shape[1]
	w = np.zeros(num_attributes)
	closest_enemy = np.zeros_like(w)
	enemy_distance = 999
	closest_friend = np.zeros_like(w)
	friend_distance = 999

	for i in range(num_data):
		for j in range(num_data):

			if i != j:
				current_distance = np.linalg.norm(data[i] - data[j])

				if tags[i] == tags[j] and current_distance < friend_distance:
					friend_distance = current_distance
					closest_friend = data[j]
				elif tags[i] != tags[j] and current_distance < enemy_distance:
					enemy_distance = current_distance
					closest_enemy = data[j]

		w = w + abs(data[i] - closest_enemy) - abs(data[i] - closest_friend)

	"""
	#########################
	### greedy con KDTree ###
	#########################

	num_data = data.shape[0]
	num_attributes = data.shape[1]
	w = np.zeros(num_attributes)
	equipo1 = tags[0]
	iteraciones = [0,0]

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
		if tags[i] == equipo1:
			indice = iteraciones[0]
		else:
			indice = iteraciones[1]
		

		w = w + abs(data[i] - data[nearest_ind_enemy[0,1]]) - abs(data[i] - data[nearest_ind_ally[indice,1]])


		if tags[i] == equipo1:
			iteraciones[0] = iteraciones[0]+1
		else:
			iteraciones[1] = iteraciones[1]+1
	"""		
	w_max = np.max(w)

	for i in range(w.shape[0]):
		if w[i] < 0.0:
			w[i] = 0.0
		else:
			w[i] = w[i] / w_max

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

for i in range(0,5):

	data_train = np.r_[conjuntos[(i+1)%5], conjuntos[(i+2)%5], conjuntos[(i+3)%5], conjuntos[(i+4)%5]]
	data_test = conjuntos[i]

	time_ini = time()
	w = relief(data_train[:,0:-1], data_train[:,-1], tags_class)
	time_end = time()
	#print(time_end - time_ini)

	f, hr, rr = k_NN(data_train[:,0:-1], data_train[:,-1], w)

	print('TRAIN_' + str(i))
	print('f = ' + str(f))
	print('hit_rate = ' + str(hr))
	print('reduction_rate = ' + str(rr))
	print('')

	f, hr, rr = k_NN(data_test[:,0:-1], data_test[:,-1], w)

	print('TEST_' + str(i))
	print('f = ' + str(f))
	print('hit_rate = ' + str(hr))
	print('reduction_rate = ' + str(rr))
	print('\n')