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

# Funciones codificación de etiquetas
byte2string = lambda x: x.decode('utf-8')
string2int = lambda x: int(x)

# Funcion para leer archivos arff
def read_arff(path):
	file, meta = arff.loadarff(path)

	df_data = pd.DataFrame(file)	# Transformamos file (los datos) en un data-frame
	data = df_data.values			# Transformamos este data-frame en una matriz para que sea más manejable
	df_meta = pd.DataFrame(meta)	# Transformamos meta en un data-frame
	meta = df_meta.values			# Transformamos este data-frame en una matriz para que sea más manejable

	return data, meta

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

# Función para dividir el conjunto de datos en 5 subconjuntos de mismas proporciones
#def _5_fold_cross_validation(data):


# Función auxiliar para calcular distancias entre dos elementos
def euclidean_distance(x_i, x_j):
	# Los vectores x_i y x_j no contendrán la etiqueta
	diff = x_i-x_j		# Resta de componentes
	mul = diff*diff		# Elevado al cuadrado
	res = mul**0.5		# Raiz cuadrada --------- SE PUEDE QUITAR? ----------
	return np.sum(res)	# Sumatoria de todas las distancias

# Vamos a implementar el algoritmo de búsqueda local
#def local_search(data, tags):

# Algoritmo Greedy
def relief(data, tags, tags_class):
	num_data = data.shape[0]
	num_attributes = data.shape[1]
	w = np.zeros(num_attributes)

	for i in range(num_data):
		allies  = data[ tags == tags[i] ]
		enemies	= data[ tags != tags[i] ]
		#print(data[i].shape)
		row = enemies.shape[0]
		col = enemies.shape[1]

		enemies = np.append(data[i],enemies)
		enemies = np.reshape(enemies,(row+1,col))


		ally_tree	= cKDTree(allies)
		enemy_tree	= cKDTree(enemies)

		nearest_dist_ally, nearest_ind_ally = ally_tree.query(allies, k=2)
		nearest_dist_enemy, nearest_ind_enemy = enemy_tree.query(enemies, k=2)

		print(nearest_ind_enemy)

		w = w + abs(data[i] - data[nearest_ind_enemy[0,1]]) - abs(data[i] - data[nearest_ind_ally[i,1]])

	w_max = np.max(w)

	for i in range(w.shape[0]):
		if w[i] < 0.0:
			w[i] = 0.0
		else:
			w[i] = w[i] / w_max

	print(w)

	return w



###########################################################################################
#################################### MAIN #################################################
###########################################################################################

if len(sys.argv) != 2: 							# Indicar como segundo parámetro el nombre del fichero con el que se va a trabajar
	archivo = input('Indique path/archivo.arff que desea abrir: \n')
else:
	archivo = sys.argv[1] 						# Leemos el fichero y lo cargamos en 'file'

data, meta = read_arff(archivo)
print(data)

tags = []
for i in data:
	tags.append(byte2string(i[-1]))
tags = np.asarray(tags)

data = data[:,0:-1]

num_element_in_class, tags_class = get_tags_class(tags)

print(tags)

print(num_element_in_class)

print(tags_class)

print(data[tags == tags_class[0]])
w = relief(data, tags, tags_class)