# -*- coding: utf-8 -*-
"""
Práctica 1: APC
Estudiante: JJavier Alonso Ramos

"""

from scipy.io import arff
import numpy as np
import pandas as pd
import sys 								# Leer parámetros de entrada

# Funcion para leer archivos arff
def read_arff(path):
	file, meta = arff.loadarff(path)

	df_data = pd.DataFrame(file)	# Transformamos file (los datos) en un data-frame
	data = df_data.values			# Transformamos este data-frame en una matriz para que sea más manejable
	df_meta = pd.DataFrame(meta)	# Transformamos meta en un data-frame
	meta = df_meta.values			# Transformamos este data-frame en una matriz para que sea más manejable

	return data, meta

# Función para dividir el conjunto de datos en 5 subconjuntos de mismas proporciones
def _5_fold_cross_validation(data):


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
def relief(data, tags):
	# data no contendrá la columna de etiquetas
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
				current_distance = euclidean_distance(data[i], data[j])

				if tags[i] == tags[j] and current_distance < friend_distance:
					friend_distance = current_distance
					closest_friend = data[j]
				elif tags[i] != tags[j] and current_distance < enemy_distance:
					enemy_distance = current_distance
					closest_enemy = data[j]

		w = w + abs(data[i] - closest_enemy) - abs(data[i] - closest_friend)

	w_max = np.max(w)

	for i in range(w.shape[0]):
		if w[i] < 0.0:
			w[i] = 0.0
		else:
			w[i] = w[i] / w_max

	return w


if len(sys.argv) != 2: 							# Indicar como segundo parámetro el nombre del fichero con el que se va a trabajar
	archivo = input('Indique path/archivo.arff que desea abrir: \n')
else:
	archivo = sys.argv[1] 						# Leemos el fichero y lo cargamos en 'file'

data, meta = read_arff(archivo)


print(data.shape)
print(meta)

tags = data[:,data.shape[1]-1]
data = data[:,0:data.shape[1]-1]

print(data.shape)
print(tags.shape)
w = relief(data,tags)

print(w)