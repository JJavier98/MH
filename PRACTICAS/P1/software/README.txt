data: almacena archivos de datos
doc: aquí se encuentra la memoria
src: contiene el script de python

Hemos creado un archivo de instalación de un entorno de pytthon con todos los módulos utilizados en la práctica que estás escritos en requeriments.txt.
Para crearlo debe ejecutar: ./install.sh
Para activarlo: source /env/bin/active

Tras esto, para ejecutar la práctica con semilla=1 tan solo escribir `make`
Si se quiere especificar la semilla a usar -> editar el archivo Makefile de la siguiente forma:
            python3.6 src/Practica1.py 'semilla'

o ejecutar manualmente desde la terminal.
