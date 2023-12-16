# Bienvenidos a mi repo de Github

En este repositorio registro todo lo referente a mis clases de Machine Learning en el máster en Data Science que estoy cursando actualmente.

***

### Repositorio de Github:
El trabajo se encuentra en la carpeta ./EDA_dataset_tarea de mi repositorio personal, en el cual voy guardando el progreso de cada clase.

El link del repo de github es el siguiente: https://github.com/rodrifer10/mds_ml_rfc.git

### Autor:
Rodrigo Fernández Campos - rodrigo.fernandezc@cunef.edu / rfc10.data@gmail.com

***

# Práctica 1: EDA_DataSet

Este trabajo es la primer etapa de un proyecto en el cual analizaremos una base de datos de aplicaciones para apertura de cuentas bancarias y aplicaremos modelos buscando predecir posibles fraudes.

#### Los principales objetivos de esta primera práctica son:

* Conocer el DataSet con el que vamos a trabajar
* Realizar un análisis de las variables que lo componen y un análisis general se sus instancias
* Comprender la distribución de los datos, las relaciones entre variables y sobre todo con la variable Target
* Depurarlo con el fin de aplicar modelos de Machine Learning sobre él en próximas entregas
* Dar concluisiones sobre cómo se conforma, sus principales componentes y qué podemos observar con un análisis exploratorio

En los notebooks se encuentra el Análisis Exploratorio de Datos dividido en 3 etapas de procesamiento del dataset.

#### En este análisis hemos logrado:

* Conocer el significado y los valores de cada una de las variables
* Saber cuántos valores nulos, missings y outliers contiene nuestro dataset, para luego haber decidido cual sería su tratamiento.
* Comprender no solo el tipo de dato que contienen las variables analizadas, sino también su naturaleza y características, tanto para el análisis gráfico y descriptivo, como el de su futuro uso en nuestros modelos predictivos
* Entender que tratamos con un dataset desbalanceado, algo que resulta natural por la naturaleza del problema que abordamos, en donde los casos de fraude son absolutamente minoritarios en relación a los casos de no fraude en las aplicaciones de cuentas bancarias.
* Separar de manera estratificada al dataset en subsets de train y de test
* Realizar un análisis gráfico exhaustivo de todas las variables del dataset, destacando los principales detalles a tener en cuenta sobre ellas y sobre su relación con la variable objetivo.
* Comprender las corelaciones entre las variables, tanto numéricas como categóricas
* Encodificar las variables categóricas, convirtiendo sus valores en numéricos para poder procesarlas con éxito en los modelos que elijamos
* Realizar un escalado de todas las variables ya transformadas y preprocesadas por si necesitamos usarlo a la hora de aplicar nuestros modelos

#### Composición del repositorio específico del EDA:

Este directorio está compuesto por las siguientes carpetas:
* data: carpeta contenedora de todos los archivos con datos del proyecto. Dentro de ella se encuentran 3 carpetas:
    - raw: dataset original descargado de Kaggle
    - interim: datasets en estado de preprocesamiento, se trata de archivos transitorios hasta que esté completamente procesado el dataset
    - processed: dataset procesado y listo para ser utilizado en el modelado
* docs: carpeta que contiene documentos de interés para el caso en análisis
* functions: directorio que contiene las funciones .py auxiliares para importar en los notebooks
* html: carpeta en donde se encuentran todos los archivos html exportados
* images: directorio para guardar imágenes relevantes o que se importen en notebooks
* models: directorio en el que se guardaran los modelos .pickle
* notebooks: carpeta que contiene los notebooks con todos los procesos realizados
* reports: directorio reservado para presentar algún reporte que pueda generarse de nuestros análisis



