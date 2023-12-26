# Bienvenidos a mi repo de Github

En este repositorio registro todo lo referente a mis clases de Machine Learning en el máster en Data Science que estoy cursando actualmente.

***

# Práctica de Machine Learning:

### Repositorio de Github:
El trabajo se encuentra en la carpeta ./ML_Project_Bank_Fraud de mi repositorio personal, en el cual voy guardando el progreso de cada clase.

El link del repo de github es el siguiente: https://github.com/rodrifer10/mds_ml_rfc.git

### Autor:
Rodrigo Fernández Campos - rodrigo.fernandezc@cunef.edu / rfc10.data@gmail.com


#### Los principales objetivos de esta práctica son:

* <u>Generales</u>:
    * Poner en práctica los conocimientos aprendidos en la materia de Machine Learning en un proyecto "real"
    * Aprender todo lo posible en relación al análisis de los datos, su procesamiento y una serie de modelos de clasificación que utilizaremos
    * Entender el funcionamiento de un proceso de modelado de un proyecto real en primera persona y en parte de manera autodidácta, con pruebas y errores
1. <u>EDA</u>:
    * Conocer el DataSet con el que vamos a trabajar
    * Realizar un análisis de las variables que lo componen y un análisis general se sus instancias
    * Comprender la distribución de los datos, las relaciones entre variables y sobre todo con la variable Target
    * Dar concluisiones sobre cómo se conforma, sus principales componentes y qué podemos observar con un análisis exploratorio
2. <u>Aplicación de modelos predictivos</u>:
    * Realizar exitósamente todos los procedimientos de procesamiento de variables y selección de las mismas para poder aplicar modelos sin problemas a los datos
    * Realizar comparaciones y entender porqué determinados modelos funcionan mejor que otros
    * Determinar las mejores maneras de medir los resultados de las aplicaciones en cada caso
    * Entender como tratar con datos desbalanceados, que técnicas se pueden utilizar y constatar si su aplicación realmente es efectiva, al menos en este caso
    * Verificar la aplicación de modelos en distintas configuraciones de datos para ver como cambian las métricas elegidas
    * Lograr una mejora de modelos mediante el uso de técnicas como el Cross Validation y el Hyperparameters tunning
    * Aplicar exitósamente el/los modelo/s elegido/s, medirlo y sacar conclusiones en base a los resultados obtenidos
3. <u>Explicabilidad</u>:
    * Sección en la que buscaremos dar explicaciones en relación al funcionamiento interno del modelo, mediante el uso de otros modelos específicos para estos objetivos
    * La idea será comprender de que manera funciona, cuales son las variables más y menos importantes, de que depende que el modelo tome una decisión u otra según el caso que esté evaluando, entre otros temas interesantes a tratar.


#### Composición del repositorio:

Este directorio está compuesto por las siguientes carpetas:
* data: carpeta contenedora de todos los archivos con datos del proyecto. Dentro de ella se encuentran 3 carpetas:
    - raw: dataset original descargado de Kaggle
    - interim: datasets en estado de preprocesamiento, se trata de archivos transitorios hasta que esté completamente procesado el dataset
    - processed: dataset procesado y listo para ser utilizado en el modelado
* docs: carpeta que contiene documentos de interés para el caso en análisis
* src: directorio que contiene las funciones .py auxiliares para importar en los notebooks
* html: carpeta en donde se encuentran todos los archivos html exportados
* images: directorio para guardar imágenes relevantes o que se importen en notebooks
* models: directorio en el que se guardaran los modelos .pickle
* notebooks: carpeta que contiene los notebooks con todos los procesos realizados
* reports: directorio reservado para presentar algún reporte que pueda generarse de nuestros análisis
* experiments: notebook dedicado a pruebas o desarrollos que no ameritan estar en el core principal del proyecto pero que son incluídos en el directorio por motivos de documentación del proceso
* env: carpeta con los requerimientos del enviroment para poder correr todo el código sin problemas.



