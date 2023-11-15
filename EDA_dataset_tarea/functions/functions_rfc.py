import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from termcolor import colored, cprint
import scipy.stats as ss
import warnings


###-------------------------------- Funciones de la cátedra utilizadas ---------------------------------------
def duplicate_columns(frame):
    '''
    Lo que hace la función es, en forma de bucle, ir seleccionando columna por columna del DF que se le indique
    y comparar sus values con los de todas las demás columnas del DF. Si son exactamente iguales, añade dicha
    columna a una lista, para finalmente devolver la lista con los nombres de las columnas duplicadas.
    '''
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []

    for t, v in groups.items():

        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)

        for i in range(lcs):
            ia = vs.iloc[:,i].values
            for j in range(i+1, lcs):
                ja = vs.iloc[:,j].values
                if np.array_equal(ia, ja):
                    dups.append(cs[i])
                    break
    return dups

### -----------------------

def dame_variables_categoricas(dataset=None):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función dame_variables_categoricas:
    ----------------------------------------------------------------------------------------------------------
        -Descripción: Función que recibe un dataset y devuelve una lista con los nombres de las 
        variables categóricas
        -Inputs: 
            -- dataset: Pandas dataframe que contiene los datos
        -Return:
            -- lista_variables_categoricas: lista con los nombres de las variables categóricas del
            dataset de entrada con menos de 100 valores diferentes
            -- 1: la ejecución es incorrecta
    '''
    if dataset is None:
        print(u'\nFaltan argumentos por pasar a la función')
        return 1
    lista_variables_categoricas = []
    other = []
    for i in dataset.columns:
        if (dataset[i].dtype!=float) & (dataset[i].dtype!=int):
            unicos = int(len(np.unique(dataset[i].dropna(axis=0, how='all'))))
            if unicos < 100:
                lista_variables_categoricas.append(i)
            else:
                other.append(i)

    return lista_variables_categoricas, other

### ----

def get_deviation_of_mean_perc(pd_loan, list_var_continuous, target, multiplier):
    """
    Devuelve el porcentaje de valores que exceden del intervalo de confianza
    :type series:
    :param multiplier:
    :return:
    """
    pd_final = pd.DataFrame()
    
    for i in list_var_continuous:
        
        series_mean = pd_loan[i].mean()
        series_std = pd_loan[i].std()
        std_amp = multiplier * series_std
        left = series_mean - std_amp
        right = series_mean + std_amp
        size_s = pd_loan[i].size
        
        perc_goods = pd_loan[i][(pd_loan[i] >= left) & (pd_loan[i] <= right)].size/size_s
        perc_excess = pd_loan[i][(pd_loan[i] < left) | (pd_loan[i] > right)].size/size_s
        
        if perc_excess>0:    
            pd_concat_percent = pd.DataFrame(pd_loan[target][(pd_loan[i] < left) | (pd_loan[i] > right)]\
                                            .value_counts(normalize=True).reset_index()).T
            pd_concat_percent.columns = [pd_concat_percent.iloc[0,0], 
                                         pd_concat_percent.iloc[0,1]]
            pd_concat_percent = pd_concat_percent.drop(target,axis=0)
            pd_concat_percent['variable'] = i
            pd_concat_percent['sum_outlier_values'] = pd_loan[i][(pd_loan[i] < left) | (pd_loan[i] > right)].size
            pd_concat_percent['porcentaje_sum_null_values'] = perc_excess
            pd_final = pd.concat([pd_final, pd_concat_percent], axis=0).reset_index(drop=True)
            
    if pd_final.empty:
        print('No existen variables con valores nulos')
        
    return pd_final

#####

def cramers_v(confusion_matrix):
    """ 
    calculate Cramers V statistic for categorial-categorial association.
    uses correction from Bergsma and Wicher,
    Journal of the Korean Statistical Society 42 (2013): 323-328
    
    confusion_matrix: tabla creada con pd.crosstab()
    
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


###----------------------------------------- Funciones Propias -----------------------------------------------

def tipos_vars(df=None, show=True):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función tipos_vars:
    ----------------------------------------------------------------------------------------------------------
    - Funcionamiento:
        La función recibe como argumento un dataframe, analiza cada una de sus variables y muestra
        en pantalla el listado, categorizando a cada una como "categoric","bool" o "numeric". Para
        variables categóricas y booleanas se muestra el listado de categorías. Si son numéricas solo
        se informa el Rango y la Media de la variable.
        Además, luego de imprimir la información comentada, la función devuelve 3 listas, cada una
        con los nombres de las variables pertenecientes a cada grupo ("bools", "categoric" y "numeric").
        El orden es: 1. bools, 2. categoric, 3. numeric.
    - Inputs:
        -- df: DataFrame de Pandas a analizar
        -- show: Argumento opcional, valor por defecto True. Si show es True, entonces se mostrará la
        información básica de con cada categoría. Si es False, la función solo devuelve las listas con
        los nombres de las variables según su categoría.
    - Return:
        -- list_bools: listado con el nombre de las variables booleanas encontradas
        -- list_cat: listado con el nombre de las variables categóricas encontradas
        -- list_num: listado con el nombre de las variables numéricas encontradas
    '''
    # Realizo una verificación por si no se introdujo ningún DF
    if df is None:
        print(u'No se ha especificado un DF para la función')
        return None
    
    # Genero listas vacías a rellenar con los nombres de las variables por categoría
    list_bools = []
    list_cat = []
    list_num = []
    
    # Analizo variables, completo las listas e imprimo la información de cada variable en caso de que el Show no se desactive
    for i in df.columns:
        if len(df[i].unique()) <= 2 and df[i].dtype=='int64':
            list_bools.append(i)
            if show:
                print(f"{i} {colored('(boolean)','blue')} :  {df[i].unique()}")
        elif len(df[i].unique()) < 50:
            list_cat.append(i)
            if show:
                print(f"{i} {colored('(categoric)','red')} (\033[1mType\033[0m: {df[i].dtype}): {df[i].unique()}")
        else:
            list_num.append(i)
            if show:
                print(f"{i} {colored('(numeric)','green')} : \033[1mRange\033[0m = [{df[i].min():.2f} to {df[i].max():.2f}], \033[1mMean\033[0m = {df[i].mean():.2f}")
    
    # Finalmente devuelvo las listas con los nombres de las variables por cada categoría
    return list_bools,list_cat,list_num

#####

def dame_info(var_list, df, df_info, only_desc=False, names_col='Variable', descrip_col='Description'):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función dame_info:
    ----------------------------------------------------------------------------------------------------------
    - Funcionamiento: Función bastante específica para el trabajo de este DF de fraude, aunque bastante
    necesaria. Útil para llamarla mientras se analiza el comportamiento de X variables para obtener contexto
    sobre éstas. Recibe una lista de variables de las que se desea obtener una breve descripción, el DF en el
    que se encuentran dichas variables como parámetros obligatorios, y el DF con la descripción de las
    variables. Luego busca y trae las descripciones de cada variable solicitada desde el diccionario de datos,
    además de mostrar la cantidad de nulls y los valores que puede tomar cada variable.
    - Inputs:
        -- var_list: Listado de variables cuya descripción se necesita imprimir
        -- df: DataFrame de Pandas en donde se encuentran las variables a analizar
        -- only_desc: Permite elegir ver únicamente la descripción en texto de la variable sin datos extra
        -- df_info: DataFrame de Pandas en el que se encuentre el diccionario de datos
        -- names_col: Nombre de la columna de df_info en donde se encuentran los nombres de las variables a
        buscar
        -- descrip_col= Nombre de la columna del df_indo en donde se encuentra la descripción de cada variable
    - Return: No hay return, solo imprime cada descripción de cada variable solicitada.
    '''    
    
    for i,n in zip(var_list,range(1,(len(var_list)+1))):
        if i not in df.columns.values:
            print(f'\033[1m {n}. {i}\033[0m : Variable no encontrada en el DF')
        else:
            if only_desc:
                print(f'\033[1m {n}.',i,'\033[0m',':',df_info.set_index(names_col).loc[i,descrip_col],'')
            else:
                print(f'\033[1m {n}.',i,'\033[0m',':',df_info.set_index(names_col).loc[i,descrip_col],'')
                print(f'- Nulls: {df[i].isna().sum()}')
                if df[i].dtype in ['string','object']:
                    print('- Values:',df[i].unique(),'\n')
                elif (df[i].dtype in ['int','int64','bool','float','float64']) & (len(df[i].unique())<=10):
                    print('- Values:',df[i].sort_values().unique(),'\n')
                elif (df[i].dtype in ['float','float64','int','int64']) &  (len(df[i].unique())>10):
                    print(f"- Range = [{df[i].min():.2f} to {df[i].max():.2f}], Mean = {df[i].mean():.2f}\n")
                    
#####

def corr_cat(df,target=None,target_transform=False):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función corr_cat:
    ----------------------------------------------------------------------------------------------------------
    - Funcionamiento:
        La función recibe como un dataframe, detecta las variables categóricas y calcula una especie de
        matriz de correlaciones mediante el uso del estadístico Cramers V. En la función se incluye la
        posibilidad de que se transforme a la variable target a string si no lo fuese y que se incluya en la
        lista de variables a analizar. Esto último  puede servir sobre todo para casos en los que la variable
        target es un booleano o está codificada.
    - Inputs:
        -- df: DataFrame de Pandas a analizar
        -- target: String con nombre de la variable objetivo
        -- target_transform: Transforma la variable objetivo a string para el procesamiento y luego la vuelve
        a su tipo original.
    - Return:
        -- corr_cat: matriz con los Cramers V cruzados.
    '''
    df_cat_string = list(df.select_dtypes('category').columns.values)
    
    if target_transform:
        t_type = df[target].dtype
        df[target] = df[target].astype('string')
        df_cat_string.append(target)

    corr_cat = []
    vector = []

    for i in df_cat_string:
        vector = []
        for j in df_cat_string:
            confusion_matrix = pd.crosstab(df[i], df[j])
            vector.append(cramers_v(confusion_matrix.values))
        corr_cat.append(vector)

    corr_cat = pd.DataFrame(corr_cat, columns=df_cat_string, index=df_cat_string)
    
    if target_transform:
        df_cat_string.pop()
        df[target] = df[target].astype(t_type)

    return corr_cat

#####
    
def double_plot(df, col_name, is_cont, target):
    """
    ----------------------------------------------------------------------------------------------------------
    Función double_plot:
    ----------------------------------------------------------------------------------------------------------
     Me inspiré en una función de la cátedra para crear mi propia función de gráficos para cada variable
     según su tipo.
     - Funcionamiento:
        La función recibe como un dataframe y la variable a graficar. En base a si es continua o
        si es categórica, se mostrarán dos gráficos de un tipo o de otro
            - Para variables continuas se muestra un histograma y un boxplot en base al Target.
            - Para variables categóricas se muestran dos barplots, uno con la variable sola y la otra en base
            al target. Además, este segundo aplica una transformación logarítmica a la escala del eje y. Esto
            está pensado especialmente para este dataset, debido a que el desbalanceo es tan grande que casi
            no se llegan a percibir los valores 1 en la variable objetivo. Por eso para diferenciar se grafica
            de esta manera.
    - Inputs:
        -- df: DataFrame de Pandas a analizar
        -- col_name: Columna del DF a graficar
        -- is_cont: True o False. Determina si la variable a graficar es continua o no
        -- target: Variable objetivo del DF
    """
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,3), dpi=90)
    
    if is_cont:
        sns.histplot(df[col_name], kde=False, ax=ax1, color='limegreen')
    else:
        barplot_df = pd.DataFrame(df[col_name].value_counts()).reset_index()
        sns.barplot(barplot_df, x=col_name, y='count', palette='YlGnBu', ax=ax1)
    ax1.set_xlabel(col_name)
    ax1.set_ylabel('Count')
    ax1.set_title(col_name)
    plt.xticks(rotation = 90)

    if is_cont:
        sns.boxplot(data=df, x=col_name, y=df[target].astype('string'), palette=['deepskyblue','crimson'], ax=ax2)
        ax2.set_ylabel('')
        ax2.set_title(col_name + ' by '+target)
    else:
        barplot2_df = df.groupby(col_name)[target].value_counts(normalize=True).to_frame('proportion').reset_index()
        sns.barplot(data=barplot2_df, x=col_name, y='proportion', hue=barplot2_df[target].astype('string'), palette=['deepskyblue','crimson'], ax=ax2)
        plt.yscale('log')
        ax2.set_ylabel('Proportion')
        
        #Prueba descartada:
        #barplot2_df = df.pivot_table(columns=[target,col_name], aggfunc='count').iloc[0,:].reset_index()
        #sns.barplot(data=barplot2_df, x=col_name, y=np.log(barplot2_df.iloc[:,2]), hue=barplot2_df[target].astype('string'), palette=['deepskyblue','crimson'], ax=ax2)
        #ax2.set_ylabel('Log(Count)')
        
        ax2.set_title(target)
        plt.xticks(rotation = 90)
    ax2.set_xlabel(col_name)
    
    plt.tight_layout()


