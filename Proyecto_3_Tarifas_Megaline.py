# Cargar todas las librerías
import pandas as pd
import numpy as np
from scipy import stats as st
import math as mt
from matplotlib import pyplot as plt


# %% [markdown]
# ## 1.2 Cargar datos <a id='carga_datos'></a>

# %%
# Carga los archivos de datos en diferentes DataFrames
df_users = pd.read_csv('files/datasets/megaline_users.csv')
df_calls = pd.read_csv('files/datasets/megaline_calls.csv')
df_messages = pd.read_csv('files/datasets/megaline_messages.csv')
df_internet = pd.read_csv('files/datasets/megaline_internet.csv')
df_plans = pd.read_csv('files/datasets/megaline_plans.csv')


# %% [markdown]
# ## 2 Preparar los datos <a id='preparar_datos'></a>

# %% [markdown]
# ## 2.1 Tarifas <a id='tarifas'></a>

# %%
# Imprime la información general/resumida sobre el DataFrame de las tarifas
df_plans.info()


# %%
# Imprime una muestra de los datos para las tarifas
df_plans.head()


# %% [markdown]
# <span style="color:darkgreen">
# El Dataframe tiene un total de 8 columnas con 2 filas solamente, no hay datos ausentes y tampoco duplicados. Se puede concluir con lo anterior, debido a que es un DataFrame pequeño y se puede observar  simple vista.
# </span>

# %% [markdown]
# ## 2.2 Corregir datos <a id='corregir_datos'></a>

# %% [markdown]
# <span style="color:darkgreen">
# No es necesario hacer correciones al DataFrame
# </span>

# %% [markdown]
# ## 2.3 Enriquecer los datos <a id='enriquecer_datos'></a>

# %% [markdown]
# <span style="color:darkgreen">
# Se cambia la columna de `mb_per_month_included` a Gigabytes.  
#     
# También se cambia el nombre de la columna de `mb_per_month_included` a `gb_per_month_included`
# </span>

# %%
# La columna 'mb_per_month_included' se divide entre 1024 para transformar a gigabytes
# Así quedan 15 y 30 gb para el plan Surf y el plan Ultimate, respectivamente
df_plans['mb_per_month_included'] = df_plans['mb_per_month_included'] / 1024
# Se renombra la columna 'mb_per_month_included'
df_plans = df_plans.rename(columns= {'mb_per_month_included':'gb_per_month_included'})
df_plans


# %% [markdown]
# ## 2.4 Usuarios/as <a id='usuarios'></a>

# %%
# Imprime la información general/resumida sobre el DataFrame de usuarios
df_users.info()


# %%
# Se imprime una muestra de datos para usuarios
df_users.head()

# %%
# se usa el método describe para una exploración de los datos (estadísticas descriptivas)
df_users.describe()

# %%
# Se hace una exploración de los datos de la columna 'plan' con describe()
df_users['plan'].describe()

# %%
# Se comprueba si hay datos nulos
df_users.isna().sum()

# %%
# Se comprueba si hay datos duplicados
df_users.duplicated().sum()

# %% [markdown]
# <span style="color:darkgreen"> El Dataframe de usuarios tiene 500 filas y 8 columnas, las columnas `reg_date` y `churn_date` no tiene el tipo de datos de fecha, es necesario cambiar el tipo de dato de estas columnas, además, la columna `churn_date` tiene valores ausentes, pero estos valores ausentes significan que la tarifa se estaba usando cuando fue extraída esta base de datos, por lo que no se eliminarán o sustituiran con otro valor.  
# La edad promedio de los usuarios o usuarias es de 45 años, el cliente más jóven tiene 18 años, mientras que el cliente de mayor edad tiene 75 años.  
# El plan Surf es el más frecuente entre los usuarios o usuarias.  
# También, la columna `city` tiene la ciudad y el estado del usuario, ésta columna se puede dividir para tener una columna únicamente con la ciudad y otra con el estado.
# </span>

# %% [markdown]
# ### 2.4.1 Corregir los datos <a id='corregir_usuarios'></a>

# %%
# Se cambia el tipo de dato de la columna 'reg_date' y 'churn_date' con el método to_datetime()
df_users['reg_date'] = pd.to_datetime(df_users['reg_date'])
df_users['churn_date'] = pd.to_datetime(df_users['churn_date'])

# %%
# Con el atributo dtypes se verifica el cambio de tipos de datos de las columnas 'reg_date' y 'churn_date'
df_users.dtypes

# %% [markdown]
# ### 2.4.2 Enriquecer los datos <a id='enriquecer_usuarios'></a>

# %%
# Ahora se divide la columna 'city' para separar la ciudad y el estado
# También se crea una nueva columna para el/los estados
# Se emplea str.split() para realizar dicha tarea
df_users[['city', 'state']] = df_users.city.str.split(', ', expand= True)

# %%
# Se muestran los 3 primeros valores del DataFrame
df_users.head()

# %% [markdown]
# <span style="color:darkgreen"> En la columna `state` se observa que el nombre de los estados está acompañado con **MSA**, para saber el significado de dicho texto sería necesario preguntar al equipo que capturó los datos, sin embargo, para fines de este proyecto no es relevante y sólo nos interesa el estado, por lo tanto se eliminará de la columna.
# </span>

# %%
# Para reemplazar MSA de la columna 'state' se hace mediante str.replace(' MSA','')
df_users['state'] = df_users['state'].str.replace(' MSA','')

# %%
# Se imprime una muestra de datos del DataFrame de los usuarios
df_users.head()

# %% [markdown]
# <div class="alert alert-block alert-info">
# <b> </b> <a class="tocSkip"></a>  
# 
# Para los datos nulos en la columna `churn_date` se decidió no completarlos o llenarlos con otra información, porque son clientes aún estaba usando el servicio. Por tanto, estos valores nulos significan que los usuarios aún están usando el servicio.
# </div>
# 

# %% [markdown]
# <span style="color:green"> Ahora ya se tiene la ciudad y el estado en columnas separadas.
# </span>

# %% [markdown]
# ## 3 Llamadas <a id='llamadas'></a>

# %%
# Imprime la información general/resumida sobre el DataFrame de las llamadas
df_calls.info()


# %%
# Imprime una muestra de datos para las llamadas
df_calls.head()

# %%
# se usa el método describe para una exploración de los datos (estadísticas descriptivas)
df_calls.describe()

# %%
# Se verifica si hay valores nulos y duplicados
df_calls.isna().sum()

# %%
df_calls.duplicated().sum()

# %% [markdown]
# <span style="color:green"> Se tienen 4 columnas con 137735 filas, la columna `id` tiene un tipo de dato *object*, pero está combinado con el **user_id**, por lo tanto se dejarán así esos datos. La columna `call_date` tiene un tipo de dato *object* por lo tanto se cambiará a tipo fecha (datetime). Con los métodos `isna().sum()` y `duplicated().sum()` se verificó si hay valores nulos en el DataFrame y no los hay en este caso.
# </span>

# %% [markdown]
# ### 3.1 Corregir los datos <a id='corregir_llamadas'></a>

# %%
# Se cambia el tipo de dato de la columna 'call_date' con el método to_datetime()
df_calls['call_date'] = pd.to_datetime(df_calls['call_date'])

# %%
# Con el atributo dtypes se verifica el cambio de tipos de datos de las columnas 'reg_date' y 'churn_date'
df_calls.dtypes

# %% [markdown]
# ### 3.2 Enriquecer los datos <a id='enriquecer_llamadas'></a>

# %% [markdown]
# <span style="color:green"> Se crea una columna para el mes en que se hizo la llamada.
# </span>

# %%
# Se crea la columna 'call_month' y se usa el atributo df.month sobre la columna 'call_date' para extraer únicamente el mes
df_calls['call_month'] = df_calls['call_date'].dt.month

# %%
df_calls.head()

# %%
# Hay llamadas con una duración de cero, por lo tanto se conservan los datos que sean diferentes de cero en la duración
df_calls = df_calls[~(df_calls['duration'] == 0.0)]
df_calls.head()

# %%
# Los minutos se redondean al entero superior inmediato con ceil() de numpy
df_calls['duration'] = df_calls['duration'].apply(np.ceil)
df_calls.head()

# %% [markdown]
# ## 4 Mensajes <a id='mensajes'></a>

# %%
# Imprime la información general/resumida sobre el DataFrame de los mensajes
df_messages.info()


# %%
# Imprime una muestra de datos para los mensajes
df_messages.head()

# %%
# Se verifica si hay valores nulos y duplicados
df_messages.isna().sum()

# %%
df_messages.duplicated().sum()

# %% [markdown]
# <span style="color:green"> Se tienen 3 columnas con 76051 filas, la columna `id` tiene un tipo de dato *object*, también está combinado con el **user_id**, por lo tanto se dejarán así esos datos. La columna `message_date` tiene un tipo de dato *object* por lo tanto se cambiará a tipo fecha (datetime). Con los métodos `isna().sum()` y `duplicated().sum()` se verificó si hay valores nulos en el DataFrame y no los hay en este caso.
# </span>

# %% [markdown]
# ### 4.1 Corregir los datos <a id='corregir_mensajes'></a>

# %%
# Se cambia el tipo de dato de la columna 'message_date' con el método to_datetime()
df_messages['message_date'] = pd.to_datetime(df_messages['message_date'])

# %%
# Con el atributo dtypes se verifica el cambio de tipos de datos 
df_messages.dtypes

# %% [markdown]
# ### 4.2 Enriquecer los datos <a id='enriquecer_mensajes'></a>

# %% [markdown]
# <span style="color:green"> Se crea una columna para el mes en que se envío el mensaje.
# </span>

# %%
# Se crea la columna 'message_month' y se usa el atributo df.month sobre la columna 'message_date' para extraer únicamente el mes
df_messages['message_month'] = df_messages['message_date'].dt.month

# %%
df_messages.head()

# %% [markdown]
# ## 5 Internet <a id='internet'></a>

# %%
# Imprime la información general/resumida sobre el DataFrame de internet
df_internet.info()

# %%
# Imprime una muestra de datos para el tráfico de internet
df_internet.head()

# %%
# se usa el método describe para una exploración de los datos (estadísticas descriptivas)
df_internet.describe()

# %%
# Se verifica si hay valores nulos y duplicados
df_internet.isna().sum()

# %%
df_internet.duplicated().sum()

# %%
# Se verifica si hay usuarios que no usaron internet
# Se filtra el DataFrame
df_internet[df_internet['mb_used'] == 0].head()

# %% [markdown]
# <span style="color:green"> Se tienen 4 columnas con 104825 filas, la columna `id` tiene un tipo de dato *object*, también está combinado con el **user_id**, por lo tanto se dejarán así esos datos. La columna `session_date` tiene un tipo de dato *object* por lo tanto se cambiará a tipo fecha (datetime). Con los métodos `isna().sum()` y `duplicated().sum()` se verificó si hay valores nulos en el DataFrame y no los hay en este caso. En promedio los usuarios o usuarias usan 366 mb.
# </span>

# %% [markdown]
# ### 5.1 Corregir los datos <a id='corregir_internet'></a>

# %%
# Se cambia el tipo de dato de la columna 'message_date' con el método to_datetime()
df_internet['session_date'] = pd.to_datetime(df_internet['session_date'])

# %%
# Con el atributo dtypes se verifica el cambio de tipos de datos 
df_internet.dtypes

# %% [markdown]
# ### 5.2 Enriquecer los datos <a id='enriquecer_internet'></a>

# %% [markdown]
# <span style="color:green"> Se crea una columna para el mes de la sesión.
# </span>

# %%
# Se crea la columna 'session_month' y se usa el atributo df.month sobre la columna 'session_date' para extraer únicamente el mes
df_internet['session_month'] = df_internet['session_date'].dt.month

# %%
# Se conservan los datos donde los usuarios si tuvieron consumo de interntet
# Se filtra el DataFrame
df_internet = df_internet[~(df_internet['mb_used'] == 0)]
df_internet.head()

# %%
# Se Transforman los megabytes a gigabytes
df_internet['mb_used'] = df_internet['mb_used']/1024
# se renombra la columna 
df_internet = df_internet.rename(columns= {'mb_used':'gb_used'})

# %%
# Se redondea al entero superior inmediato
df_internet['gb_used'] = df_internet['gb_used'].apply(np.ceil)
df_internet.head()

# %% [markdown]
# ## 6 Estudiar las condiciones de las tarifas <a id='condiciones_tarifa'></a>

# %%
# Imprime las condiciones de la tarifa y asegúrate de que te quedan claras
df_plans

# %%
# Calcula el número de llamadas hechas por cada usuario al mes. Guarda el resultado.
# Se hace un merge de los DataFrames df_users y df_calls para complementar la información
merge_users_calls = df_users.merge(df_calls, on= 'user_id')
merge_users_calls.head()

# %%
# Se agrupan las llamadas por usuarios y mes, se contabilizan las llamadas hechas con count()
# El Series resultante se asigna a num_calls
num_calls = merge_users_calls.groupby(['user_id', 'plan', 'call_month'])['duration'].count()
num_calls

# %%
# Calcula la cantidad de minutos usados por cada usuario al mes. Guarda el resultado.
# Se agrupan las llamadas por usuarios y mes, se suman los minutos de las llamadas hechas con sum()
# El Series resultante se asigna a sum_minutes
sum_minutes = merge_users_calls.groupby(['user_id', 'plan', 'call_month'])['duration'].sum()
sum_minutes

# %%
# Calcula el número de mensajes enviados por cada usuario al mes. Guarda el resultado.
# Se hace un merge de los DataFrames df_users y df_messages para complementar la información
merge_users_messages = df_users.merge(df_messages, on= 'user_id')
merge_users_messages.head()


# %%
# Se agrupan los mensajes por usuarios y mes, se contabilizan los mensajes enviados con count() de la columna 'id'
# El Series resultante se asigna a sum_minutes
count_messages = merge_users_messages.groupby(['user_id','plan', 'message_month'])['id'].count()
count_messages

# %%
# Calcula el volumen del tráfico de Internet usado por cada usuario al mes. Guarda el resultado.
# Se hace un merge de los DataFrames df_users y df_internet para complementar la información
merge_users_internet = df_users.merge(df_internet, on= 'user_id')
merge_users_internet.head()

# %%
# Se agrupan el el volumen del tráfico de Internet usado por usuarios y mes, se contabilizan se suman los datos con sum() de la columna 'mb_used'
# El Series resultante se asigna a sum_internet
sum_internet = merge_users_internet.groupby(['user_id', 'plan', 'session_month'])['gb_used'].sum()
sum_internet

# %%
# Fusiona los datos de llamadas, minutos, mensajes e Internet con base en user_id y month
# Los Series num_calls, sum_minutes, count_message y sum_internet se concatenan con concat, con axis= 'columns' para asegurarnos de que se combinaran como columnas
calls_mints_internet = pd.concat([num_calls, sum_minutes, count_messages, sum_internet], axis='columns')
# Se renombran las columnas
calls_mints_internet.columns = ['total_calls', 'sum_minutes', 'count_messages', 'vol_internet']
# con reset_index() se reinicia el índice del DataFrame calls_mints_internet
calls_mints_internet.reset_index(inplace= True)

# %%
# Se imprime una muestra de datos del DataFrame calls_mints_internet
calls_mints_internet.head()

# %% [markdown]
# <span style="color:green"> Después de reiniciar el índice del DataFrame `calls_mints_internet`, el nombre de la columna del mes se asigno el nombre de `level_2` el cual se cambiará por `month` con el método `rename`. 
# </span>

# %%
# Se cambia el nombre de la columna 'level_2' por 'month'
calls_mints_internet = calls_mints_internet.rename(columns= {'level_2':'month'})
# Se imprime una muestra de datos del DataFrame calls_mints_internet
calls_mints_internet.head()

# %%
# Se crea una función para asignar la tarifa con base al plan del usuario
def monthly_pay(plan):
    if plan == 'ultimate':
        return 70
    else:
        return 20

# %%
# Añade la información de la tarifa
# Se crea la columna 'monthly_pay' donde se asigna la tarifa con la función que se creó 'monthly_pay()'
# con el método apply() se aplica a la columna 'monthly_pay'
calls_mints_internet['monthly_pay'] = calls_mints_internet['plan'].apply(monthly_pay)


# %%
# Se imprime una muestra de datos del DataFrame calls_mints_internet, esta ocasión usando el método sample()
calls_mints_internet.sample(10)

# %%
# se verifican los datos nulos
calls_mints_internet.isna().sum()

# %% [markdown]
# <span style="color:green"> Al fusionar los datos de las llamadas, minutos, mensajes e Internet, en algunos casos algunos valores son nulos (`NaN`), ya que en algunos casos en determinado mes el usuario o usuaria sólo usó los minutos para llamadas, mensajes o Internet, los otros servicios no los usó, los cuales se reemplazarán con 0.
# </span>

# %%
calls_mints_internet.fillna(0, inplace= True)

# %% [markdown]
# <span style="color:green"> Para calcular los ingresos mensuales por usuario o usuaria, se crea un función para  calcular los el costo de los minutos extra, el costo de los mensajes extra y los gibabytes extra, es decir tres funciones para cada cálculo.  Y otra función será `total_pay()` lo que hará es sumar el costo extra de los minutos, mensajes e internet a la tarifa mensual del plan.
# </span>

# %%
def extra_minutes(minutes_used, plan_name, plan_limit):
    '''
    Función para calcular el costo de los minutos extra de acuerdo al plan. Si el cliente usa menos o la misma
    cantidad de minutos de llamada no se le cobra nada extra, pero si se exceden los límites del 
    paquete se calculan los recursos extra que se hayan excedido y se devuelve ese valor/costo extra.
    '''
    if plan_name == 'ultimate':
        if minutes_used <= plan_limit:
            return 0
        else:
            minutes_extra = minutes_used - plan_limit
            return minutes_extra * 0.01
    else:
        if minutes_used <= plan_limit:
            return 0
        else:
            minutes_extra = minutes_used - plan_limit
            return minutes_extra * 0.03
        
def extra_messages(messages_used, plan_name, plan_limit):
    '''
    Función para calcular el costo de los mensajes extra de acuerdo al plan. Si el cliente usa menos o la misma
    cantidad de mensajes no se le cobra nada extra, pero si se exceden los límites del 
    paquete se calculan los recursos extra que se hayan excedido y se devuelve ese valor/costo extra.
    '''
    if plan_name == 'ultimate':
        if messages_used <= plan_limit:
            return 0
        else:
            messages_extra = messages_used - plan_limit
            return messages_extra * 0.01
    else:
        if messages_used <= plan_limit:
            return 0
        else:
            messages_extra = messages_used - plan_limit
            return messages_extra * 0.03
        
def extra_internet(internet_used, plan_name, plan_limit):
    '''
    Función para calcular el costo del internet o gigas extra de acuerdo al plan. Si el cliente usa menos o la misma
    cantidad de gigabytes no se le cobra nada extra, pero si se exceden los límites del 
    paquete se calculan los recursos extra que se hayan excedido y se devuelve ese valor/costo extra.
    '''
    if plan_name == 'ultimate':
    
        if internet_used <= plan_limit:
            return 0
        else:
            internet_extra = internet_used - plan_limit
            return internet_extra * 7
    else:
        if internet_used <= plan_limit:
            return 0
        else:
            internet_extra = internet_used - plan_limit
            return internet_extra * 10

def total_pay(df):
    '''
    Función que calcula el total del pago, se suman los recursos extra a la tarifa base de acuerdo al plan.
    '''
    plan = df['plan']
    minutos = df['sum_minutes']
    mensajes = df['count_messages']
    internet = df['vol_internet']

    if plan == 'ultimate':
        
        minutes_extra_usd = extra_minutes(minutos, plan, 3000) # costo de los minutos extra
        messages_extra_usd = extra_messages(mensajes, plan, 1000) # costo de los mensajes extra
        internet_extra_usd = extra_internet(internet, plan, 30) # costo de los gigas extra
        total_pay = 70 + minutes_extra_usd + messages_extra_usd + internet_extra_usd # pago total

    else:
        minutes_extra_usd = extra_minutes(minutos, plan, 500)
        messages_extra_usd = extra_messages(mensajes, plan, 50)
        internet_extra_usd = extra_internet(internet, plan, 15)
        total_pay = 20 + minutes_extra_usd + messages_extra_usd + internet_extra_usd

    return total_pay

# %%
# Calcula el ingreso mensual para cada usuario
# se crea una columna nueva 'usd_total_pay'
# se emplea el método apply() con la función total_pay() sobre el DataFrame 'calls_mints_internet'
# apply() se usa con axis= 1 para que pase los valores de fila de la función
calls_mints_internet['usd_total_pay'] = calls_mints_internet.apply(total_pay, axis= 1)

# %%
# Se imprime una muestra de datos del DataFrame calls_mints_internet
calls_mints_internet.head()

# %% [markdown]
# <span style="color:green"> Ahora el DataFrame `calls_mints_internet` tiene los ingresos mensuales por usuario, esta tarea fue más facil de realizar debido a la función `total_pay()` que se creó.
# </span>

# %% [markdown]
# ## 7 Estudia el comportamiento de usuario <a id='usuario_comportamiento'></a>

# %% [markdown]
# ### 7.1 Llamadas <a id='llamadas_comportamiento'></a>

# %%
# Compara la duración promedio de llamadas por cada plan y por cada mes. Traza un gráfico de barras para visualizarla. 
# Se filtra el DataFrame sólo para el plan surf y se guarda en surf_plan
surf_plan = calls_mints_internet[calls_mints_internet['plan'] == 'surf']
surf_plan.head()

# %%
# Se filtra el DataFrame sólo para el plan ultimate y se guarda en ultimate_plan
ultimate_plan = calls_mints_internet[calls_mints_internet['plan'] == 'ultimate']
ultimate_plan.head()

# %%
# Se agrupa por mes el DataFrame 'surf_plan' y se calcula el promedio de los minutos de las llamadas (columna 'sum_minutes')
# EL resultado se asigna a 'month_surf_mean'
month_surf_mean = surf_plan.groupby('month')['sum_minutes'].mean()
month_surf_mean

# %%
# Se agrupa por mes el DataFrame 'month_ultimate_mean' y se calcula el promedio de los minutos de las llamadas (columna 'sum_minutes')
# EL resultado se asigna a 'month_ultimate_mean'
month_ultimate_mean = ultimate_plan.groupby('month')['sum_minutes'].mean()
month_ultimate_mean

# %%
# se concatenan los Series 'month_surf_mean' y 'month_ultimate_mean'
surf_ultimate_mean = pd.concat([month_surf_mean, month_ultimate_mean], axis='columns')
# Se renombran las columnas
surf_ultimate_mean.columns = ['min_mean_surf', 'min_mean_ultimate']
# con reset_index() se reinicia el índice del DataFrame 'surf_ultimate_mean'
surf_ultimate_mean.reset_index(inplace= True)
surf_ultimate_mean

# %%
# Se grafica la duración promedio de llamadas por cada plan y por cada mes
surf_ultimate_mean.plot(x= 'month',
                       kind= 'bar',
                       figsize= [12,8],
                       fontsize= 12,
                       rot= 360
                       )
plt.title('Duración promedio de llamadas por plan y mes', fontsize=15)
plt.xlabel('Mes', fontsize=15)
plt.ylabel('Duración promedio de llamadas (minutos)', fontsize=15)
plt.legend(['Plan Surf', 'Plan Ultimate'], fontsize= 14)
plt.show()

# %%
# Compara el número de minutos mensuales que necesitan los usuarios de cada plan. Traza un histograma.
# A partir de los DataFrames 'surf_plan' y 'ultimate_plan' se crean sus respectivos histogramas en el mismo gráfico
surf_plan['sum_minutes'].plot(kind='hist',
                              bins= 60, 
                              alpha= 0.6,
                              figsize= [12,8]
                             )
ultimate_plan['sum_minutes'].plot(kind='hist',
                              bins= 60, 
                              alpha= 0.6,
                              figsize= [12,8]
                             )


plt.title('Número de minutos mensuales por plan', fontsize= 14) # para colocar un título
plt.ylabel('Frecuencia', fontsize= 14) # Para nombrar el eje y
plt.legend(['Plan Surf', 'Plan Ultimate'], fontsize= 12) # para colocar las leyendas del gráfico
plt.show()


# %%
print('Plan Surf:')
# Se calcula la varianza con var() de la duración mensual de llamadas.
print('La varianza de los minutos es:', round(surf_plan['sum_minutes'].var(), 1))
# se calcula la moda de los datos con mode() de la duración mensual de llamadas.
print('La moda de los minutos es:', surf_plan['sum_minutes'].mode()[0])
# se emplea el método describe() para calcular los estadísticos descriptivos como la media, mediana, desviación estándar
print(surf_plan['sum_minutes'].describe())

# %%
print('Plan Ultimate:')
# Se calcula la varianza con var() de la duración mensual de llamadas.
print('La varianza de los minutos es:', round(ultimate_plan['sum_minutes'].var(), 1))
# se calcula la moda de los datos con mode() de la duración mensual de llamadas.
print('La moda de los minutos es:', ultimate_plan['sum_minutes'].mode()[0])
# se emplea el método describe() para calcular los estadísticos descriptivos como la media, mediana, desviación estándar
print(ultimate_plan['sum_minutes'].describe())

# %%
# Traza un diagrama de caja para visualizar la distribución de la duración mensual de llamadas
calls_mints_internet.boxplot(column= 'sum_minutes',
                             by= 'plan',
                             fontsize= 12,
                             grid= False,
                             figsize= [12,8]
                             )
plt.title('Distribución de la duración mensual de llamadas por plan', fontsize= 14) # para colocar un título
plt.xlabel('Plan', fontsize= 14) # Para nombrar el eje x
plt.ylabel('Duración promedio de llamadas (minutos)', fontsize= 14) # Para nombrar el eje y
plt.show()


# %% [markdown]
# <span style="color:green">  **Conclusiones:**  
# • En el gráfico de barras se puede ver que en febrero (mes 2) hay un mayor número de minutos para los usuarios del plan Ultimate y en junio (mes 6) para los usuarios del plan Surf, para el resto de los meses no se observa una diferencia muy grande.  
# • Con base en el histograma se observa que para mabos planes se observa una distribución similar, pero hay más datos para el plan surf. 
# • Al calcular la media para ambos planes es muy similar y ambas tienen una varianza muy grande, esto quiere decir que los datos están muy separados de la media. Lo mismo sucede con la desviación estándar, que es de 234.4 y 240.5 para el plan Surf y Ultimate, respectivamente, lo cual indica que los datos son muy hetegéneos.  
# • Con el diagrama de caja podemos observar valores atípicos, ya que hay valores mayores que el valor máximo (1.5 veces el rango intercuartílico (IQR) del cuartil Q3). Lo anterior sucede para ambos planes. Para el plan Surf el valor máximo es de 1510 y para el plan Ultimate 1369.
# </span>

# %% [markdown]
# ### 7.2 Mensajes <a id='mensajes_comportamiento'></a>

# %%
# Comprara el número de mensajes que tienden a enviar cada mes los usuarios de cada plan
# Se agrupa por mes el DataFrame 'surf_plan' y se calcula el promedio de los mensajes (columna 'count_messages')
# EL resultado se asigna a 'month_surf_mean_messages'
month_surf_mean_messages = surf_plan.groupby('month')['count_messages'].mean()
month_surf_mean_messages


# %%
# Se agrupa por mes el DataFrame 'ultimate_plan' y se calcula el promedio de los mensajes (columna 'count_messages')
# EL resultado se asigna a 'month_surf_mean_messages'
month_ultimate_mean_messages = ultimate_plan.groupby('month')['count_messages'].mean()
month_ultimate_mean_messages

# %%
# se concatenan los Series 'month_surf_mean_messages' y 'month_ultimate_mean_messages'
surf_ultimate_mean_messages = pd.concat([month_surf_mean_messages, month_ultimate_mean_messages], axis='columns')
# Se renombran las columnas
surf_ultimate_mean_messages.columns = ['messages_mean_surf', 'messages_mean_ultimate']
# con reset_index() se reinicia el índice del DataFrame 'surf_ultimate_mean_messages'
surf_ultimate_mean_messages.reset_index(inplace= True)
surf_ultimate_mean_messages


# %%
# Se grafica la cantidad promedio de mensajes por cada plan y por cada mes
surf_ultimate_mean_messages.plot(x= 'month',
                       kind= 'bar',
                       figsize= [12,8],
                       fontsize= 12, 
                       rot= 360
                       )
plt.title('Mensajes promedio por plan y mes', fontsize=15)
plt.xlabel('Mes', fontsize=15)
plt.ylabel('Cantidad promedio de mensajes', fontsize=15)
plt.legend(['Plan Surf', 'Plan Ultimate'], fontsize= 14)
plt.show()

# %%
# A partir de los DataFrames 'surf_plan' y 'ultimate_plan' se crean sus respectivos histogramas en el mismo gráfico
surf_plan['count_messages'].plot(kind='hist',
                              bins= 40, 
                              alpha= 0.6,
                              figsize= [14,8],
                              xticks= range(0,300,20)
                             )
ultimate_plan['count_messages'].plot(kind='hist',
                              bins= 40, 
                              alpha= 0.6,
                              figsize= [14,8],
                              xticks= range(0,300,20)
                             )


plt.title('Cantidad de mensajes por plan', fontsize= 14) # para colocar un título
plt.ylabel('Frecuencia', fontsize= 14) # Para nombrar el eje y
plt.legend(['Plan Surf', 'Plan Ultimate'], fontsize= 12) # para colocar las leyendas del gráfico
plt.show()

# %%
print('Plan Surf:')
# Se calcula la varianza con var() de la cantidad de mensajes.
print('La varianza de los minutos es:', round(surf_plan['count_messages'].var(), 1))
# se calcula la moda de los datos con mode() de la cantidad de mensajes
print('La moda de los minutos es:', surf_plan['count_messages'].mode()[0])
# se emplea el método describe() para calcular los estadísticos descriptivos como la media, mediana, desviación estándar
print(surf_plan['count_messages'].describe())

# %%
print('Plan Ultimate:')
# Se calcula la varianza con var() de la cantidad de mensajes.
print('La varianza de los minutos es:', round(ultimate_plan['count_messages'].var(), 1))
# se calcula la moda de los datos con mode() de la cantidad de mensajes
print('La moda de los minutos es:', ultimate_plan['count_messages'].mode()[0])
# se emplea el método describe() para calcular los estadísticos descriptivos como la media, mediana, desviación estándar
print(ultimate_plan['count_messages'].describe())

# %%
# Se traza un diagrama de caja para visualizar la distribución de los mensajes enviados
calls_mints_internet.boxplot(column= 'count_messages',
                             by= 'plan',
                             fontsize= 12,
                             grid= False,
                             figsize= [12,8]
                             )
plt.title('Distribución de los mensajes enviados por plan', fontsize= 14) # para colocar un título
plt.xlabel('Plan', fontsize= 14) # Para nombrar el eje x
plt.ylabel('No. de Mensajes', fontsize= 14) # Para nombrar el eje y
plt.show()

# %% [markdown]
# <span style="color:green"> **Conclusiones:**  
# • En el gráfico de barras se puede observar que durabte todos los meses del año los usuarios del plan ultimate en promedio enviaron más mensajes.
# • Con base en el histograma se observa que los mensajes enviados esta entre 0 y 5, para ambos planes se observa una distribución similar. Además el histrograma tiene un sesgo hacia la derecha, es decir no es una distribución simétrica. La cantidad de mensajes más frecuentes para el plan Surf son 0, mientras que para el plan ultimate también son 0.     
# • Al calcular la media para ambos planes la cantidad de mensajes enviados es muy similar y ambos planes tienen una varianza muy grande, esto quiere decir que los datos están muy separados de la media. Con la desviación estandar podemos saber que el 68 % de los datos están entre 0 y 64.6 para el plan Surf y pararl plan Ultimate están entre 2.79 y 72.3.  
# • Con el diagrama de caja podemos observar valores atípicos, ya que hay valores mayores que el valor máximo (1.5 veces el rango intercuartílico (IQR) del cuartil Q3). Lo anterior sucede para ambos planes. El valor máximo para el plan Surf es de 266 mensajes y para el plan Ultimate 166.
# </span>

# %% [markdown]
# ### 7.3 Internet <a id='internet_comportamiento'></a>

# %%
# Compara la cantidad de tráfico de Internet consumido por usuarios por plan
# Se agrupa por mes el DataFrame 'surf_plan' y se calcula el promedio del tráfico de internet (columna 'vol_internet')
# EL resultado se asigna a 'month_surf_mean_internet'
month_surf_mean_internet = surf_plan.groupby('month')['vol_internet'].mean()
month_surf_mean_internet


# %%
# Se agrupa por mes el DataFrame 'ultimate_plan' y se calcula el promedio del tráfico de internet (columna 'vol_internet')
# EL resultado se asigna a 'month_ultimate_mean_internet'
month_ultimate_mean_internet = ultimate_plan.groupby('month')['vol_internet'].mean()
month_ultimate_mean_internet

# %%
# se concatenan los Series 'month_surf_mean_internet' y 'month_ultimate_mean_internet'
surf_ultimate_mean_internet = pd.concat([month_surf_mean_internet, month_ultimate_mean_internet], axis='columns')
# Se renombran las columnas
surf_ultimate_mean_internet.columns = ['internet_mean_surf', 'internet_mean_ultimate']
# con reset_index() se reinicia el índice del DataFrame 'surf_ultimate_mean_messages'
surf_ultimate_mean_internet.reset_index(inplace= True)
surf_ultimate_mean_internet

# %%
# Se grafica la cantidad promedio de tráfico de internet por cada plan y por cada mes
surf_ultimate_mean_internet.plot(x= 'month',
                       kind= 'bar',
                       figsize= [14,10],
                       fontsize= 12, 
                       rot= 360
                       )
plt.title('Tráfico de Internet promedio por plan y mes', fontsize=15)
plt.xlabel('Mes', fontsize=15)
plt.ylabel('Promedio de tráfico de internet (gigabytes)', fontsize=15)
plt.legend(['Plan Surf', 'Plan Ultimate'], fontsize= 12)
plt.show()

# %%
# A partir de los DataFrames 'surf_plan' y 'ultimate_plan' se crean sus respectivos histogramas en el mismo gráfico
surf_plan['vol_internet'].plot(kind='hist',
                              bins= 40, 
                              alpha= 0.6,
                              figsize= [14,8]
                             )
ultimate_plan['vol_internet'].plot(kind='hist',
                              bins= 40, 
                              alpha= 0.6,
                              figsize= [14,8]                              
                             )


plt.title('Tráfico de internet (gigabytes)', fontsize= 14) # para colocar un título
plt.ylabel('Frecuencia', fontsize= 14) # Para nombrar el eje y
plt.legend(['Plan Surf', 'Plan Ultimate'], fontsize= 12) # para colocar las leyendas del gráfico
plt.show()

# %%
print('Plan Surf:')
# Se calcula la varianza con var() de los gigas usados.
print('La varianza de los gigas usados es:', round(surf_plan['vol_internet'].var(), 1))
# se calcula la moda de los datos con mode() de los gigas usados
print('La moda de los gigas usados es:', surf_plan['vol_internet'].mode()[0], 'y', surf_plan['vol_internet'].mode()[1])
# se emplea el método describe() para calcular los estadísticos descriptivos como la media, mediana, desviación estándar
print(surf_plan['vol_internet'].describe())

# %%
print('Plan Ultimate:')
# Se calcula la varianza con var() de los gigas usados.
print('La varianza de los gigas usados es:', round(ultimate_plan['vol_internet'].var(), 1))
# se calcula la moda de los datos con mode() de los gigas usados
print('La moda de los gigas usados es:', ultimate_plan['vol_internet'].mode()[0])
# se emplea el método describe() para calcular los estadísticos descriptivos como la media, mediana, desviación estándar
print(ultimate_plan['vol_internet'].describe())

# %%
# Se traza un diagrama de caja para visualizar la distribución del tráfico de internet
calls_mints_internet.boxplot(column= 'vol_internet',
                             by= 'plan',
                             fontsize= 12,
                             grid= False,
                             figsize= [12,8]
                             )
plt.title('Distribución del tráfico de internet por plan', fontsize= 14) # para colocar un título
plt.xlabel('Plan', fontsize= 14) # Para nombrar el eje x
plt.ylabel('Tráfico de internet (gigabytes)', fontsize= 14) # Para nombrar el eje y
plt.show()

# %% [markdown]
# <span style="color:green">  **Conclusiones:**  
# • En el gráfico de barras se puede observar que durante los meses de enero a mayo, los usuarios del plan ultimate usaron más gigabytes. A partir de febrero (mes 2) los usuarios en promedio excedieron el límite de los gigas incluidos en su plan. A partir de julio (mes 7) los usuarios o  usuarias del plan Surf tuvieron consumos similares a los clientes del plan Ultimate. Los usuarios o usuarias del plan Surf a partir de febrero excedieron en el doble el límite de gigas incluidos en su plan. En ambos planes se exceden los gigas incluidos en cada plan, lo cual ocurre a partir de febrero.   
# • Con base en el histograma se observa que ambos planes tienen una distribución similar. Para el plan Surf los gigas usados más frecuentes son 41 y 50, mientras que para los clientes del plan ultimate es 42.  
# • Al calcular la media para ambos planes el tráfico de internet es muy similar y para  ambos planes tienen una varianza muy grande, esto quiere decir que los datos están muy separados de la media. Lo mismo ocurre con la desviación estándar, sel 68 % de los datos se encuentran entre 22.3 y 57.98 para el plan Surf, para el plan Ultimate están entre 24.56 y 57.59.  
# • Con el diagrama de caja podemos observar valores atípicos, ya que también hay valores mayores que el valor máximo (1.5 veces el rango intercuartílico (IQR) del cuartil Q3). Lo anterior sucede para ambos planes, lo cuál se podría deber a los usuarios que exceden el límite de su plan. Para el plan Ultimate también hay valores menores que el valor mínimo.
# </span>

# %% [markdown]
# ## 8 Ingreso <a id='ingreso'></a>

# %%
# Se compara la cantidad de tráfico de ingresos por usuarios por plan
# Se agrupa por mes el DataFrame 'surf_plan' y se calcula el promedio de los ingresos (columna 'usd_total_pay')
# EL resultado se asigna a 'month_surf_mean_internet'
month_surf_mean_revenue = surf_plan.groupby('month')['usd_total_pay'].mean()
month_surf_mean_revenue

# %%
# Se agrupa por mes el DataFrame 'ultimate_plan' y se calcula el promedio de los ingresos (columna 'usd_total_pay')
# EL resultado se asigna a 'month_ultimate_mean_revenue'
month_ultimate_mean_revenue = ultimate_plan.groupby('month')['usd_total_pay'].mean()
month_ultimate_mean_revenue

# %%
# se concatenan los Series 'month_surf_mean_revenue' y 'month_ultimate_mean_revenue'
surf_ultimate_mean_revenue = pd.concat([month_surf_mean_revenue, month_ultimate_mean_revenue], axis='columns')
# Se renombran las columnas
surf_ultimate_mean_revenue.columns = ['revenue_mean_surf', 'revenue_mean_ultimate']
# con reset_index() se reinicia el índice del DataFrame 'surf_ultimate_mean_messages'
surf_ultimate_mean_revenue.reset_index(inplace= True)
surf_ultimate_mean_revenue

# %%
# Se grafica la cantidad promedio de ingresos por cada plan y por cada mes
surf_ultimate_mean_revenue.plot(x= 'month',
                       kind= 'bar',
                       figsize= [14,10],
                       fontsize= 12, 
                       rot= 360
                       )
plt.title('Ingresos promedio por plan y mes', fontsize=15)
plt.xlabel('Mes', fontsize=15)
plt.ylabel('Promedio de ingresos (usd)', fontsize=15)
plt.legend(['Plan Surf', 'Plan Ultimate'], fontsize= 12)
plt.axhline(y=20, xmin=0, xmax=12, color= 'red') # se agrega una línea para localizar la tarifa mensual del plan
plt.axhline(y=70, xmin=0, xmax=12, color= 'darkgrey') # se agrega una línea para localizar la tarifa mensual del plan
plt.show()

# %% [markdown]
# <span style="color:green"> Ahora los histogramas para los ingresos por plan se grafican por separado para una mejor visualización de los datos.
# </span>

# %%
surf_plan['usd_total_pay'].plot(kind='hist',
                              bins= 50,
                              alpha= 0.6,
                              figsize= [14,8]
                               )

ultimate_plan['usd_total_pay'].plot(kind='hist',
                              bins= 20,
                              alpha= 0.6,
                              figsize= [12,8],
                              color= 'green'
                               )

plt.title('Ingresos (usd)', fontsize= 14) # para colocar un título
plt.ylabel('Frecuencia', fontsize= 14) # Para nombrar el eje y
plt.legend(['Plan Surf', 'Plan Ultimate'], fontsize= 12) # para colocar las leyendas del gráfico
plt.show()

# %%
print('Plan Surf:')
# Se calcula la varianza con var() de los ingresos
print('La varianza de los ingresos es:', round(surf_plan['usd_total_pay'].var(), 1))
# se calcula la moda de los datos con mode() de los ingresos
print('La moda de los ingresos es:', surf_plan['usd_total_pay'].mode()[0])
# se emplea el método describe() para calcular los estadísticos descriptivos como la media, mediana, desviación estándar
print(surf_plan['usd_total_pay'].describe())

# %%
print('Plan Ultimate:')
# Se calcula la varianza con var() de los ingresos
print('La varianza de los ingresos es:', round(ultimate_plan['usd_total_pay'].var(), 1))
# se calcula la moda de los datos con mode() de los ingresos
print('La moda de los ingresos es:', ultimate_plan['usd_total_pay'].mode()[0])
# se emplea el método describe() para calcular los estadísticos descriptivos como la media, mediana, desviación estándar
print(ultimate_plan['usd_total_pay'].describe())

# %%
# Se traza un diagrama de caja para visualizar el ingreso para el plan surf
calls_mints_internet.boxplot(column= 'usd_total_pay',
                             by= 'plan',
                             fontsize= 12,
                             grid= False,
                             figsize= [12,8]
                             )
plt.title('Distribución de los ingresos para los planes', fontsize= 14) # para colocar un título
plt.xlabel('Plan', fontsize= 14) # Para nombrar el eje x
plt.ylabel('Ingresos (usd)', fontsize= 14) # Para nombrar el eje y
plt.show()

# %%
# Se cuentan los usuarios para cada plan
calls_mints_internet['plan'].value_counts()

# %% [markdown]
# <span style="color:green"> 
# La cantidad promedio de minutos consumidos, mensajes enviados, internet consumido e ingresos se grafica para cada plan para observar su evolución a lo largo de los meses.  
# </span>

# %%
# se grafican la cantidad de minutos promedio para cada plan en un gráfico de líneas
surf_ultimate_mean.plot(x= 'month',
                        figsize= (10, 5),
                        xticks= range(1,13,1),
                        color=['blue', 'green']
                       )

plt.title('Promedio de minutos usados para cada plan', fontsize= 14)
plt.xlabel('Mes', fontsize= 14)
plt.ylabel('Minutos consumidos', fontsize= 14)
plt.legend(['Plan Surf', 'Plan Ultimate'], fontsize= 12)
plt.show()

# %% [markdown]
# <span style="color:green"> 
# 
# Para ambos planes la cantidad de minutos usados aumenta a partir del mes de abril, unicamente en febrero los cliente ultimate enviaron más mensajes. Los usuarios o usuarias de ambos planes no excedieron los límites de su plan.
# </span>

# %%
# se grafican la cantidad de mensajes promedio consumidos para cada plan en un gráfico de líneas
surf_ultimate_mean_messages.plot(x= 'month',
                        figsize= (10, 5),
                        xticks= range(1,13,1),
                        color=['blue', 'green']
                       )

plt.title('Promedio de mensajes usados para cada plan', fontsize= 14)
plt.xlabel('Mes', fontsize= 14)
plt.ylabel('Mensajes enviados', fontsize= 14)
plt.legend(['Plan Surf', 'Plan Ultimate'], fontsize= 12)
plt.show()

# %% [markdown]
# <span style="color:green"> 
# ↑
# En ambos planes no excedieron el límite de mensajes, los clientes ultimate enviaron más mensajes a partir de febrero. 
# </span>

# %%
# se grafican la cantidad de gigabytes promedio consumidos para cada plan en un gráfico de líneas
surf_ultimate_mean_internet.plot(x= 'month',
                        figsize= (10, 5),
                        xticks= range(1,13,1),
                        color=['blue', 'green']
                       )

plt.title('Promedio de gigabytes consumidos para cada plan', fontsize= 14)
plt.xlabel('Mes', fontsize= 14)
plt.ylabel('Gigabytes consumidos', fontsize= 14)
plt.legend(['Plan Surf', 'Plan Ultimate'], fontsize= 12)
plt.show()

# %% [markdown]
# <span style="color:green"> 
# ↑
# A partir de febrero en ambos planes se excedieron en el límite de gigas, los clientes ultimate enviaron más mensajes a partir de febrero hasta mayo. Los usuarios o usuarias del plan Surf excedieron el límite de gigas el doble a partir de febrero.
# </span>

# %%
# se grafican el ingreso promedio para cada plan en un gráfico de líneas
surf_ultimate_mean_revenue.plot(x= 'month',
                        figsize= (10, 5),
                        xticks= range(1,13,1),
                        color=['blue', 'green']
                       )

plt.title('Ingresos promedio en usd para cada plan', fontsize= 14)
plt.xlabel('Mes', fontsize= 14)
plt.ylabel('Ingresos (usd)', fontsize= 14)
plt.legend(['Plan Surf', 'Plan Ultimate'], fontsize= 12)
plt.show()

# %% [markdown]
# <span style="color:green"> 
# ↑
# Los usuarios o usuarias del plan Surf excedieron pagaron más dinero a partir de abril respecto a los clientes ultimate, además, la tarifa aumenta, mientras que la tarifa de los clientes ultimate disminuye.  
# </span>

# %% [markdown]
# <span style="color:green"> **Conclusiones:**  
# • En el gráfico de barras se puede observar que el ingreso es mayor para los usuarios con el plan Surf. Los usuarios o usuarias con el plan Surf exceden en la mayoría de los meses la tarifa del plan, la tarifa es de 20 usd, mientras que, los usuarios con el plan ultimate también exceden la tarifa del plan (70 usd). Sin embargo, los clientes con el plan Surf exceden la tarifa en mayor proporción.  
# • Con base en el histograma para los usuarios con el plan Surf la mayoría de los ingresos están en 20 usd, pero hay un sesgo a la derecha, esto se debe a datos atípicos y que se puede verificar en el diagrama de caja. Lo anterior puede ser consecuencia de los usuarios que pagaron más de su tarifa base.    
# • Para los usuarios con el plan Ultimate los ingresos en su mayoría son menores de 70 usd, también hay valores que provocan un sesgo a la derecha del histograma, que pueden ser los usuarios o usuarias que se excedieron en su tarifa mensual, con el diagrama de caja se puede observar que hay valores atípicos, ya que también hay valores mayores que el valor máximo.  
# • Al calcular la media para ambos planes los ingresos son diferente, pero los usuarios con el plan Surf pagaron en promedio  282.1 usd, 14 veces más de su tarifa base. Por otro lado, los clientes del plan Ultimate en promedio ppagaron 166.6 usd, alrededor de dos veces más de su tarifa base, a pesar que se excedieron en su pago base este es menor en comparación con la cantidad que pagaron los usuarios o usuarias del plan Surf. Para  ambos planes tienen una varianza muy grande, esto quiere decir que los datos están muy separados de la media.  
# • El plan Ultimate tiene menos usuarios 720, mientras que el plan Surf tiene más usuarios 1573, estos últimos son los que pagan el doble de su tarifa base al mes.
# </span>

# %% [markdown]
# ## 9 Prueba las hipótesis estadísticas <a id='hipotesis'></a>

# %% [markdown]
# <span style="color:green">  **Primera prueba de hipótesis**  
# Ho: Los ingresos promedios de los usuarios de los planes Ultimate y Surf son iguales.  
# Ha: Los ingresos promedios de los usuarios de los planes Ultimate y Surf son diferentes.  
# Para probar la hipótesis se empleará la función `scipy.stats.ttest_ind(array1, array2, equal_var)`.  
# El DataFrame `surf_plan` y `ultimate_plan` tienen la información de los ingresos de cada plan y son los que se pasarán como argumento en los parámetros `array1` y `array2`, respectivamente.  
#     Previamente se calculó la varianza para cada plan y son diferentes (plan surf= 2965.0 y plan ultimate= 129.8), con base en lo anterior en el parámetro `equal_var` se establecerá como `False` ya que las varianzas son diferentes.  
#     El valor de alfa será de 5 % (0.05).
# </span>

# %%
# Prueba las hipótesis
# valor de alfa
alpha= 0.05
# se asigna el resultado en 'results_plan'
results_plan = st.ttest_ind(surf_plan['usd_total_pay'], ultimate_plan['usd_total_pay'], equal_var= False)

print('El valor p es:', results_plan.pvalue)

if results_plan.pvalue < alpha:
    print('Se rechaza la hipótesis nula')
else:
    print('No se rechaza la hipótesis nula')

# %% [markdown]
# <span style="color:green"> De acuerdo al resultado, podemos rechazar la hipótesis nula de que el ingreso promedio de los planes Ultimate y Surf son iguales. El resultado indica que los ingresos de los usuarios difiere para cada plan surf o ultimate con una confianza estadística del 95%.
# </span>

# %% [markdown]
# <span style="color:green">  **Segunda prueba de hipótesis**  
# Ho: Los ingresos promedios de los usuarios del área NY-NJ es igual al de las otras regiones.  
# Ha: Los ingresos promedios de los usuarios del área NY-NJ es diferente al de las otras regiones. 
# Para probar la hipótesis también se empleará la función `scipy.stats.ttest_ind(array1, array2, equal_var)`.  
# El DataFrame `surf_plan` y `ultimate_plan` tienen la información de los ingresos de cada plan y son los que se pasarán como argumento en los parámetros `array1` y `array2`, respectivamente.  
#     El parámetro `equal_var` se establecerá como `True` (prederterminado).  
#     El valor de alfa será de 5 % (0.05).
# </span>

# %%
# El DatFrame df_users tiene la información del área a la que corresponden los usuarios
# Se imprime una muestra de df_users
df_users.head()

# %%
# Se filtra el DataFrame df_users en donde sólo se muestren el área NY-NJ
df_users[df_users['state'] == 'NY-NJ']

# %% [markdown]
# <span style="color:green"> Al revisar la información del DataFrame filtrado `df_users` no se encontró el estado o área sólo con `NY-NJ`, sin embargo, para el área `NY-NJ-PA` si hay información y las ciudad para estos estados si pertenecen a estos, Por lo tanto se usará la información de esta área para la prueba de hipótesis.  
#     A continuación, se muestran los datos filtrados para el área `NY-NJ-PA`.
# </span>

# %%
# Se filtra el DataFrame df_users en donde sólo se muestren el área NY-NJ-PA
# El resultado se asigna a area_NY_NJ_PA
df_users[df_users['state'] == 'NY-NJ-PA'].head()

# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>  </b> <a class="tocSkip"></a>  
#     
# No se encontró un área que sea solamente NY-NJ, se ha decidido emplear el área NY-NJ-PA para la prueba de hipótesis.
# </div>
# 

# %% [markdown]
# <span style="color:green"> El DataFrame `calls_mints_internet` tiene la información del ingreso total por mes para cada usuario, entonces es necesario fusionar los datos con el DataFrame `df_users` para añadir la columna del estado ('state').  
#     Esta tarea de fusionar los DataFrames se puede hacer con un merge(), ambos DataFrames tienen en común la columna `user_id`. 
#     Se crea un DataFrame llamado `df_users_id_state` a partir del DataFrame `df_users` convervando unicamente las columnas `user_id` y `state`. Después se hace el merge entre ambos DataFrames para agregar el estado.
# </span>

# %%
# Se guardan las columnas de 'user_id' y 'state' en 'df_users_id_state'
df_users_id_state = df_users[['user_id', 'state']]
df_users_id_state.head()

# %%
# Se hace el merge entre 'calls_mints_internet' y 'df_users_id_state'
calls_mints_internet_state = calls_mints_internet.merge(df_users_id_state, on= 'user_id')
calls_mints_internet_state.head()

# %% [markdown]
# <span style="color:green"> Ahora se filtran el DataFrame en donde el estado sea `NY-NJ-PA` y DataFrame en donde sólo se tengan las otras regiones.
# 
# </span>

# %%
# Se filtra el DataFrame 'calls_mints_internet_state' donde sólo se tenga el área NY-NJ-PA 
df_NY_NJ_PA = calls_mints_internet_state[calls_mints_internet_state['state'] == 'NY-NJ-PA']
# se imprime una muestra del DataFrame resultante
df_NY_NJ_PA.head()

# %%
# Se filtra el DataFrame 'calls_mints_internet_state' donde sólo se tenga el resto de áreas
# Se emplea ~ para obtener el resultado contrario del filtro
df_otras_areas = calls_mints_internet_state[~(calls_mints_internet_state['state'] == 'NY-NJ-PA')]
df_otras_areas.head()

# %% [markdown]
# <span style="color:green"> Ahora se hace la prueba de hipótesis
# 
# </span>

# %%
# Prueba las hipótesis
# valor de alfa
alpha= 0.05
# se asigna el resultado en 'results_area'
results_area = st.ttest_ind(df_NY_NJ_PA['usd_total_pay'], df_otras_areas['usd_total_pay'], equal_var= False)

print('El valor p es:', results_area.pvalue)

if results_area.pvalue < alpha:
    print('Se rechaza la hipótesis nula')
else:
    print('No se rechaza la hipótesis nula')


# %% [markdown]
# <span style="color:green"> Con base en el resultado del valor p se rechaza la hipótesis nula de que los ingresos promedios de los usuarios del área NY-NJ_PA es igual al de las otras regiones con una confianza estadística del 95%., ya que  existe una diferencia en los ingresos de acuerdo a la región, en este caso si hay una diferencia entre el área de NY-NJ-PA y el resto de áreas.
# 
# </span>

# %% [markdown]
# ## 10 Resumen general de los pasos realizados: <a id='pasos'></a>
# 
# <span style="color:green"> 
# Para trabajar con todo el conjunto de datos se importaron las librerías necesarias y se importaron los Datasets. Después se hizo un exploración de cada DatFrame con `info()` para detectar de manera general la cantidad de datos, columnas y valores ausentes. Posteriormente, los datos se procesaron para cada DataFrame encontrando y eliminando los valores duplicados y ausentes, así como detectar el tipo de dato de las columnas.  
# 
# Se corrigieron los datos en donde se consideró importante hacerlo, como cambiar los tipos de datos. Además, se enriquecieron los DataFrames agregando más columnas y redondeando algunos valores numéricos de ciertas columnas. También se buscaron los valores ausentes y duplicados, en este caso no se encontraron.
# 
# Se realizaron algunos ``merge()`` para unir diferentes DataFrames, ya que la información para hacer los análisis está en varios conjunto de datos. De igual forma se utilizó ``concat()`` para unir algunos Series, porque estos se complementaban para completar la información y a partir del DataFrame resultante hacer los diferentes gráficos. Cuando se unieron los DataFrames de las llamadas, minutos, mensajes e Internet, en algunos casos habia valores  nulos (`NaN`), ya que en determinado mes el usuario o usuaria sólo usó los minutos para llamadas, mensajes o Internet, los otros servicios no los usó, pero se decidió dejarán así estos valores nulos y no sustituirlos con 0, debido a que podría afectar los calculos posteriores.
#  
# Se crearon tres funciones cada una para calcular el costo extra de minutos, mensajes o gygas si el usuario o usuaria se excedió en el consumo. Y una cuarta función se creó para calcular el pago total, en donde a la tarifa base se sumó el costo extra de los servicios consumidos si el cliente se excedia de su límite.
#     
# Se hicieron dos pruebas de hipótesis, para ambas pruebas se seleccionó `scipy.stats.ttest_ind(array1, array2, equal_var)` ya que se están comparando las medias de dos poblaciones. Previo a realizar las pruebas de hipótesis, los valores nulos de la columna `usd_total_pay` de cada DataFrame se tuvieron que sustituir con 0, ya que si no se hacia de es manera la prueba no hacia ningún cálculo.
# </span>
# 

# %% [markdown]
# ## 11 Conclusión general <a id='conclusion'></a>
# <span style="color:green"> **Conclusiones:**  
# • Los usuarios o usuarias del Plan Surf pagan más de su tarifa mensual de 20 usd, 14 veces más su tarifa base ya que pagan en promedio 282.1 usd. Mientras que los usuarios o usuarias ultimate pagan en promedio 166.6 usd, que también se exceden de su tarifa base pero en menor medida que los usuarios Surf. Sin embargo, hay una variabilidad muy alta de los datos ya que sus varianzas son muy grandes.  
# • Asimismo, para los minutos de las llamadas, mensajes y los gigabytes la varianza es muy alta para ambos planes, lo que sugiere que hay una variabilidad en los datos. Lo anterior puede hacer que sea más difícil tener conclusiones precisas de los datos.  
# • En ambos planes los usuarios o usuarias se exceden en los límites de sus planes, ya que los diagramas de caja se pudo observar que hay valores atípicos, ya que son mayores al valor máximo. Lo anterior se puede corroborar con los histogramas, ya que en algunos casos tienen un sesgo a la derecha. Además, la distribución de los datos no es menor a la cantidad de minutos, mensajes o gigas que incluye su plan.  
# • El plan Surf tiene más usuarios o usurias que el plan Ultimate.  
# • De acuerdo a la primera prueba de hipótesis, podemos rechazar la hipótesis nula de que el ingreso promedio de los planes Ultimate y Surf son iguales. El resultado indica que los ingresos de los usuarios difiere para cada plan surf o ultimate.  
# • Y el resultado de la segunda prueba de hipótesis indica que los ingresos para el área existe una diferencia en los ingresos de acuerdo a la región, en este caso hay una diferencia entre el área de NY-NJ-PA y el resto de áreas.  
# </span>

# %%



