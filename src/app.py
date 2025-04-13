# Your code here
from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd
import requests
import matplotlib.pyplot as plt

resource_url = "https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv"

response = requests.get(resource_url)

if response.status_code == 200:
    print("¡Petición exitosa! Código:", response.status_code)
    with open("/workspaces/Finarosalina_Regresion_lineal_EDA_insurance/data/raw/medical_insurance_cost.csv", "w", encoding="utf-8") as dataset:
        dataset.write(response.text)
else:
    print("Error al descargar el archivo. Código de estado:", response.status_code)

ds=pd.read_csv("/workspaces/Finarosalina_Regresion_lineal_EDA_insurance/data/raw/medical_insurance_cost.csv")

ds.describe()

# Verificar si hay filas duplicadas: si hay 1 que se elimina
ds=ds.drop_duplicates()
ds.head()

import matplotlib.pyplot as plt
import seaborn as sns
fig, axis = plt.subplots(2, 3, figsize=(12, 10))

# Crear un histograma múltiple
sns.histplot(ax=axis[0, 0], data=ds, x="charges").set(ylabel=None)
sns.histplot(ax=axis[0, 1], data=ds, x="sex").set(ylabel=None)
sns.histplot(ax=axis[0, 2], data=ds, x="smoker").set(ylabel=None)
sns.histplot(ax=axis[1, 0], data=ds, x="region")
sns.histplot(ax=axis[1, 1], data=ds, x="children").set(ylabel=None)


# Eliminar el subplot vacío que sobra (axis[3, 2])
fig.delaxes(axis[1, 2])

# Ajustar el layout
plt.tight_layout()

# Mostrar el plot
plt.show()

fig, axis = plt.subplots(2, 2, figsize = (7, 5), gridspec_kw={'height_ratios': [6, 1]})

# Variables numéricas: 'age', 'bmi'

# Crear una figura múltiple con histogramas y diagramas de caja
sns.histplot(ax = axis[0, 0], data = ds, x = "age").set(xlabel = None)
sns.boxplot(ax = axis[1, 0], data = ds, x = "age")
sns.histplot(ax = axis[0, 1], data = ds, x = "bmi").set(xlabel = None, ylabel = None)
sns.boxplot(ax = axis[1, 1], data = ds, x = "bmi")

# Ajustar el layout
plt.tight_layout()

# Mostrar el plot
plt.show()


outlier_bmi= ds[ds['bmi']>46]
outlier_bmi

sns.histplot(data=outlier_bmi, x="bmi")
sns.scatterplot(data=outlier_bmi, x="bmi", y="charges")

ds=ds[ds['bmi']<47]

# bmi y bmi con charges
fig, axis = plt.subplots(2, 2, figsize = (10, 7))

# Crear un diagrama de dispersión múltiple
sns.regplot(ax = axis[0, 0], data = ds, x = "bmi", y = "charges")
sns.heatmap(ds[["charges", "bmi"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 0], cbar = False)
sns.regplot(ax = axis[0, 1], data = ds, x = "age", y = "charges").set(ylabel=None)
sns.heatmap(ds[["charges", "age"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 1])

# Ajustar el layout
plt.tight_layout()

# Mostrar el plot
plt.show()

fig, axis = plt.subplots(2, 2, figsize = (15, 7))

sns.countplot(ax = axis[0, 0], data = ds, x = "sex", hue = "charges")
sns.countplot(ax = axis[0, 1], data = ds, x = "smoker", hue = "charges").set(ylabel = None)
sns.countplot(ax = axis[1, 0], data = ds, x = "region", hue = "charges").set(ylabel = None)
sns.countplot(ax = axis[1, 1], data = ds, x = "children", hue = "charges")

plt.tight_layout()
plt.show()

fig, axis = plt.subplots(2, 2, figsize = (15, 7))

sns.scatterplot(ax = axis[0, 0], data = ds, x = "sex", y= "charges")
sns.scatterplot(ax = axis[0, 1], data = ds, x = "smoker", y= "charges").set(ylabel = None)
sns.scatterplot(ax = axis[1, 0], data = ds, x = "region", y= "charges").set(ylabel = None)
sns.scatterplot(ax = axis[1, 1], data = ds, x = "children", y= "charges")

plt.tight_layout()
plt.show()

fig, axis = plt.subplots(figsize = (10, 5), ncols = 2)

sns.barplot(ax = axis[0], data = ds, x = "sex", y = "charges", hue = "children")
sns.barplot(ax = axis[1], data = ds, x = "sex", y = "charges", hue = "smoker").set(ylabel = None)

plt.tight_layout()

plt.show()

ds["sex_n"] = pd.factorize(ds["sex"])[0]
ds["smoker_n"] = pd.factorize(ds["smoker"])[0]
ds["region_n"] = pd.factorize(ds["region"])[0]

fig, axis = plt.subplots(figsize = (7, 5))

sns.heatmap(ds[["sex_n", "region_n", "smoker_n", "children", "charges"]].corr(), annot = True, fmt = ".2f")

plt.tight_layout()

plt.show()

fig, axis = plt.subplots(figsize = (10, 7))

sns.heatmap(ds[["sex_n", "region_n", "smoker_n", "children", "charges", "age", "bmi" ]].corr(), annot = True, fmt = ".2f")

plt.tight_layout()
plt.show()

sns.boxplot(data=ds, x="sex", y="charges")


sns.boxplot(data=ds, x="children", y="charges")

# Análisis de outliers

ds.describe()

fig, axis = plt.subplots(3, 3, figsize = (15, 10))

sns.boxplot(ax = axis[0, 0], data = ds, y = "charges")
sns.boxplot(ax = axis[0, 1], data = ds, y = "age")
sns.boxplot(ax = axis[0, 2], data = ds, y = "sex")
sns.boxplot(ax = axis[1, 0], data = ds, y = "bmi")
sns.boxplot(ax = axis[1, 1], data = ds, y = "children")
sns.boxplot(ax = axis[1, 2], data = ds, y = "sex_n")
sns.boxplot(ax = axis[2, 0], data = ds, y = "smoker_n")
sns.boxplot(ax = axis[2, 1], data = ds, y = "region_n")

plt.tight_layout()

plt.show()

ds.sort_values("charges", ascending=False).head(20)

filtro_ds= ds[ds['smoker'] == 'yes']
filtro_ds.describe()

# Calcular el primer cuartil (Q1) y el tercer cuartil (Q3)
Q1 = ds['charges'].quantile(0.25)
Q3 = ds['charges'].quantile(0.75)
IQR = Q3 - Q1

# Definir los límites inferior y superior
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

print(f"Los límites superior e inferior para la búsqueda de outliers son {round(upper_limit, 2)} y {round(lower_limit, 2)}, con un rango intercuartílico de {round(IQR, 2)}")

ds[ds['charges']>34394.27].value_counts()  # Length: 138. Sería eliminar demasiados valores del grupo de fumadores.

# Definir los límites inferior y superior
lower_limit = Q1 - 3 * IQR
upper_limit = Q3 + 3 * IQR

print(f"Los límites superior e inferior para la búsqueda de outliers son {round(upper_limit, 2)} y {round(lower_limit, 2)}, con un rango intercuartílico de {round(IQR, 2)}")

# Filtrar primero los valores bajos de charges y crear un nuevo DataFrame
ds_minimo = ds[ds['charges'] < 52184.24].copy()
ds_minimo.drop(['region', 'region_n', 'children', 'smoker', 'sex', 'sex_n'], axis=1, inplace=True)
ds_minimo.head()

output_path = '/workspaces/Finarosalina_Regresion_lineal_EDA_insurance/data/processed/ds_minimo.csv'
ds_minimo.to_csv(output_path, index=False)


from sklearn.model_selection import train_test_split


# Dividimos el conjunto de datos en muestras de train y test
X = ds_minimo.drop("charges", axis = 1)
y = ds_minimo["charges"]

# split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


X_train.head()

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import pandas as pd

# Variables numéricas a escalar 
variables_para_escalar = ['age', 'bmi']

# Inicializamos el escalador
scaler = MinMaxScaler()

# Escalar solo las variables que se deben escalar
X_train_escalado = X_train.copy()
X_train_escalado[variables_para_escalar] = scaler.fit_transform(X_train[variables_para_escalar])

# Escalar los datos de prueba (usando el mismo escalador)
X_test_escalado = X_test.copy()
X_test_escalado[variables_para_escalar] = scaler.transform(X_test[variables_para_escalar])

# Crear el modelo de regresión lineal
modelo = LinearRegression()

# Entrenar el modelo con los datos escalados (age y bmi), 
modelo.fit(X_train_escalado, y_train)

# Realizar predicciones
y_pred = modelo.predict(X_test_escalado)

# Mostrar las primeras predicciones para verificar
print(y_pred[:5])



from sklearn.linear_model import LinearRegression

# Crear el modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo con los datos escalados
model.fit(X_train_escalado, y_train)


print(f"Intercepto (a): {model.intercept_}")
print(f"Coeficientes (b): {model.coef_}")

y_pred = model.predict(X_test_escalado)

X_test_escalado.head()

# Predecir con el modelo ya entrenado
y_pred = model.predict(X_test_escalado)

# Asegurarte de que sea una Serie 1D con el mismo índice
y_pred = pd.Series(y_pred, index=y_test.index)


from sklearn.metrics import mean_squared_error, r2_score

print(f"Error cuadrático medio: {mean_squared_error(y_test, y_pred)}")
print(f"Coeficiente de determinación: {r2_score(y_test, y_pred)}")

# Abre el archivo de origen (explore.ipynb)
with open('/workspaces/Finarosalina_Regresion_lineal_EDA_insurance/src/explore.ipynb', 'r') as source_file:
    content = source_file.read()

# Escribe el contenido en el archivo de destino (app.py)
with open('/workspaces/Finarosalina_Regresion_lineal_EDA_insurance/src/app.py', 'w') as destination_file:
    destination_file.write(content)
