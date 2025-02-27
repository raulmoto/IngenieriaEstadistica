# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Importamos las librerías que vamos a utilizar

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#1a. Creamos una lista de datos para la Estatura y el Sexo correspondiente

listaEstatura = [1.62, 1.71, 1.57, 1.61, 1.80, 1.91 ,1.58, 1.63, 1.62, 1.70, 1.75, 1.68, 1.54 ,1.79 ,1.72, 1.68, 1.90, 1.69, 1.73 ,1.85, 1.60 ,1.60, 1.62, 1.77 ,1.71, 1.89, 1.92, 1.65, 1.99, 2.05 ,1.41 ,1.67, 1.93 ,1.55,2.04, 1.73 ,1.80, 1.83, 1.75, 1.66 ,1.93, 1.85 ,1.84 ,1.68, 1.63 ,1.75 ,1.77 ,1.84, 1.85 ,1.90 ,2.00, 1.83 ,2.01, 1.82 ,1.65, 1.72, 1.68 ,1.73 ,1.54, 1.65]
listaSexo = [2 ,1 ,2 ,1 ,1, 2 ,2 ,2, 2 ,1 ,1, 1 ,1 ,2 ,2 ,2 ,2 ,2,2, 2, 2, 2, 2 ,2 ,2 ,1, 1, 2 ,2 ,2, 2 ,1, 1 ,1 ,2 ,2, 2 ,2, 2 ,1 ,1, 1, 2, 1 ,2, 2, 2, 1, 2, 2 ,2 ,2, 1 ,2 ,2,1 ,1, 2, 2, 2]
textSexList = []

#1b. Reemplazamos el valor numérico por texto
def remplazamosDatos(lisSexo):
    for i in lisSexo:
        if lisSexo[i]== 1:
            textSexList.append("H")
        else:
            textSexList.append("M")

remplazamosDatos(listaSexo)
print(str(textSexList))

#1c. Generamos un DataFrame con pd.DataFrame()
df = pd.DataFrame({"Estatura": listaEstatura,'Sexo':textSexList})
df

#2a. Realizamos el resumen estadístico. PISTA: hay una función que genera las 
#    variables básicas que necesitamos. 
df.groupby(['Sexo']).describe()

#2b. Generamos una tabla de frecuencias. 
df.groupby(['Sexo']).agg(fequency = ("Sexo","count"))

#2c. Generamos un gráfico de dispersión.
plt.scatter(x = df.Sexo,y = df.Estatura)
plt.title("grafico de Dispersion")
plt.xlabel("Eje X")
plt.xlabel("Eje Y")

#2d. Generamos un gráfico de cajas y bigotes.

sns.boxplot(data = df,x = "Sexo",y = "Estatura")

#    BONUS: hay un tipo de gráfico que representa también la función de distribución. Intenta encontrarlo. 
#2e. Generamos un histograma.
df.hist(by="Sexo")


#3a. Realizamos el resumen estadístico. PISTA: hay una función que genera las 
#    variables básicas que necesitamos. 



#3c. Generamos un diagrama de barras. 

plt.bar(df.Sexo,df.Estatura)

#3d. Generamos un diagrama de sectores.

sect = df.groupby(["Sexo"]).agg(frequency=("Sexo","count"))
desfase = (0, 0.1)
plt.pie(sect.frequency,labels = sect.index,autopct = "%0.1F %%",explode = desfase)

#4a. Halla la estatura media por sexo
from statistics import mean
mean(listaEstatura) 
