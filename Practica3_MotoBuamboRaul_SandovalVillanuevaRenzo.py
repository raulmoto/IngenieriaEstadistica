# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:18:08 2025

@author: renzo 
@author: raul moto
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.stats import binom
from scipy.stats import poisson

import warnings
warnings.filterwarnings('ignore')

"""
    Problema 1: La resistencia a la tracción de cierta pieza de sujeción de acero es una variable
    aleatoria, pues no existirán dos piezas de este tipo con exactamente la misma resistencia. La
    variable aleatoria que representa dicha resistencia en una pieza génerica se puede modelar con una
    distribución normal de media 268 kg/cm² y desviación típica 11 kg/cm².
     • Dibuja f(x) y F(x).
     • ¿Cuál es la probabilidad de que la resistencia de una pieza cualquiera sea menor de 270kg/cm²?
     • ¿Cuál es la probabilidad de que la resistencia de una pieza cualquiera esté comprendida entre 255 y 280 kg/cm²?
     • ¿Cuál es el valor de la resistencia que sólo es superada por el 25% de las piezas?
     • Si se rechazan todas las piezas con resistencias menor de 242 kg/cm². ¿Cuál será el porcentajede piezas rechazadas?

"""
distribucion = 268
desviacion = 11
X = np.linspace(distribucion - 4*desviacion, distribucion + 4*desviacion, 1000)
plt.figure(figsize=(12, 5))
# Calculamos (f(x)) y (F(x))

# (f(x))
c_df = norm.pdf(X, distribucion, desviacion)
plt.subplot(1, 2, 1)
plt.plot(X, c_df, "b-", label="(f(x))")
plt.title("Densidad")
plt.xlabel("Resistencia (kg/cm²)")
plt.ylabel("Densidad")
plt.legend()

# (F(x))
c_dF = norm.cdf(X, distribucion, desviacion)
plt.subplot(1, 2, 2)
plt.plot(X, c_dF, "r-", label="(F(x))")
plt.title("Distribucion acumulada")
plt.xlabel("Resistencia (kg/cm²)")
plt.ylabel("Probabilidad acumulada")
plt.legend()

plt.tight_layout()
plt.show()


#Calculamos las probabilidades
# P(X < 270) resistencia menor a 270 kg/cm²
probabilidad_menor_270 = norm.cdf(270, distribucion, desviacion)
procentaje_menor_270 = probabilidad_menor_270 * 100
print("Probabilidad de resistencia menor a 270 kg/cm²: " + str(procentaje_menor_270) + "%")

# P(255 < X < 280) resistencia comprendidad entre 255 y 280 kg/cm²
probabilidad_entre_255_280 = norm.cdf(280, distribucion, desviacion) - norm.cdf(255, distribucion, desviacion)
porcentaje_entre_255_280 = probabilidad_entre_255_280 * 100
print("Probabilidad de resistencia comprendidad entre 255 y 280 kg/cm²: " + str(porcentaje_entre_255_280) + "%")

# Resistencia que supera el 25%, lo calculamos con el percentil 75
resistencia_25 = norm.ppf(0.75, distribucion, desviacion)
print("Resistencia que supera el 25%: " + str(resistencia_25) + "kg/cm²")

#Calculamos porcentaje
# P(X<242) resistencias menores que 242 kg/cm²
porcentaje_rechazadas = norm.cdf(242, distribucion, desviacion) * 100
print("Porcentaje rechazado: " + str(porcentaje_rechazadas) + "%")

"""
     Problema 2: Un proceso productivo produce un 5% de artículos defectuosos. Supondremos
     que el estado defectuoso/aceptable de un artículo es independiente del resto de los artículos. Los
     artículos se venden en lotes de 90 artículos. Queremos saber:
      • Probabilidad de que en un lote haya exactamente 4 artículos defectuosos.
      • Probabilidad de que en un lote tenga menos de 8 artículos defectuosos.
      • Probabilidad de que en un lote tenga 6 ó menos artículos defectuosos.
      • Calcula los cuartiles del número de artículos defectuosos que hay en un lote.
"""
articulos = 90
probabilidad = 0.05
# Calculamos las probabilidades
# De que haya exactamente 4 artículos defectuosos
probabilidad_4 = binom.pmf(4, articulos, probabilidad)
porcentaje_4 = probabilidad_4*100
print("Probabilidad de que hayan 4 articulos defectuosos: " + str(porcentaje_4) + "%")

# (P(X < 8)) menos de 8 defectuosos
probabilidad_menos_8 = binom.cdf(7, articulos, probabilidad)
porcentaje_menos_8 = probabilidad_menos_8*100
print("Probabilidad de que hayan menos de 8 articulos defectuosos: " + str(porcentaje_menos_8) + "%")

# (P(X <= 6)) 6 o menos defectuosos
probabilidad_6_o_menos = binom.cdf(6, articulos, probabilidad)
porcentaje_6_o_menos = probabilidad_6_o_menos*100
print("Probabilidad de que hayan 6 o menos articulos defectuosos: " + str(porcentaje_6_o_menos*100) + "%")

# Calculamos cuartiles
Q1 = binom.ppf(0.25, articulos, probabilidad)
Q2 = binom.ppf(0.50, articulos, probabilidad)
Q3 = binom.ppf(0.75, articulos, probabilidad)
print("Cuartiles")
print("Q1 = " + str(Q1))
print("Q2 = " + str(Q2))
print("Q3 = " + str(Q3))

#Grafica de distribucion
valores = range(0, 16)
probabilidades = [binom.pmf(i, articulos, probabilidad) for i in valores]

plt.bar(valores, probabilidades)
plt.title("Distribucion binomial")
plt.xlabel("Número de defectuosos")
plt.ylabel("Probabilidad")
plt.show()

"""
    Problema 3: Un servidor de internet recibe una media 4.7 accesos al minuto durante la
    jornada laboral. En ese tiempo, los usuarios acceden al servidor con un ritmo estable y de forma
    independiente, de forma que los accesos por unidad de tiempo pueden aproximarse a un proceso
    de Poisson.
     • ¿Qué porcentaje del tiempo tendrá más de 4 accesos por minuto?
     • Considerando el servidor anterior, ¿es cierto que el 50% de los minutos se reciben menos de dos accesos?
     • Durante los fines de semana, el servidor anterior sólo recibe una media de 4 accesos por minuto.
       ¿Cuál es la probabilidad de que el servidor esté más de un minuto sin recibir llamadas?
"""
# Probabilidad
# (P(X > 4)) mas de 4 accesos por minuto
med_inter = 4.7
prob_mas_4 = 1 - poisson.cdf(4, med_inter)
por_mas_4 = prob_mas_4 * 100
print("Porcentaje del tiempo tendrá más de 4 accesos por minuto: " + str(por_mas_4 * 100) + "%")

# 50% de accesos
prob_menos_2 = poisson.cdf(1, med_inter)
por_menos_2 = prob_menos_2 * 100
print("2 accesos en el 50% de minutos, probabilidad: " + str(por_menos_2) + "%")

# Probabilidad mas de 1 minuto sin accesos en fin de semana con media 4 de internet
med_inter_4 = 4
prob_sin_accesos = np.exp(-med_inter_4 * 1)
por_sin_accesos = prob_sin_accesos * 100
print("Probabilidad de que el servidor esté más de un minuto sin recibir llamadas: " + str(por_sin_accesos * 100) + "%")

# Grafica
x = np.arange(0, 15)
pmf = poisson.pmf(x, med_inter)
plt.bar(x, pmf)
plt.title("Distribución de Poisson")
plt.xlabel("Accesos por minuto")
plt.ylabel("Probabilidad")
plt.show()