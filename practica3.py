# -*- coding: utf-8 -*-
"""
Created on Tue May  6 09:08:40 2025

@author: Raul-CDH
"""
%matplotlib inline

"""

    Problema 1.
    
    La resistencia a la tracción de cierta pieza de sujeción de acero es una 
    variable aleatoria, pues no existirán dos piezas de este tipo con exactamente 
    la misma resistencia. La variable aleatoria que representa dicha resistencia 
    en una pieza génerica se puede modelar con una distribución normal de media 
    268 kg/cm² y desviación típica 11 kg/cm².
    
    Dibuja f(x) y F(x).
    ¿Cuál es la probabilidad de que la resistencia de una pieza cualquiera sea menor
    de 270 kg/cm²?
    ¿Cuál es la probabilidad de que la resistencia de una pieza cualquiera esté 
    comprendida entre 255 y 280 kg/cm²?
    ¿Cuál es el valor de la resistencia que sólo es superada por el 25% de las 
    piezas?
    Si se rechazan todas las piezas con resistencias menor de 242 kg/cm². ¿Cuál 
    será el porcentaje de piezas rechazadas?

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parámetros
mu, sigma = 268, 11
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)  # Rango de valores

# PDF (f(x))
pdf = norm.pdf(x, mu, sigma)

# CDF (F(x))
cdf = norm.cdf(x, mu, sigma)

# Gráficas
plt.figure(figsize=(12, 5))

# PDF
plt.subplot(1, 2, 1)
plt.plot(x, pdf, 'b-', label='PDF (f(x))')
plt.title('Función de Densidad (PDF)')
plt.xlabel('Resistencia (kg/cm²)')
plt.ylabel('Densidad')
plt.legend()

# CDF
plt.subplot(1, 2, 2)
plt.plot(x, cdf, 'r-', label='CDF (F(x))')
plt.title('Función de Distribución Acumulada (CDF)')
plt.xlabel('Resistencia (kg/cm²)')
plt.ylabel('Probabilidad Acumulada')
plt.legend()

plt.tight_layout()
plt.show()

#probabilidad P(X < 270)
prob_menor_270 = norm.cdf(270, mu, sigma)
print(f"P(X < 270): {prob_menor_270:.4f} ({(prob_menor_270 * 100):.2f}%)")

#probabilidad P(255 < X < 280)
prob_entre_255_280 = norm.cdf(280, mu, sigma) - norm.cdf(255, mu, sigma)
print(f"P(255 < X < 280): {prob_entre_255_280:.4f} ({(prob_entre_255_280 * 100):.2f}%)")

#resistencia qeu supera el 25% de las piezas (percentil 75)
percentil_75 = norm.ppf(0.75, mu, sigma)
print(f"Resistencia superada por el 25%: {percentil_75:.2f} kg/cm²")

#Porcentaje de piezas rechazadas (X<242)
porcentaje_rechazadas = norm.cdf(242, mu, sigma) * 100
print(f"Porcentaje rechazado: {porcentaje_rechazadas:.2f}%")



"""
    6.
    Un proceso productivo produce un 5% de artículos defectuosos. Supondremos que el estado defectuoso/aceptable de un artículo es independiente del resto de los artículos. Los artículos se venden en lotes de 90 artículos. Queremos saber:
    
    Probabilidad de que en un lote haya exactamente 4 artículos defectuosos.
    Probabilidad de que en un lote tenga menos de 8 artículos defectuosos.
    Probabilidad de que en un lote tenga 6 ó menos artículos defectuosos.
    Calcula los cuartiles del número de artículos defectuosos que hay en un lote.

"""

from scipy.stats import binom

n, p = 90, 0.05
prob_4 = binom.pmf(4, n, p)
print(f"P(X=4): {prob_4:.4f} ({prob_4*100:.2f}%)")

#probabilidad de menos de 8 defectuosos (P(X < 8))
prob_menos_8 = binom.cdf(7, n, p)  # Nota: cdf incluye el límite superior
print(f"P(X < 8): {prob_menos_8:.4f} ({prob_menos_8*100:.2f}%)")


#probabilidad e 6 o menos defectuosos (P(X <= 6))
prob_6_o_menos = binom.cdf(6, n, p)
print(f"P(X ≤ 6): {prob_6_o_menos:.4f} ({prob_6_o_menos*100:.2f}%)")

#Cuartiles del número de defectuosos
Q1 = binom.ppf(0.25, n, p)
Q2 = binom.ppf(0.50, n, p)
Q3 = binom.ppf(0.75, n, p)
print(f"Cuartiles: Q1 = {Q1}, Q2 (mediana) = {Q2}, Q3 = {Q3}")

#Gráfica de la Distribución

import matplotlib.pyplot as plt

r_values = range(0, 16)  # Rango relevante
probabilidades = [binom.pmf(k, n, p) for k in r_values]

plt.bar(r_values, probabilidades)
plt.title("Distribución Binomial (n=90, p=0.05)")
plt.xlabel("Número de defectuosos")
plt.ylabel("Probabilidad")
plt.show()


"""
    Un servidor de internet recibe una media 4.7 accesos al minuto durante la jornada laboral. En ese tiempo, los usuarios acceden al servidor con un ritmo estable y de forma independiente, de forma que los accesos por unidad de tiempo pueden aproximarse a un proceso de Poisson.
    
    ¿Qué porcentaje del tiempo tendrá más de 4 accesos por minuto?
    Considerando el servidor anterior, ¿es cierto que el 50% de los minutos se reciben menos de dos accesos?
    Durante los fines de semana, el servidor anterior sólo recibe una media de 4 accesos por minuto. 
    ¿Cuál es la probabilidad de que el servidor esté más de un minuto sin recibir llamadas?


"""

# Probabilidad de más de 4 accesos por minuto  (P(X > 4))
from scipy.stats import poisson

lambda_laboral = 4.7
prob_mas_4 = 1 - poisson.cdf(4, lambda_laboral)  # P(X > 4) = 1 - P(X ≤ 4)
print(f"P(X > 4): {prob_mas_4:.4f} ({(prob_mas_4 * 100):.2f}%)")

#¿El 50% de los minutos tiene menos de 2 accesos?
prob_menos_2 = poisson.cdf(1, lambda_laboral)
print(f"P(X < 2): {prob_menos_2:.4f} ({(prob_menos_2 * 100):.2f}%)")
print(f"¿Es cierto que P(X < 2) ≈ 50%? {prob_menos_2 >= 0.5}")

#mediana
mediana = poisson.ppf(0.5, lambda_laboral)
print(f"Mediana (accesos/min): {mediana}")

# Probabilidad de más de 1 minuto sin accesos (fin de semana)

lambda_finde = 4
prob_sin_accesos = np.exp(-lambda_finde * 1)  # t = 1 minuto
print(f"P(T > 1 min): {prob_sin_accesos:.4f} ({(prob_sin_accesos * 100):.2f}%)")


#gafica de referencia
import seaborn as sns

# Gráfica de Poisson (lambda = 4.7)
x = np.arange(0, 15)
pmf = poisson.pmf(x, lambda_laboral)
plt.bar(x, pmf)
plt.title("Distribución de Poisson ($\lambda = 4.7$)")
plt.xlabel("Accesos por minuto")
plt.ylabel("Probabilidad")
plt.show()
