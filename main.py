"""
Author: AbelaBeta
Project Name: Hidden Markov Model
Ojala funcione, no doy mas. Solo espero 
que si llegase a haber inteligencia en mis
textos, espero que no sea la mia.
"""

from operator import invert
import numpy as np

#OBSERVACIONES
import scipy.io.wavfile as waves
import matplotlib.pyplot as plt
from scipy.fft import fft

def normalizar(sennal):
    ampMax = max(abs(sennal))
    sennal_normalizada = []
    for m in sennal: 
        sennal_normalizada.append(m/ampMax)
    return sennal_normalizada

def filtro(sennal, num_ventanas):
    n = len(sennal)
    h = n/num_ventanas
    a = 0
    b = n-1
    sennal_filtrada = []
    for i in range(num_ventanas):
        sennal_filtrada.append([])
    for i in range(num_ventanas):
        for j in range(n):
            if j>=a+h*i and j<a+h*(i+1):
                sennal_filtrada[i].append(sennal[j])
            else:
                sennal_filtrada[i].append(0)
    return sennal_filtrada

def obtener_energia(sennal):
    energia = []
    for m in sennal:
        magnitud = [abs(n) for n in m]
        energia.append(sum([e**2 for e in magnitud]))
    return energia

def guardar_observaciones(ruta, num_ventanas, dir_obs):
    for i in range(30):
        ruta_audio = ruta+str(i+1)+".wav"
        muestra = cargar_audio(ruta_audio)
        muestra = normalizar(muestra)
        muestra = filtro(muestra,num_ventanas)
        f = open(dir_obs, 'a')
        f.write(str([int(n) for n in obtener_energia(muestra)])+'$')
        f.close()

def cargar_observaciones(ruta):    
    f = open(ruta, 'r')
    cont = f.read()
    f.close() 
    cont = cont[:len(cont)-1]
    cont = cont.split('$')
    observaciones = []
    for i in cont:
        observaciones.append(eval(i))
    return observaciones

def cargar_audio(ruta_audio):
    sonido, muestra = waves.read(ruta_audio)
    return muestra


class HMM:
  def __init__(self, nEst, mTrans, pi):
    self.N = nEst
    self.A = mTrans
    self.pi = pi
    self.b = [] #matriz de emision
    for i in range(nEst):
      self.b.append([])
    self.v = [] #Posibles valores de las observaciones

  def forward(self, o):
    fwd = []
    for i in range(self.N):
      fwd.append([])
      for j in range(len(o)):
        fwd[i].append(0)
    #Iniciacion:
    for i in range(self.N):
      fwd[i][0] = self.pi[i]*self.b[i][self.v.index(o[0])]
    #Induccion:
    for j in range(self.N):
      for t in range(len(o)-1):
        aux = []
        for i in range(self.N):
          aux.append(fwd[i][t]*self.A[i][j]) 
        fwd[j][t+1] =  sum(aux) * self.b[j][self.v.index(o[t+1])]

    #Finalizacion:
    suma = 0
    for i in range(self.N):
      suma+=fwd[i][len(o)-1]
    return suma
  def backward(self, o):
    bwd = []
    for i in range(self.N):
      bwd.append([])
      for j in range(len(o)):
        bwd[i].append(0)
    #Iniciacion:
    for i in range(self.N):
      bwd[i][len(o)-1] = 1
    #Induccion:
    for i in range(self.N):
      for t in reversed(range(len(o)-1)):
        aux = []
        for j in range(self.N):
          aux.append(self.A[i][j]*self.b[j][self.v.index(o[t+1])]*bwd[j][t+1])
      bwd[i][t] = sum((aux))
    #Terminacion:
    prob = 0
    for i in range(self.N):
      prob += bwd[i][0]*self.pi[i]*self.b[i][self.v.index(o[0])]
    return prob
  def procesar_observaciones(self, o):
    #validando si el valor de la observacion esta en v
    v = []
    #registrando los valores de observaciones que no existen en v
    for i in o:
      if i not in self.v:
        v.append(i)
        self.v.append(i)
    #registrando probabilidades de emision para valores de v
    for i in v:
      for j in range(self.N):
        self.b[j].append(0)
    #calculando la probabilidad de emision
    for i in range(self.N):
      if o[i] in v:
        self.b[i][self.v.index(o[i])] = 1/o.count(o[i])
  def entrenamiento(self, o):
    prob = 0
    self.procesar_observaciones(o)
    #var fordward
    fwd = []
    for i in range(self.N):
      fwd.append([])
      for j in range(len(o)):
        fwd[i].append(0)
    #Iniciacion:
    for i in range(self.N):
      fwd[i][0] = self.pi[i]*self.b[i][self.v.index(o[0])]
    #Induccion:
    for j in range(self.N):
      for t in range(len(o)-1):
        aux = []
        for i in range(self.N):
          aux.append(fwd[i][t]*self.A[i][j]) 
        fwd[j][t+1] =  sum(aux) * self.b[j][self.v.index(o[t+1])]
    print(fwd)
    #Backward
    bwd = []
    for i in range(self.N):
      bwd.append([])
      for j in range(len(o)):
        bwd[i].append(0)
    print(bwd)
    #Iniciacion:
    for i in range(self.N):
      bwd[i][len(o)-1] = 1
    print(bwd)
    #Induccion:
    for i in range(self.N):
      for t in reversed(range(len(o)-1)):
        aux = []
        for j in range(self.N):
          aux.append(self.A[i][j]*self.b[j][self.v.index(o[t+1])]*bwd[j][t+1])
        bwd[i][t] = sum(aux)

# guardar_observaciones(ruta="BD/detener/", num_ventanas=3, dir_obs="observaciones/obs_detener.txt")
# guardar_observaciones(ruta="BD/arrancar/", num_ventanas=3, dir_obs="observaciones/obs_arrancar.txt")
# guardar_observaciones(ruta="BD/preparados/", num_ventanas=4, dir_obs="observaciones/obs_preparados.txt")
h = HMM(3,[[1,0,0],[0,1,0],[0,0,1]],[1/3, 1/3, 1/3])
h.A = [[0.25,0.50,0.25],[0.5,0.3,0.2],[1/3,1/3,1/3]]
h.b = [[0.25,1,0],[0.25,0,1],[0.5,0,0]]
h.v = [1,2,3]
h.entrenamiento([1,2,3])