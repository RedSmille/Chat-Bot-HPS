# ------------------------ IMPORTACIONES ------------------------

import json              # Para trabajar con archivos JSON
import pickle            # Para guardar y cargar objetos en archivos binarios
import numpy as np       # Para trabajar con arreglos numéricos
import re                # Expresiones regulares para tokenización
import unicodedata       # Para eliminar acentos
import random            # Para mezclar los datos de entrenamiento

# Módulos de Keras para construir y entrenar el modelo
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.optimizers.schedules import ExponentialDecay

# ------------------------ FUNCIONES AUXILIARES ------------------------

# Normaliza un texto (minúsculas + elimina acentos) y lo tokeniza en palabras
def normalizar_texto(texto):
    texto = texto.lower()
    texto = ''.join(c for c in unicodedata.normalize('NFKD', texto) if unicodedata.category(c) != 'Mn')
    return re.findall(r'\b\w+\b', texto)

# ------------------------ CARGAR INTENCIONES ------------------------

# Cargar el archivo JSON que contiene las intenciones y patrones
with open('Informacion.json', 'r', encoding='utf-8') as archivo:
    Intentos = json.load(archivo)

# ------------------------ PROCESAR DATOS ------------------------

Palabras = []       # Lista de todas las palabras tokenizadas
Clases = []         # Lista de todas las clases (tags)
Documentos = []     # Lista de tuplas (tokenización, clase)
IgnorarPalabras = ['.', ',', ';', ':', '"', "'", '(', ')', '[', ']', '{', '}', '¿', '?', '¡', '!']

# Recorrer cada intención y sus frases de ejemplo
for intento in Intentos['intents']:
    for patron in intento['preguntas']:       
        palabras_tokenizadas = normalizar_texto(patron)     # Tokenizar
        Palabras.extend(palabras_tokenizadas)               # Agregar palabras a la lista general
        Documentos.append((palabras_tokenizadas, intento['tag']))  # Asociar tokens con su clase
        if intento["tag"] not in Clases:
            Clases.append(intento['tag'])                   # Agregar nueva clase si no está registrada

# Eliminar puntuación y duplicados, luego ordenar listas
Palabras = [p for p in Palabras if p not in IgnorarPalabras]
Palabras = sorted(list(set(Palabras)))
Clases = sorted(list(set(Clases)))

# Guardar vocabulario y clases en archivos .pkl
pickle.dump(Palabras, open('words.pkl', 'wb'))
pickle.dump(Clases, open('classes.pkl', 'wb'))

# ------------------------ CREAR DATOS DE ENTRENAMIENTO ------------------------

Entrenamiento = []                      # Lista con pares (bolsa, salida esperada)
SalidaVacia = [0] * len(Clases)         # Plantilla para codificación one-hot

for doc in Documentos:
    patron_palabras = doc[0]
    
    # Crear vector de bolsa de palabras: 1 si la palabra está en el patrón, 0 si no
    bolsa = [1 if palabra in patron_palabras else 0 for palabra in Palabras]
    
    # Crear vector de salida (one-hot) según la clase correspondiente
    salida = list(SalidaVacia)
    salida[Clases.index(doc[1])] = 1
    
    Entrenamiento.append([bolsa, salida])  # Agregar entrada + salida esperada

# Mezclar datos de entrenamiento
random.shuffle(Entrenamiento)

# Separar entradas (X) y salidas (Y)
EntrenamientoX = np.array([fila[0] for fila in Entrenamiento])
EntrenamientoY = np.array([fila[1] for fila in Entrenamiento])

# ------------------------ DEFINIR MODELO DE RED NEURONAL ------------------------

modelo = Sequential()  # Modelo secuencial (lineal)

# Capa de entrada con 128 neuronas y activación ReLU
modelo.add(Dense(128, input_shape=(len(EntrenamientoX[0]),), activation='relu'))
modelo.add(Dropout(0.5))  # Capa Dropout para evitar overfitting

# Segunda capa oculta con 64 neuronas
modelo.add(Dense(64, activation='relu'))
modelo.add(Dropout(0.5))  # Otro Dropout

# Capa de salida con tantas neuronas como clases, activación softmax
modelo.add(Dense(len(EntrenamientoY[0]), activation='softmax'))

# ------------------------ COMPILAR MODELO ------------------------

# Programar la tasa de aprendizaje para que decaiga exponencialmente con el tiempo
tasa_aprendizaje = ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9)

# Optimización con SGD y nesterov
optimizador = SGD(learning_rate=tasa_aprendizaje, momentum=0.9, nesterov=True)

# Compilar el modelo con pérdida categórica y métrica de precisión
modelo.compile(loss='categorical_crossentropy', optimizer=optimizador, metrics=['accuracy'])

# ------------------------ ENTRENAR Y GUARDAR MODELO ------------------------

# Entrenar el modelo con los datos
modelo.fit(EntrenamientoX, EntrenamientoY, epochs=200, batch_size=5, verbose=1)

# Guardar modelo entrenado en archivo
modelo.save('chatbot_model.keras')

print("✅ Modelo entrenado y guardado con éxito")
