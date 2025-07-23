# Librerías necesarias para servidor HTTP, manejo de archivos y procesamiento
import http.server
import socketserver
import json
import os
import pickle
import numpy as np
import re
import unicodedata
from keras.models import load_model
from respuestas_chatbot import ObtenerRespuesta  # Función personalizada para obtener la respuesta del bot
import locale

# Establecer configuración regional en español (para fechas u otros datos localizados)
try:
    locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')
except locale.Error:
    print("La configuración regional 'es_ES.UTF-8' no está disponible.")

# Cargar archivo de intenciones en formato JSON
try:
    with open('Informacion.json', 'r', encoding='utf-8') as archivo:
        Intentos = json.load(archivo)
except Exception as e:
    print(f"❌ Error cargando 'Informacion.json': {e}")
    Intentos = {}

# Cargar palabras, clases y el modelo previamente entrenado
try:
    Palabras = pickle.load(open('words.pkl', 'rb'))     # Lista de palabras conocidas
    Clases = pickle.load(open('classes.pkl', 'rb'))     # Lista de intenciones (clases)
    Modelo = load_model('chatbot_model.keras')          # Modelo de Keras entrenado
except Exception as e:
    print(f"❌ Error cargando archivos del modelo: {e}")
    exit(1)

# ----------------------- FUNCIONES DE PROCESAMIENTO DE TEXTO -----------------------

# Función para normalizar texto: elimina acentos y convierte a minúsculas
def NormalizarTexto(Texto):
    Texto = Texto.lower()
    return ''.join(c for c in unicodedata.normalize('NFKD', Texto) if unicodedata.category(c) != 'Mn')

# Tokeniza una oración en palabras
def Tokenizar(oracion):
    oracion = NormalizarTexto(oracion)
    return re.findall(r'\b\w+\b', oracion)

# Genera una bolsa de palabras en forma de vector binario según las palabras conocidas
def BolsaDePalabras(oracion):
    palabras_oracion = Tokenizar(oracion)
    bolsa = [1 if palabra in palabras_oracion else 0 for palabra in Palabras]
    return np.array(bolsa)

# Genera todos los n-gramas (hasta n=4) posibles a partir de los tokens
def GenerarNGramas(tokens, max_n=4):
    ngramas = set()
    longitud = len(tokens)

    for n in range(1, max_n + 1):
        for i in range(longitud - n + 1):
            ngrama = " ".join(tokens[i:i + n])
            ngramas.add(ngrama)
    
    return ngramas

# Busca coincidencias exactas entre n-gramas del input y las frases de ejemplo en el archivo JSON
def BuscarConNGramas(oracion, intentos_json):
    tokens = Tokenizar(oracion)
    ngramas = GenerarNGramas(tokens)

    for intento in intentos_json["intents"]:  # Recorre intenciones en orden
        for frase in intento["preguntas"]:
            frase_normalizada = NormalizarTexto(frase)
            if frase_normalizada in ngramas:
                return [{"Intencion": intento["tag"], "Probabilidad": "1.0"}]
    
    # Si no se encuentra ninguna coincidencia
    return [{"Intencion": "unknown", "Probabilidad": "0.0"}]

# ----------------------- CONFIGURACIÓN DEL SERVIDOR HTTP -----------------------

# Puerto donde se ejecutará el servidor
PUERTO = 8000  

# Clase que maneja las peticiones HTTP al chatbot
class ManejadorChatbot(http.server.SimpleHTTPRequestHandler):
    
    # Maneja peticiones GET (como abrir index.html)
    def do_GET(self):
        if self.path == '/':
            self.path = '/index.html'  # Redirecciona raíz al archivo principal
        if os.path.exists(self.path[1:]):  # Verifica que el archivo exista
            return super().do_GET()
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Archivo no encontrado')

    # Maneja peticiones POST (envío de preguntas al chatbot)
    def do_POST(self):
        try:
            longitud = int(self.headers.get('Content-Length', 0))  # Obtener tamaño del cuerpo del POST
            datos_post = self.rfile.read(longitud)  # Leer contenido del POST
            datos = json.loads(datos_post.decode('utf-8'))  # Decodificar JSON
            pregunta = datos.get('prompt', '').strip()  # Obtener texto del mensaje enviado

            if not pregunta:
                # Si no se envió texto, responder con un mensaje de advertencia
                respuesta = {"response": ["Por favor, ingresa un mensaje o pregunta."]}
            else:
                # Buscar intención con n-gramas y obtener respuesta del chatbot
                intentos = BuscarConNGramas(pregunta, Intentos)
                texto_respuesta = ObtenerRespuesta(intentos, Intentos)
                respuesta = {"response": texto_respuesta}

            # Enviar respuesta HTTP al cliente
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(respuesta, ensure_ascii=False).encode('utf-8'))

        except Exception as e:
            # En caso de error interno, responder con código 500 y mensaje de error
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}, ensure_ascii=False).encode('utf-8'))

# ----------------------- INICIAR SERVIDOR -----------------------

# Crear e iniciar el servidor en modo multihilo (permite varias conexiones)
with socketserver.ThreadingTCPServer(('0.0.0.0', PUERTO), ManejadorChatbot) as httpd:
    print(f'Servidor ejecutándose en: http://localhost:{PUERTO}/')
    httpd.serve_forever()  # Mantener el servidor en ejecución indefinidamente
