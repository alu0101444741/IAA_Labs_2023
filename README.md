# Proyecto PLN. Informe
Roberto Carrazana Pernía</br>
alu0101444741@ull.edu.es

## Librerías
* pandas: lectura/escritura de conjuntos de datos.
* nltk: de aquí se obtienen las herramientas para truncar y/o lematizar las palabras. 
* pyspellchecker: proporciona el corrector ortográfico.
* re, string: utilidades para el procesamiento de palabras.
* math: uso de funciones como 'floor'.
* numpy: esta librería se usa únicamente por su función para calcular logaritmos (numpy.log).
* Dict: tipo de dato importado para almacenar palabras y su frecuencia en una misma estructura.

## Directorios del proyecto
* input: Archivos csv utilizados como entrada: F75_train, F75_train_1 y F75_train_2.
* output: Archivos txt (vocabulario, corpus y modelos de lenguaje) y csv (clasificacion_alu y resumen_alu).
* src: Todo el código desarrollado.

## Implementación
Para ejecutar el programa: python src/main.py. Se imprimirá por consola la precisión alcanzada para los dos casos planteados, siendo "primer caso" el aprendizaje a través de F75_train y "segundo caso" aquel donde se hace la división en conjuntos de entrenamiento y test.

### Preprocesamiento
El preprocesamiento llevado a cabo para palabra se puede ver en la función preprocess_token.
```python
def preprocess_token(token: str, stemming: bool = True, lemmatization: bool = False):
    if (token is None): return(None)
    token = token.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation
    token = re.sub(r'[0-9]', '', token) # Remove numbers
    # Must not be a one-letter-word, a stopword, numeric, ...
    if ((len(token) <= 1) or (token in stop_words)):
        return(None)
    
    if (token[0].lower() != token[0]): return(token) # is a name --> Improve accuracy in some cases

    return(token)
```
Cada palabra (token) se pasa a minúsculas, se eliminan los espacios en blanco que hubiera, se le pasa al corrector (SpellChecker) y finalmente se le aplica el truncamiento o lematización según la opción seleccionada.

### Clases
* Vocabulary (***./src/vocabulary.py***): encargada de almacenar el vocabulario obtenido de procesar F75_train.csv.</br>
* Corpus (***./src/corpus.py***): almacena el corpus.</br>
* LanguageModel (***./src/language_model.py***): almacena un corpus y añade la probabilidad calculada de cada palabra.</br>

### Funciones
En los archivos ***functions.py*** y ***classification.py*** se encuentran todas las funciones adicionales empleadas. Algunas de ellas como "classify_document" tienen su razón de ser principalmente en mejorar la legibilidad del programa.</br>
Si hubiera que mencionar algunas de las más importantes, un ejemplo sería "classify_all", la cual realiza la clasificación de un conjunto según los modelos de lenguaje obtenidos previamente.
Aquellas funciones o variables auxiliares que no son parte fundamental del programa, pero han sido utilizadas durante el desarrollo del mismo se encuentran en el archivo ***utilities.py***.</br>
Por último, el programa ***out_files.py*** sirve para generar los archivos txt que se encuentran en el directorio "output".

## Errores
### Caso: "F75_train". **Error:** 18.05% (Acc.:81.95%)
### Caso: "F75_train_1" y "F75_train_2". **Error:** 20.0% (Acc.: 80.0%)

### Notas
Ciertos aspectos del preprocesado afectan de distinta forma a la precisión en cada uno de los casos propuestos.
1. La corrección ortográfica a través de SpellChecker (de la librería pyspellchecker) dispara el tiempo de ejecución de 5 segundos a 20 minutos.
    * Para K = 3: Primer caso: 61.9%. Segundo caso: 56.4%. 
    * Para K = 4: Primer caso: 64.02%. Segundo caso: 60.0%. 
    * Para K = {5, 6, 7} se mantiene la precisión alcanzada en K = 4.
2. Sustituyendo la corrección ortográfica a través de SpellChecker por una más rudimentaria y eliminando el truncamiento.
    * Para K = 0: Primer caso: 88.44%. Segundo caso: 65.6%. 
    * Para K = 1: Primer caso: 85.79%. Segundo caso: 79.0%. 
    * Para K = 2: Primer caso: 81.95%. Segundo caso: 80.0%. (Solución final)
    * Para K > 2: La precisión de ambos comienza a descender
    * Estos valores de precisión disminuyen aproximadamente un 10% si se utiliza truncamiento.
    * Por otro lado, usando lematización los valores no cambian de manera significativa pero se muestra mejor que el truncamiento.
3. Tanto el paso a minúsculas como la eliminación de espacios en blanco parece no cambiar nada el resultado.