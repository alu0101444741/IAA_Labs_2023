# Proyecto PLN. Informe
Roberto Carrazana Pernía</br>
alu0101444741@ull.edu.es

## Librerías
* pandas: lectura/escritura de conjuntos de datos
* nltk: de aquí se obtienen las herramientas para corregir y truncar y/o lematizar las palabras. 
* math: uso de funciones como 'floor'
* numpy: esta librería se usa únicamente por su función para calcular logaritmos (numpy.log)
* Dict: tipo de dato importado para almacenar palabras y su frecuencia en una misma estructura

## Implementación
### Preprocesamiento
El preprocesamiento llevado a cabo para palabra se puede ver en la función preprocess_token.
```python
def preprocess_token(token: str, stemming: bool = True, lemmatization: bool = False):
    token = token.strip(' ') # Removing blank spaces
    token = token.lower() # To lower case                    
    token = spell_checker.correction(token) # Spellchecking
    if (token is None): return(None)
    # Must not be a one-letter-word, a stopword, numeric, ...
    if ((len(token) == 1) or token in stop_words or has_numbers(token) or has_special_characters(token)):
        return(None)
    # Stemming
    if (stemming): token = stemmer.stem(token)
    # Lemmatization
    if (lemmatization): token = lemmatizer.lemmatize(token) 
    return(token)
```
Cada palabra (token) se pasa a minúsculas, se eliminan los espacios en blanco que hubiera, se le pasa al corrector (SpellChecker) y finalmente se le aplica el truncamiento o lematización según la opción seleccionada.

### Clases
Vocabulary (***./src/vocabulary.py***): encargada de almacenar el vocabulario obtenido de procesar F75_train.csv
Corpus (***./src/corpus.py***): almacena el corpus
LanguageModel (***./src/language_model.py***): almacena un corpus y añade la probabilidad calculada de cada palabra
### Funciones
En los archivos ***functions.py*** y ***classification.py*** se encuentran todas las funciones adicionales empleadas en el programa. Varias de ellas como "classify_document" o "has_numbers" tienen su razón de ser principalmente en mejorar la legibilidad del programa.
Si hubiera que mencionar algunas de las más importantes, un ejemplo sería "classify_all", la cual realiza la clasificación de un conjunto según los modelos de lenguaje obtenidos previamente.

### 
## Errores
### Caso: "F75_train". **Error:**
### Caso: "F75_train_1" y "F75_train_2". **Error:**