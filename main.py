# Importamos librerias
from fastapi import FastAPI
import pandas as pd
import dateutil.parser as dparser
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score


#-------------------------------------------------------------------------------------------------------------------------------


# Instanciamos nuestra clase de la API
app = FastAPI()


#-------------------------------------------------------------------------------------------------------------------------------


# Cargamos los archivos
steam_games = pd.read_parquet('Data_Consumible/steam_games.parquet')
users_items = pd.read_parquet('Data_Consumible/users_items.parquet')
user_reviews = pd.read_parquet('Data_Consumible/user_reviews.parquet')
reviews_posted = pd.read_parquet('Data_Consumible/reviews_posted.parquet')


#-------------------------------------------------------------------------------------------------------------------------------


@app.get('/userdata/{user_id}')
def userdata(user_id: str):
    usuario_tabla_items = users_items[users_items['user_id'] == user_id]

    if not usuario_tabla_items.empty:
        user_items_ids = usuario_tabla_items['user_items_ids']
        user_items_ids = [int(elemento) for elemento in user_items_ids[0]]
        usuario_steamGames = steam_games[steam_games['id'].isin(user_items_ids)]
        Cantidad_gastado = usuario_steamGames['price'].sum() - usuario_steamGames['discount_price'].sum()

        usuario_tabla_reviews = user_reviews[user_reviews['user_id'] == user_id]
        Porcentaje_recomendación = usuario_tabla_reviews['porcentaje_recomendacion'].values[0].item()

        Cantidad_items = usuario_tabla_items['items_count'].values[0].item()

        return {
            'Cantidad de dinero gastado': Cantidad_gastado,
            'Porcentaje de recomendación': Porcentaje_recomendación,
            'Cantidad de items': Cantidad_items
        }
    else:
        return {'Usuario no existe'}


#-------------------------------------------------------------------------------------------------------------------------------


@app.get('/countreviews/{fecha_inicial}, {fecha_final}')
def countreviews(fecha_inicial : str, fecha_final : str):

    # Analizar las fechas usando dateutil.parser
    fecha_inicial = dparser.parse(fecha_inicial, fuzzy=True)
    fecha_final = dparser.parse(fecha_final, fuzzy=True)
    fecha_inicial = pd.to_datetime(fecha_inicial)
    fecha_final = pd.to_datetime(fecha_final)

    # Definir una función para filtrar las filas en función del rango de fechas
    def filtrar_por_rango(fila, fecha_inicial, fecha_final):
        posted = fila['posted']
        last_edited = fila['last_edited']

        # Verificar si al menos una de las dos fechas está dentro del rango
        if (fecha_inicial <= posted <= fecha_final) or (fecha_inicial <= last_edited <= fecha_final):
            return True
        else:
            return False

    # Filtrar el DataFrame según el rango de fechas
    df_filtrado = reviews_posted[reviews_posted.apply(filtrar_por_rango, args=(fecha_inicial, fecha_final), axis=1)]

    # Calcular la cantidad de usuarios únicos
    Cantidad_usuarios = len(df_filtrado['user_id'].unique())

    # Calcular la cantidad de recomendaciones positivas (True)
    cantidad_recomendations = len(df_filtrado)
    mask_true = df_filtrado['recomendation'] == True
    cantidad_true = mask_true.sum()

    # Calcular el porcentaje de recomendación
    Porcentaje_recomendacion = (cantidad_true / cantidad_recomendations) * 100

    # Redondear el porcentaje a dos decimales
    Porcentaje_recomendacion = round(Porcentaje_recomendacion, 2)

    return {'Cantidad de usuarios': Cantidad_usuarios,
            'Porcentaje de recomendación': Porcentaje_recomendacion}
    

#-------------------------------------------------------------------------------------------------------------------------------


@app.get('/genre/{genero}')

def genre(genero : str):

  with open('Data_Consumible/generos_ranking.json', 'r') as archivo:
    generos_ranking = json.load(archivo)

  # Convierte las claves del diccionario en una lista
  generos_ordenados = list(generos_ranking.keys())

  # Verifica si el género buscado está en la lista
  if genero in generos_ordenados:
      Puesto_ranking = generos_ordenados.index(genero) + 1
      return {'Puesto en el ranking': Puesto_ranking}
  else:
      return print(f"'{genero}' no se encuentra en la lista de géneros.")


#-------------------------------------------------------------------------------------------------------------------------------


@app.get('/userforgenre/{genero}')
def userforgenre(genero : str):

    with open('Data_Consumible/generos_usuarios.json', 'r') as archivo:
        generos_usuarios = json.load(archivo)

    # Verificar si el género existe en el diccionario
    if genero not in generos_usuarios:
        return None

    # Obtener el diccionario de usuarios y playtime para el género dado
    usuarios_playtime = generos_usuarios[genero]

    # Ordenar el diccionario por el segundo elemento de las listas (playtime)
    genero_ordenado = dict(sorted(usuarios_playtime.items(), key=lambda x: x[1][1], reverse=True))

    # Tomar los cinco mayores elementos
    cinco_mayores = dict(list(genero_ordenado.items())[:5])

    return {f'Top 5 de usuarios con más horas de juego en el género {genero} ': cinco_mayores}


#-------------------------------------------------------------------------------------------------------------------------------


@app.get('/developer/{desarrollador}')
def developer(desarrollador):

    # Filtrar el DataFrame por desarrollador y hacer una copia explícita
    df_filtrado = steam_games[steam_games['developer'] == desarrollador].copy()

    # Verificar si el DataFrame filtrado está vacío
    if df_filtrado.empty:
        return "No se encontraron juegos para el desarrollador especificado."

    # Extraer el año de la columna 'release_date'
    df_filtrado['release_year'] = df_filtrado['release_date'].dt.year

    # Obtener los años únicos
    años_unicos = df_filtrado['release_year'].unique()

    # Crear un diccionario para almacenar los resultados
    resultados = {}

    # Iterar a través de los años y contar la cantidad de juegos lanzados en cada año
    for año in años_unicos:
        cantidad_juegos = len(df_filtrado[df_filtrado['release_year'] == año])
        
        # Filtrar el DataFrame para obtener juegos gratuitos en el año actual
        juegos_gratuitos = df_filtrado[(df_filtrado['release_year'] == año) & (df_filtrado['price'] == 0)]
        cantidad_gratuitos = len(juegos_gratuitos)
        
        # Almacenar los resultados en un diccionario
        resultados[año] = {
            'Cantidad de items': cantidad_juegos,
            'Porcentaje de contenido FREE': (cantidad_gratuitos/cantidad_juegos)*100
        }

    return resultados
    

#-------------------------------------------------------------------------------------------------------------------------------


@app.get('/sentiment_analysis/{año}')
def sentiment_analysis(año : int):

    # Filtra las filas donde el año en 'posted' o 'last_edited' sea igual al año proporcionado
    df_filtrado = reviews_posted[(reviews_posted['posted'].dt.year == año) | (reviews_posted['last_edited'].dt.year == año)]

    # Calcula la cantidad de comentarios negativos, neutrales y positivos
    Cantidad_negativos = (df_filtrado['comment'] == 0).sum()
    Cantidad_neutrales = (df_filtrado['comment'] == 1).sum()
    Cantidad_positivos = (df_filtrado['comment'] == 2).sum()

    return {'Negative': Cantidad_negativos,
            'Neutral': Cantidad_neutrales,
            'Positive': Cantidad_positivos}
    

#-------------------------------------------------------------------------------------------------------------------------------


@app.get('/recomendacion/{titulo}')
def recomendacion_juego(id_de_producto : int, n=5):

    # Crear un vectorizador de recuento (CountVectorizer)
    vectorizer = CountVectorizer()

    # Convertir las listas de especificaciones en cadenas de texto
    especificaciones_texto = [" ".join(map(str, especificaciones)) for especificaciones in steam_games['especificaciones']]

    # Obtener las representaciones vectoriales de las especificaciones
    matriz_vectorial = vectorizer.fit_transform(especificaciones_texto)

    # Obtener la representación vectorial del juego dado su ID
    juego_seleccionado = steam_games[steam_games['id'] == id_de_producto]

    if juego_seleccionado.empty:
        return "ID de juego no encontrado"

    vector_juego_seleccionado = vectorizer.transform([" ".join(map(str, juego_seleccionado['especificaciones'].values[0]))])

    # Calcular la similitud de Jaccard entre el juego dado y todos los juegos en el DataFrame
    similitudes = []
    for i in range(matriz_vectorial.shape[0]):
        # Calcular la intersección y unión de conjuntos
        interseccion = sum((vector_juego_seleccionado.toarray() & matriz_vectorial[i].toarray())[0])
        union = sum((vector_juego_seleccionado.toarray() | matriz_vectorial[i].toarray())[0])

        # Calcular la similitud de Jaccard como la intersección dividida por la unión
        similitud = interseccion / union if union > 0 else 0.0
        similitudes.append(similitud)

    # Obtener los índices de los juegos más similares (excluyendo el juego dado)
    indices_similares = sorted(range(len(similitudes)), key=lambda i: similitudes[i], reverse=True)[:n+1]
    indices_similares = [i for i in indices_similares if steam_games.iloc[i]['id'] != id_de_producto]

    # Obtener los IDs y nombres recomendados (en lugar de especificaciones) sin el juego dado
    recomendaciones = [(steam_games.iloc[i]['id'], steam_games.iloc[i]['app_name']) for i in indices_similares]

    return recomendaciones