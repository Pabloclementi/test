import streamlit as st
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from google.cloud import bigquery
from google.oauth2 import service_account

# Leer las credenciales desde los secretos de Streamlit
credentials_json = st.secrets["GOOGLE_CREDENTIALS"]

# Configurar la conexión a BigQuery usando las credenciales desde el secreto
credentials = service_account.Credentials.from_service_account_info(credentials_json)
client = bigquery.Client(credentials=credentials, location="us-central1")

# Reemplaza esta URL con la URL de tu imagen de fondo
background_image_url = "https://cdn.prod.website-files.com/5ddedd0e3047ab406ee3c37e/64aeef75a9175bfa44144333_Stadium_8.0.jpg"
st.markdown(f"""
    <style>
        .stApp {{
            background-image: url("{background_image_url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: #E0E0E0; /* Color del texto en general */
        }}
        .title {{
            font-size: 2.5em;
            font-weight: bold;
            text-align: center;
            padding-top: 20px;
            color: #FFFFFF; /* Color del título en blanco */
            text-shadow: 2px 2px 6px #000; /* Sombra del texto para mejorar la legibilidad */
        }}
        .stTextInput > label {{
            color: #FFFFFF; /* Color de las etiquetas de los campos de texto */
        }}
        .stSelectbox > label {{
            color: #FFFFFF; /* Color de las etiquetas de los campos de selección */
        }}
        .stTextInput>div>input {{
            color: #000000; /* Color del texto dentro del campo de entrada de texto */
            background-color: #FFFFFF; /* Fondo blanco del campo de entrada de texto */
        }}
        .stSelectbox>div>input {{
            color: #000000; /* Color del texto dentro del campo de selección */
            background-color: #FFFFFF; /* Fondo blanco del campo de selección */
        }}
    </style>
""", unsafe_allow_html=True)

# Verificar si el archivo existe antes de cargarlo
def load_data():
    query = """
    SELECT 
        stadium,
        name,
        avg_rating,
        num_of_reviews,
        gmap_id,
        url,
        category
    FROM 
        divine-builder-431018-g4.horizon.Google_metadata
    """
    estados = client.query(query).to_dataframe()
    return estados

estados = load_data()

# Definir las funciones
def filter_by_city(estados, stadium):
    return estados[estados['stadium'] == stadium]

def find_similar_restaurants(name, stadium, num_recommendations=5):
    city_restaurants = filter_by_city(estados, stadium)
    
    if name not in city_restaurants['name'].values:
        raise ValueError(f"El restaurante '{name}' no existe cerca del estadio '{stadium}'.")
    
    city_restaurants = city_restaurants.drop_duplicates(subset=['name'])
    
    # Asegurarse de que no haya duplicados en el índice
    restaurant_matrix = city_restaurants.pivot_table(index='name', values=['avg_rating', 'num_of_reviews'])
    
    mlb = MultiLabelBinarizer()
    categories = mlb.fit_transform(city_restaurants['category'].apply(lambda x: x.split(',')))
    categories_df = pd.DataFrame(categories, index=city_restaurants['name'], columns=mlb.classes_)
    
    combined_matrix = pd.concat([restaurant_matrix, categories_df], axis=1)
    
    # Asegurar que el índice sea único después de la concatenación
    combined_matrix = combined_matrix[~combined_matrix.index.duplicated(keep='first')]
    
    combined_matrix_filled = combined_matrix.fillna(0)
    
    # Calcular la similitud coseno
    similarity_matrix = cosine_similarity(combined_matrix_filled)
    similarity_df = pd.DataFrame(similarity_matrix, index=combined_matrix.index, columns=combined_matrix.index)
    
    # Obtener restaurantes similares
    similar_restaurants = similarity_df[name].sort_values(ascending=False).head(num_recommendations + 1).index.tolist()
    similar_restaurants.remove(name)
    
    recommendations = estados[estados['name'].isin(similar_restaurants) & (estados['stadium'] == stadium)].drop_duplicates(subset=['name']).head(num_recommendations)
    
    return recommendations[['name', 'avg_rating', 'url', 'stadium']]

# Interfaz de usuario con Streamlit
st.title("Recomendación de Restaurantes")

# Campo de selección para estadios
stadium = st.selectbox("Selecciona el estadio", estados['stadium'].unique(), key="stadium", help="Selecciona un estadio de la lista")

# Input field para el nombre del restaurante
name = st.text_input("Nombre del restaurante", key="name", help="Escribe el nombre del restaurante")

# Botón para obtener recomendaciones
if st.button("Obtener recomendaciones"):
    if name and stadium:
        try:
            recommendations = find_similar_restaurants(name, stadium)
            st.write(f"Recomendaciones para {name} cerca de {stadium}:")
            for _, row in recommendations.iterrows():
                st.markdown(f"""
                    <div class='restaurant-card'>
                        <h4>{row['name']}</h4>
                        <p>Calificación: {row['avg_rating']}</p>
                        <a href="{row['url']}" target="_blank">Ver en Google Maps</a>
                    </div>
                """, unsafe_allow_html=True)
        except ValueError as e:
            st.error(str(e))
    else:
        st.warning("Por favor, ingresa el nombre del restaurante y selecciona un estadio.")
