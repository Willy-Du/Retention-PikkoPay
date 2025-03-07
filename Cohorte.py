import pandas as pd
import numpy as np
import os
import streamlit as st
import plotly.graph_objects as go
import matplotlib.cm as cm
import matplotlib
from datetime import datetime
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# 📌 Connexion MongoDB
client = MongoClient(MONGO_URI)
db = client['storesDatabase']
users_collection = db['usertests']


# 📌 Définition de la période
date_start = datetime(2024, 6, 10, 0, 0, 0)
date_end = datetime.now()
store_id = ObjectId("65e6388eb6667e3400b5b8d8")


pipeline_new_users = [
    {"$unwind": "$receipt"},
    {"$match": {
        "receipt.isPaid": True,
        "receipt.storeId": store_id,
        "receipt.paidAt": {"$gte": date_start, "$lte": date_end}
    }},
    {"$sort": {"receipt.paidAt": 1}},
    {"$group": {
        "_id": "$_id",
        "firstPaidAt": {"$first": "$receipt.paidAt"},
        "createdAt": {"$first": "$createdAt"}
    }},
    {"$match": {
        # Filtre pour garder seulement les utilisateurs dont la date de création est dans la même semaine ISO que leur premier paiement
        "$expr": {
            "$and": [
                {"$eq": [{"$isoWeekYear": "$firstPaidAt"}, {"$isoWeekYear": "$createdAt"}]},
                {"$eq": [{"$isoWeek": "$firstPaidAt"}, {"$isoWeek": "$createdAt"}]}
            ]
        }
    }},
    {"$group": {
        "_id": {"year": {"$isoWeekYear": "$firstPaidAt"}, "week": {"$isoWeek": "$firstPaidAt"}},
        "new_users": {"$addToSet": "$_id"}
    }},
    {"$project": {"_id": 1, "total_new_users": {"$size": "$new_users"}, "new_users": 1}},
    {"$sort": {"_id.year": 1, "_id.week": 1}}
]

# 📌 Pipeline pour récupérer les utilisateurs actifs par semaine
pipeline_active_users = [
    {"$unwind": "$receipt"},
    {"$match": {
        "receipt.isPaid": True,
        "receipt.paidAt": {"$gte": date_start, "$lte": date_end},
        "receipt.storeId": store_id
    }},
    {"$group": {
        "_id": {"year": {"$isoWeekYear": "$receipt.paidAt"}, "week": {"$isoWeek": "$receipt.paidAt"}},
        "active_users": {"$addToSet": "$_id"}
    }},
    {"$sort": {"_id.year": 1, "_id.week": 1}}
]
# 📌 Exécuter les requêtes MongoDB
cursor_new_users = users_collection.aggregate(pipeline_new_users)
cursor_active_users = users_collection.aggregate(pipeline_active_users)

data_new_users = list(cursor_new_users)
data_active_users = list(cursor_active_users)

# 📌 Vérification des données
if not data_new_users or not data_active_users:
    st.error("❌ Aucune donnée trouvée ! Vérifiez la structure de votre base MongoDB.")
    st.stop()

# 📌 Transformation en DataFrame
df_new_users = pd.DataFrame(data_new_users)
df_active_users = pd.DataFrame(data_active_users)

# 📌 Extraction des années et semaines
df_new_users['year'] = df_new_users['_id'].apply(lambda x: x['year'])
df_new_users['week'] = df_new_users['_id'].apply(lambda x: x['week'])

df_active_users['year'] = df_active_users['_id'].apply(lambda x: x['year'])
df_active_users['week'] = df_active_users['_id'].apply(lambda x: x['week'])

# 📌 Générer la colonne du début de semaine
df_new_users['week_start'] = df_new_users.apply(lambda x: datetime.fromisocalendar(x['year'], x['week'], 1), axis=1)
df_active_users['week_start'] = df_active_users.apply(lambda x: datetime.fromisocalendar(x['year'], x['week'], 1), axis=1)

# 📌 Convertir `ObjectId` en `str`
df_new_users['new_users'] = df_new_users['new_users'].apply(lambda users: set(str(u) for u in users) if users else set())
df_active_users['active_users'] = df_active_users['active_users'].apply(lambda users: set(str(u) for u in users) if users else set())

# 📌 Trier et indexer les données
df_new_users = df_new_users.sort_values(by='week_start').set_index('week_start')
df_active_users = df_active_users.sort_values(by='week_start').set_index('week_start')

# 📌 Générer la liste complète des semaines entre la première et la dernière
all_weeks = pd.date_range(start=df_new_users.index.min(), end=df_new_users.index.max(), freq='W-MON')

# 📌 Créer un DataFrame pour toutes les semaines avec la colonne total_new_users initialisée à 0
all_weeks_df = pd.DataFrame(index=all_weeks)
all_weeks_df['total_new_users'] = 0

# 📌 Mettre à jour les valeurs pour les semaines qui ont des données
for idx in df_new_users.index:
    if idx in all_weeks_df.index:
        all_weeks_df.loc[idx, 'total_new_users'] = df_new_users.loc[idx, 'total_new_users']

# 📌 Calcul de rétention
user_retention = {}

# 📌 Initialiser toutes les semaines avec +0 = 0 par défaut
for idx in all_weeks:
    user_retention[idx] = {"+0": 0}

# 📌 Mettre à jour les données de rétention pour les semaines avec de nouveaux utilisateurs
for index, row in df_new_users.iterrows():
    new_user_set = row['new_users']
    user_retention[index]["+0"] = len(new_user_set)

    if not new_user_set:
        continue

    future_weeks = df_active_users.loc[df_active_users.index > index]
    for week_diff, (future_index, future_row) in enumerate(future_weeks.iterrows(), 1):
        future_users = future_row['active_users']
        retained_users = len(new_user_set.intersection(future_users)) if isinstance(future_users, set) else 0
        user_retention[index][f"+{week_diff}"] = retained_users

# 📌 Convertir les données de rétention en DataFrame
df_retention = pd.DataFrame.from_dict(user_retention, orient='index')

# 📌 S'assurer que toutes les colonnes +N existent et sont remplies avec des 0 si nécessaire
max_week_diff = max([int(col.replace("+", "")) for col in df_retention.columns if col.startswith("+")])
for week_diff in range(max_week_diff + 1):
    col_name = f"+{week_diff}"
    if col_name not in df_retention.columns:
        df_retention[col_name] = 0
    else:
        df_retention[col_name] = df_retention[col_name].fillna(0)

# 📌 Fusion avec le DataFrame de toutes les semaines
df_numeric = all_weeks_df.merge(df_retention, left_index=True, right_index=True, how='left')

# 📌 Définir la date actuelle pour déterminer quelles semaines sont dans le futur
current_date = datetime.now()
current_week_start = datetime.fromisocalendar(current_date.year, current_date.isocalendar()[1], 1)

# 📌 Déterminer la dernière semaine disponible pour chaque cohorte
last_available_week = {}
for index, row in df_numeric.iterrows():
    plus_columns = [col for col in df_numeric.columns if col.startswith("+")]
    plus_columns.sort(key=lambda x: int(x.replace("+", "")))
    
    last_week = 0
    for col in plus_columns:
        week_num = int(col.replace("+", ""))
        future_week = index + pd.Timedelta(weeks=week_num)
        
        # Si la semaine est dans le futur par rapport à aujourd'hui, arrêter
        if future_week > current_week_start:
            break
        last_week = week_num
    
    last_available_week[index] = last_week

# 📌 Remplacer les valeurs NaN par 0 pour les colonnes passées et actuelles, et laisser None pour les semaines futures
for index, row in df_numeric.iterrows():
    plus_columns = [col for col in df_numeric.columns if col.startswith("+")]
    plus_columns.sort(key=lambda x: int(x.replace("+", "")))
    
    max_week = last_available_week[index]
    
    for col in plus_columns:
        week_num = int(col.replace("+", ""))
        if week_num <= max_week:
            # Remplacer par 0 uniquement si la semaine est dans le passé ou actuelle
            if pd.isna(row[col]):
                df_numeric.at[index, col] = 0
        else:
            # Pour les semaines futures, mettre explicitement à None
            df_numeric.at[index, col] = None

# 📌 Création d'une copie pour les pourcentages
df_percentage = df_numeric.copy()

# 📌 Calculer les pourcentages uniquement pour les données disponibles (non None)
for col in df_percentage.columns:
    if col.startswith("+") and col != "+0":
        # Pour éviter la division par zéro et ne calculer que pour les valeurs non-None
        mask = (df_percentage["+0"] > 0) & (df_percentage[col].notna())
        df_percentage.loc[mask, col] = (df_percentage.loc[mask, col] / df_percentage.loc[mask, "+0"] * 100).round(1)

# 📌 Fixer +0 à 100% où il y a des utilisateurs, et 0% ailleurs
df_percentage["+0"] = df_percentage["+0"].apply(lambda x: 100 if x > 0 else 0)

# 📌 Appliquer le dégradé de rouge pour la coloration des cellules avec traitement spécial pour les cellules futures
def apply_red_gradient_with_future(val):
    """ 
    Applique un dégradé de rouge pour les valeurs disponibles
    et masque les cellules correspondant aux semaines futures
    """
    if pd.isna(val):
        # Style pour les semaines futures non disponibles
        return 'background-color: #f0f0f0; color: #f0f0f0;'
    elif pd.notna(val):
        # Dégradé de rouge pour les valeurs disponibles
        intensity = int(255 * ((1 - val / 100) ** 3))  # Exposant pour un meilleur contraste
        return f'background-color: rgba(255, {intensity}, {intensity}, 1); color: black;'
    return ''

# 📌 Afficher le tableau avec les valeurs numériques et les pourcentages avec le dégradé
st.header("📅 Tableau des cohortes hebdomadaires")
st.subheader("📊 Cohorte hebdomadaire (valeurs numériques)")
# On utilise df_numeric pour le premier tableau
st.dataframe(df_numeric)
st.subheader("📊 Cohorte hebdomadaire (%)")
# On utilise df_percentage pour le deuxième tableau avec coloration et gestion des semaines futures
st.dataframe(df_percentage.style.applymap(apply_red_gradient_with_future, subset=[col for col in df_percentage.columns if col.startswith("+")]))


# 📌 Ajout du Line Chart
st.title("📈 Évolution du pourcentage d'utilisateurs par cohorte")

# 🔥 Préparation des données pour le Line Chart
percentage_data_cleaned = df_percentage.copy()
percentage_data_cleaned = percentage_data_cleaned.drop(columns=['total_new_users'], errors='ignore')

# 🔥 Suppression des valeurs après le premier NaN
for index, row in percentage_data_cleaned.iterrows():
    first_nan = row.isna().idxmax() if row.isna().any() else None
    if first_nan and first_nan != "+0":
        percentage_data_cleaned.loc[index, first_nan:] = np.nan


# 🔥 Création du graphique interactif avec Plotly
fig = go.Figure()

# 🔥 Création d'une palette de couleurs distinctes pour les cohortes
colormap = cm.get_cmap('tab20c', len(percentage_data_cleaned))

# 🔥 Tracer chaque cohorte avec une couleur unique (les afficher en premier)
for i, (index, row) in enumerate(percentage_data_cleaned.iterrows()):
    valid_values = row[row.notna()]
    if "+0" not in valid_values.index:
        continue

    rgba_color = colormap(i / len(percentage_data_cleaned))
    color = f'rgba({int(rgba_color[0] * 255)}, {int(rgba_color[1] * 255)}, {int(rgba_color[2] * 255)}, {rgba_color[3]})'

    fig.add_trace(go.Scatter(
        x=valid_values.index,
        y=valid_values.values,
        mode='lines',
        name=f'Cohorte {index.strftime("%Y-%m-%d")}',
        line=dict(width=2, color=color),
        hoverinfo='x+y',
        opacity=0.8
    ))

# 🔥 Calcul de la courbe moyenne sur toutes les cohortes
average_curve = percentage_data_cleaned.mean(axis=0, skipna=True)

# 🔥 Ajouter la courbe de moyenne en dernier pour être au-dessus
fig.add_trace(go.Scatter(
    x=average_curve.index,
    y=average_curve.values,
    mode='lines',
    name='Moyenne par +x',
    line=dict(width=3, color='black'),  # Augmenter l'épaisseur et mettre en noir
    opacity=1.0
))

# 🔥 Mise en forme et affichage de toutes les semaines sur X
fig.update_layout(
    title="📊 Rétention des utilisateurs par semaine",
    xaxis_title="Semaines après premier achat",
    yaxis_title="Pourcentage de rétention",
    template="plotly_white",
    xaxis=dict(
        tickmode='array',
        tickvals=[f'+{i}' for i in range(len(average_curve))]  # Afficher toutes les semaines
    )
)

# 🔥 Afficher le graphique dans Streamlit
st.plotly_chart(fig)
# ========================
# 📅 Cohortes par MOIS (nouveaux utilisateurs ayant payé ce mois-là)
# ========================

st.header("📅 Tableau des cohortes mensuelles")

# 📌 Définition du `store_id`
store_id = ObjectId("65e6388eb6667e3400b5b8d8")

# 📌 Pipeline pour récupérer les nouveaux utilisateurs par mois
pipeline_new_users = [
    {"$unwind": "$receipt"},
    {"$match": {
        "receipt.isPaid": True,
        "receipt.storeId": store_id,
        "receipt.paidAt": {"$gte": date_start, "$lte": date_end}
    }},
    {"$sort": {"receipt.paidAt": 1}},
    {"$group": {
        "_id": "$_id",
        "firstPaidAt": {"$first": "$receipt.paidAt"}
    }},
    {"$group": {
        "_id": {"year": {"$year": "$firstPaidAt"}, "month": {"$month": "$firstPaidAt"}},
        "new_users": {"$addToSet": "$_id"}
    }},
    {"$project": {"_id": 1, "total_new_users": {"$size": "$new_users"}, "new_users": 1}},
    {"$sort": {"_id.year": 1, "_id.month": 1}}
]

# 📌 Pipeline pour récupérer les utilisateurs actifs par mois
pipeline_active_users = [
    {"$unwind": "$receipt"},
    {"$match": {
        "receipt.isPaid": True,
        "receipt.paidAt": {"$gte": date_start, "$lte": date_end},
        "receipt.storeId": store_id
    }},
    {"$group": {
        "_id": {"year": {"$year": "$receipt.paidAt"}, "month": {"$month": "$receipt.paidAt"}},
        "active_users": {"$addToSet": "$_id"}
    }},
    {"$sort": {"_id.year": 1, "_id.month": 1}}
]

# 📌 Exécuter les requêtes MongoDB
cursor_new_users = users_collection.aggregate(pipeline_new_users)
cursor_active_users = users_collection.aggregate(pipeline_active_users)

data_new_users = list(cursor_new_users)
data_active_users = list(cursor_active_users)

# 📌 Vérification des données
if not data_new_users or not data_active_users:
    st.error("❌ Aucune donnée trouvée ! Vérifiez la structure de votre base MongoDB.")
    st.stop()

# 📌 Transformation en DataFrame
df_new_users = pd.DataFrame(data_new_users)
df_active_users = pd.DataFrame(data_active_users)

# 📌 Extraction des années et mois
df_new_users['year'] = df_new_users['_id'].apply(lambda x: x['year'])
df_new_users['month'] = df_new_users['_id'].apply(lambda x: x['month'])

df_active_users['year'] = df_active_users['_id'].apply(lambda x: x['year'])
df_active_users['month'] = df_active_users['_id'].apply(lambda x: x['month'])

# 📌 Générer la colonne du début de mois
df_new_users['month_start'] = df_new_users.apply(lambda x: datetime(x['year'], x['month'], 1), axis=1)
df_active_users['month_start'] = df_active_users.apply(lambda x: datetime(x['year'], x['month'], 1), axis=1)

# 📌 Convertir `ObjectId` en `str`
df_new_users['new_users'] = df_new_users['new_users'].apply(lambda users: set(str(u) for u in users) if users else set())
df_active_users['active_users'] = df_active_users['active_users'].apply(lambda users: set(str(u) for u in users) if users else set())

# 📌 Trier et indexer les données
df_new_users = df_new_users.sort_values(by='month_start').set_index('month_start')
df_active_users = df_active_users.sort_values(by='month_start').set_index('month_start')

# ✅ Calcul de rétention mensuelle
monthly_retention = {}

for index, row in df_new_users.iterrows():
    new_user_set = row['new_users']
    monthly_retention[index] = {"+0": len(new_user_set)}

    if not new_user_set:
        continue

    future_months = df_active_users.loc[df_active_users.index >= index]
    for month_diff, (future_index, future_row) in enumerate(future_months.iterrows()):
        future_users = future_row['active_users']
        retained_users = len(new_user_set.intersection(future_users)) if isinstance(future_users, set) else 0  
        monthly_retention[index][f"+{month_diff}"] = retained_users

# ✅ Convertir les données de rétention en DataFrame
df_monthly_retention = pd.DataFrame.from_dict(monthly_retention, orient='index')

# ✅ Fusionner avec le DataFrame principal
df_final = df_new_users[['total_new_users']].merge(df_monthly_retention, left_index=True, right_index=True, how='left')

# ✅ Afficher les valeurs absolues (sans style)
st.subheader("📊 Cohortes mensuelles (valeurs numériques)")
st.dataframe(df_final)

# ✅ Calculer les pourcentages de rétention **en utilisant les valeurs absolues d'origine**
df_percentage = df_final.copy()

# ✅ Créer une copie des valeurs d'origine pour le calcul des pourcentages
df_percentage_calcul = df_final.copy()

# ✅ Assurer que toutes les cohortes commencent à 100%
df_percentage["+0"] = 100  # Fixer +0 à 100%

# ✅ Calcul des pourcentages pour les autres colonnes **en utilisant la copie d'origine**
for col in df_percentage.columns:
    if col.startswith("+") and col != "+0":
        df_percentage[col] = (df_percentage_calcul[col] / df_percentage_calcul["+0"] * 100).round(1)

def apply_red_gradient(val):
    """ Accentue le dégradé de rouge : rouge foncé pour 100%, blanc pour 0% """
    if pd.notna(val):
        # Calcul du rouge avec un contraste plus fort
        intensity = int(255 * ((1 - val / 100) ** 3))  # Exposant pour un contraste plus visible
        return f'background-color: rgba(255, {intensity}, {intensity}, 1); color: black;'
    return ''


st.subheader("📊 Cohortes mensuelles (%)")
st.dataframe(df_percentage.style.applymap(apply_red_gradient, subset=[col for col in df_percentage.columns if col.startswith("+")]))

# 🔥 Préparation des données pour la Line Chart
percentage_data_cleaned = df_percentage.copy()

# 🔥 Suppression des valeurs après le premier NaN
for index, row in percentage_data_cleaned.iterrows():
    first_nan = row.isna().idxmax() if row.isna().any() else None
    if first_nan and first_nan != "+0":
        percentage_data_cleaned.loc[index, first_nan:] = np.nan


# ✅ Supprimer la colonne "total_new_users" pour éviter qu'elle soit utilisée dans le graphique
if "total_new_users" in df_percentage.columns:
    df_percentage = df_percentage.drop(columns=["total_new_users"])

# ✅ Prendre uniquement les colonnes de rétention (+0, +1, +2, ...)
percentage_data_cleaned = df_percentage.copy()

# ✅ Suppression des valeurs après le premier NaN pour éviter les erreurs
for index, row in percentage_data_cleaned.iterrows():
    first_nan = row.isna().idxmax() if row.isna().any() else None
    if first_nan and first_nan != "+0":
        percentage_data_cleaned.loc[index, first_nan:] = np.nan

# ✅ Création du graphique interactif avec Plotly
fig = go.Figure()

# ✅ Création d'une palette de couleurs distinctes pour les cohortes
colormap = cm.get_cmap('tab20c', len(percentage_data_cleaned))

# ✅ Tracer chaque cohorte avec une couleur unique
for i, (index, row) in enumerate(percentage_data_cleaned.iterrows()):
    valid_values = row[row.notna()]
    if "+0" not in valid_values.index:
        continue

    rgba_color = colormap(i / len(percentage_data_cleaned))
    color = f'rgba({int(rgba_color[0] * 255)}, {int(rgba_color[1] * 255)}, {int(rgba_color[2] * 255)}, {rgba_color[3]})'

    fig.add_trace(go.Scatter(
        x=valid_values.index,
        y=valid_values.values,
        mode='lines',
        name=f'Cohorte {index.strftime("%Y-%m-%d")}',
        line=dict(width=2, color=color),
        hoverinfo='x+y',
        opacity=0.8
    ))

# ✅ Calcul de la courbe moyenne sur toutes les cohortes en pourcentage
average_curve = percentage_data_cleaned.mean(axis=0, skipna=True)

# ✅ Ajouter la courbe de moyenne en dernier pour être bien visible
fig.add_trace(go.Scatter(
    x=average_curve.index,
    y=average_curve.values,
    mode='lines',
    name='Moyenne par +x',
    line=dict(width=3, color='black'),  # Épaisseur plus grande et couleur noire
    opacity=1.0
))

# ✅ Mise en forme et affichage de tous les mois sur X
fig.update_layout(
    title="📊 Rétention des utilisateurs par mois (%)",
    xaxis_title="Mois après premier achat",
    yaxis_title="Pourcentage de rétention",
    template="plotly_white",
    xaxis=dict(
        tickmode='array',
        tickvals=[f'+{i}' for i in range(len(average_curve))]  # Afficher toutes les semaines
    ),
    yaxis=dict(
        tickformat=".1f",  # Format des valeurs Y pour afficher 1 décimale
        range=[0, 110]  # Assurer que les valeurs restent entre 0 et 100%
    )
)

# ✅ Afficher le graphique dans Streamlit
st.plotly_chart(fig)
