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

# 📌 Calcul de rétention
user_retention = {}

for index, row in df_new_users.iterrows():
    new_user_set = row['new_users']
    user_retention[index] = {"+0": len(new_user_set)}

    if not new_user_set:
        continue

    future_weeks = df_active_users.loc[df_active_users.index > index]
    for week_diff, (future_index, future_row) in enumerate(future_weeks.iterrows(), 1):  # Commencer l'énumération à 1 au lieu de 0
        future_users = future_row['active_users']
        retained_users = len(new_user_set.intersection(future_users)) if isinstance(future_users, set) else 0
        user_retention[index][f"+{week_diff}"] = retained_users

# 📌 Convertir les données de rétention en DataFrame
df_retention = pd.DataFrame.from_dict(user_retention, orient='index')

# 📌 Fusion avec `df_new_users`
df_final = df_new_users[['total_new_users']].merge(df_retention, left_index=True, right_index=True, how='left')

# Tranformer %
df_percentage = df_final.copy()
for col in df_percentage.columns:
    if col.startswith("+") and col != "+0":
        df_percentage[col] = (df_percentage[col] / df_percentage["+0"] * 100).round(1)

# 📌 Fixer +0 à 100% (s'assurer que la colonne +0 ne perturbe pas les autres calculs)
df_percentage["+0"] = 100

# 📌 Calculer les pourcentages pour les autres colonnes
for col in df_percentage.columns:
    if col.startswith("+") and col != "+0":
        df_percentage[col] = (df_percentage[col] / df_percentage["+0"] * 100).round(1)

# 📌 Réindexer et remplir les valeurs manquantes avec NaN
df_percentage = df_percentage.sort_index()  # Assurez-vous que l'index est trié correctement

# 📌 Appliquer le dégradé de rouge
def apply_red_gradient(val):
    """ Accentue le dégradé de rouge : rouge foncé pour 100%, blanc pour 0% """
    if pd.notna(val):
        intensity = int(255 * ((1 - val / 100) ** 3))  # Exposant pour un meilleur contraste
        return f'background-color: rgba(255, {intensity}, {intensity}, 1); color: black;'
    return ''

st.dataframe(df_percentage.style.applymap(apply_red_gradient, subset=[col for col in df_percentage.columns if col.startswith("+")]))



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
st.subheader("📊 Cohortes mensuelles (valeurs absolues)")
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
