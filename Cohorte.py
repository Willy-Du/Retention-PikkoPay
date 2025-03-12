import pandas as pd
import numpy as np
import os
import streamlit as st
import plotly.graph_objects as go
import matplotlib.cm as cm
import matplotlib
from datetime import datetime, timedelta
from collections import defaultdict
import plotly.express as px 
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# 📌 Connexion MongoDB
client = MongoClient(MONGO_URI)
db = client['storesDatabase']
users_collection = db['usertests']

orders_collection = db['ordertests']


# 📌 Définition de la période
date_start = datetime(2024, 6, 10, 0, 0, 0)
date_end = datetime.now()
store_id = ObjectId("65e6388eb6667e3400b5b8d8")



# 📌 Ajout du menu de navigation dans la barre latérale
st.sidebar.title("📊 Dashboard de suivi")
page = st.sidebar.radio(
    "Choisissez une section :", 
    ["Rétention", "Acquisition", "Weekly Active Users", "Bug Report"]
)

# ========================
# Partie Rétention
# ========================
if page == "Rétention":
# 📌 Pipeline pour récupérer les nouveaux utilisateurs par semaine de leur premier paiement
    pipeline_new_users = [
    { "$unwind": "$receipt" },
    { "$match": {
        "receipt.isPaid": True,
        "receipt.storeId": store_id,
        "receipt.paidAt": { "$gte": date_start, "$lte": date_end }
        }
    },
    { "$sort": { "receipt.paidAt": 1 } },
    { "$group": {
        "_id": "$_id",
        "firstPaidAt": { "$first": "$receipt.paidAt" }
        }
    },
    { "$addFields": {
        "firstPaidWeek": { "$isoWeek": "$firstPaidAt" },
        "firstPaidYear": { "$isoWeekYear": "$firstPaidAt" }
        }
    },
    { "$group": {
        "_id": { "year": "$firstPaidYear", "week": "$firstPaidWeek" },
        "new_users": { "$addToSet": "$_id" }
        }
    },
    { "$project": {
        "_id": 1,
        "total_new_users": { "$size": "$new_users" },
        "new_users": 1
        }
    },
    { "$sort": { "_id.year": 1, "_id.week": 1 } }
    ]




    # 📌 Liste des testeurs à exclure
    testers_to_exclude = [
        ObjectId("66df2f59c1271156d5468044"),
        ObjectId("670f97d3f38642c54d678d26"),
        ObjectId("65c65360b03953a598253426"),
        ObjectId("65bcb0e43956788471c88e31")
    ]

    pipeline_active_users = [
    { "$unwind": "$receipt" },
    { "$match": {
        "receipt.isPaid": True,
        "receipt.storeId": store_id,
        "receipt.paidAt": { "$gte": date_start, "$lte": date_end },
        "_id": { "$nin": testers_to_exclude }
        }
    },
    { "$addFields": {
        "paymentWeek": { "$isoWeek": "$receipt.paidAt" },
        "paymentYear": { "$isoWeekYear": "$receipt.paidAt" }
        }
    },
    { "$group": {
        "_id": { "year": "$paymentYear", "week": "$paymentWeek" },
        "active_users": { "$addToSet": "$_id" }
        }
    },
    { "$project": {
        "_id": 1,
        "total_active_users": { "$size": "$active_users" },
        "active_users": 1
        }
    },
    { "$sort": { "_id.year": 1, "_id.week": 1 } }
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
    st.title("Partie Rétention")
    st.header("📅 Tableau des cohortes hebdomadaires")
    st.subheader("📊 Cohorte hebdomadaire (valeurs numériques)")
    # On utilise df_numeric pour le premier tableau
    st.dataframe(df_numeric)
    st.subheader("📊 Cohorte hebdomadaire (%)")
    # On utilise df_percentage pour le deuxième tableau avec coloration et gestion des semaines futures
    st.dataframe(df_percentage.style.applymap(apply_red_gradient_with_future, subset=[col for col in df_percentage.columns if col.startswith("+")]))

    # Récupérer la dernière semaine disponible
    last_week = df_active_users.index.max()

    # Extraire la diagonale du tableau de rétention
    diagonal_values = []
    for idx, row in df_percentage.iterrows():
        week_diff = (last_week - idx).days // 7
        if f"+{week_diff}" in row.index and not pd.isna(row[f"+{week_diff}"]):
            # Utiliser df_numeric pour obtenir les valeurs numériques et non les pourcentages
            diagonal_values.append(df_numeric.loc[idx, f"+{week_diff}"])

    # Somme de la diagonale
    diagonal_sum = sum(diagonal_values)

    # Comparer avec le nombre d'utilisateurs uniques de la dernière semaine
    unique_users_last_week = len(df_active_users.loc[last_week, 'active_users']) if last_week in df_active_users.index else 0

    st.write(f"Somme de la diagonale : {diagonal_sum}")
    st.write(f"Utilisateurs uniques de la dernière semaine : {unique_users_last_week}")

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

    # 📌 Définir la date actuelle pour déterminer quels mois sont dans le futur
    current_date = datetime.now()
    current_month_start = datetime(current_date.year, current_date.month, 1)

    # 📌 Déterminer le dernier mois disponible pour chaque cohorte
    last_available_month = {}
    for index, row in df_final.iterrows():
        plus_columns = [col for col in df_final.columns if col.startswith("+")]
        plus_columns.sort(key=lambda x: int(x.replace("+", "")))
        
        last_month = 0
        for col in plus_columns:
            month_num = int(col.replace("+", ""))
            future_month = index + pd.DateOffset(months=month_num)
            
            # Si le mois est dans le futur par rapport à aujourd'hui, arrêter
            if future_month > current_month_start:
                break
            last_month = month_num
        
        last_available_month[index] = last_month

    # 📌 Remplacer les valeurs NaN par 0 pour les mois passés et actuels, et laisser None pour les mois futurs
    for index, row in df_final.iterrows():
        plus_columns = [col for col in df_final.columns if col.startswith("+")]
        plus_columns.sort(key=lambda x: int(x.replace("+", "")))
        
        max_month = last_available_month[index]
        
        for col in plus_columns:
            month_num = int(col.replace("+", ""))
            if month_num <= max_month:
                # Remplacer par 0 uniquement si le mois est dans le passé ou actuel
                if pd.isna(row[col]):
                    df_final.at[index, col] = 0
            else:
                # Pour les mois futurs, mettre explicitement à None
                df_final.at[index, col] = None

    # ✅ Afficher les valeurs absolues (sans style)
    st.subheader("📊 Cohortes mensuelles (valeurs numériques)")
    st.dataframe(df_final)

    # ✅ Calculer les pourcentages de rétention **en utilisant les valeurs absolues d'origine**
    df_percentage = df_final.copy()

    # ✅ Créer une copie des valeurs d'origine pour le calcul des pourcentages
    df_percentage_calcul = df_final.copy()

    # ✅ Calculer les pourcentages uniquement pour les données disponibles (non None)
    for col in df_percentage.columns:
        if col.startswith("+"):
            if col == "+0":
                # Fixer +0 à 100% pour les cohortes ayant des utilisateurs
                df_percentage[col] = df_percentage_calcul["+0"].apply(lambda x: 100 if x > 0 else 0)
            else:
                # Pour éviter la division par zéro et ne calculer que pour les valeurs non-None
                mask = (df_percentage_calcul["+0"] > 0) & (df_percentage_calcul[col].notna())
                df_percentage.loc[mask, col] = (df_percentage_calcul.loc[mask, col] / df_percentage_calcul.loc[mask, "+0"] * 100).round(1)

    # 📌 Appliquer le dégradé de rouge pour la coloration des cellules avec traitement spécial pour les cellules futures
    def apply_red_gradient_with_future(val):
        """ 
        Applique un dégradé de rouge pour les valeurs disponibles
        et masque les cellules correspondant aux mois futurs
        """
        if pd.isna(val):
            # Style pour les mois futurs non disponibles
            return 'background-color: #f0f0f0; color: #f0f0f0;'
        elif pd.notna(val):
            # Dégradé de rouge pour les valeurs disponibles
            intensity = int(255 * ((1 - val / 100) ** 3))  # Exposant pour un meilleur contraste
            return f'background-color: rgba(255, {intensity}, {intensity}, 1); color: black;'
        return ''

    st.subheader("📊 Cohortes mensuelles (%)")
    st.dataframe(df_percentage.style.applymap(apply_red_gradient_with_future, subset=[col for col in df_percentage.columns if col.startswith("+")]))

    # 🔥 Préparation des données pour la Line Chart
    # Utiliser df_percentage directement, qui a déjà les valeurs futures définies comme None
    percentage_data_cleaned = df_percentage.copy()

    # ✅ Supprimer la colonne "total_new_users" pour éviter qu'elle soit utilisée dans le graphique
    if "total_new_users" in percentage_data_cleaned.columns:
        percentage_data_cleaned = percentage_data_cleaned.drop(columns=["total_new_users"])

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
            tickvals=[f'+{i}' for i in range(len(average_curve))]  # Afficher tous les mois
        ),
        yaxis=dict(
            tickformat=".1f",  # Format des valeurs Y pour afficher 1 décimale
            range=[0, 110]  # Assurer que les valeurs restent entre 0 et 100%
        )
    )

    # ✅ Afficher le graphique dans Streamlit
    st.plotly_chart(fig)


# ========================
# Acquisition des utilisateurs
# ========================
if page == "Acquisition":

    # 📌 Pipeline MongoDB pour récupérer le nombre d'utilisateurs créés par semaine
    pipeline_new_users_per_week = [
        {"$match": {
            "createdAt": {"$gte": date_start, "$lte": date_end}
        }},
        {"$group": {
            "_id": {
                "year": {"$isoWeekYear": "$createdAt"},
                "week": {"$isoWeek": "$createdAt"}
            },
            "new_users": {"$sum": 1}
        }},
        {"$sort": {"_id.year": 1, "_id.week": 1}}
    ]

    # 📌 Exécuter la requête MongoDB
    cursor_new_users_per_week = users_collection.aggregate(pipeline_new_users_per_week)
    data_new_users_per_week = list(cursor_new_users_per_week)

    # 📌 Vérification des données
    if not data_new_users_per_week:
        st.error("❌ Aucune donnée trouvée pour les nouveaux utilisateurs par semaine !")
        st.stop()

    # 📌 Transformation en DataFrame
    df_new_users_per_week = pd.DataFrame(data_new_users_per_week)

    # 📌 Extraction des années et semaines
    df_new_users_per_week['year'] = df_new_users_per_week['_id'].apply(lambda x: x['year'])
    df_new_users_per_week['week'] = df_new_users_per_week['_id'].apply(lambda x: x['week'])

    # 📌 Générer la colonne du début de semaine
    df_new_users_per_week['week_start'] = df_new_users_per_week.apply(
        lambda x: datetime.fromisocalendar(x['year'], x['week'], 1), axis=1
    )

    # 📌 Trier et indexer les données
    df_new_users_per_week = df_new_users_per_week.sort_values(by='week_start').set_index('week_start')

    st.title("Partie Acquisition")
    # 📌 Affichage du tableau des nouveaux utilisateurs par semaine
    st.subheader("📅 Nombre de nouveaux utilisateurs par semaine")
    st.dataframe(df_new_users_per_week['new_users'])

    # 📌 Créer une courbe interactive avec Plotly
    fig = px.line(df_new_users_per_week,
                x=df_new_users_per_week.index,
                y="new_users",
                title="📈 Évolution des nouveaux utilisateurs par semaine",
                labels={"week_start": "Semaine", "new_users": "Nouveaux utilisateurs"},
                markers=True)
    st.subheader("📈 Évolution des nouveaux utilisateurs par semaine")
    st.plotly_chart(fig)

# ========================
# Weekly Active Users 
# ========================
if page == "Weekly Active Users": 
    # 📌 Pipeline pour récupérer le nombre total d'utilisateurs uniques par semaine
    pipeline_unique_users_per_week = [
        {"$unwind": "$receipt"},
        {"$match": {
            "receipt.isPaid": True,
            "receipt.paidAt": {"$gte": date_start, "$lte": date_end},
            "receipt.storeId": store_id
        }},
        {"$group": {
            "_id": {
                "year": {"$isoWeekYear": "$receipt.paidAt"},
                "week": {"$isoWeek": "$receipt.paidAt"}
            },
            "unique_users": {"$addToSet": "$_id"}
        }},
        {"$project": {
            "_id": 1,
            "total_unique_users": {"$size": "$unique_users"}
        }},
        {"$sort": {"_id.year": 1, "_id.week": 1}}
    ]

    # 📌 Exécuter la requête MongoDB
    cursor_unique_users_per_week = users_collection.aggregate(pipeline_unique_users_per_week)
    data_unique_users_per_week = list(cursor_unique_users_per_week)

    # 📌 Vérification des données
    if not data_unique_users_per_week:
        st.error("❌ Aucune donnée trouvée pour les utilisateurs uniques par semaine !")
        st.stop()

    # 📌 Transformation en DataFrame
    df_unique_users_per_week = pd.DataFrame(data_unique_users_per_week)

    # 📌 Extraction des années et semaines
    df_unique_users_per_week['year'] = df_unique_users_per_week['_id'].apply(lambda x: x['year'])
    df_unique_users_per_week['week'] = df_unique_users_per_week['_id'].apply(lambda x: x['week'])

    # 📌 Générer la colonne du début de semaine
    df_unique_users_per_week['week_start'] = df_unique_users_per_week.apply(
        lambda x: datetime.fromisocalendar(x['year'], x['week'], 1), axis=1
    )

    # 📌 Trier et indexer les données
    df_unique_users_per_week = df_unique_users_per_week.sort_values(by='week_start').set_index('week_start')

    # 📌 Générer la liste complète des semaines entre la première et la dernière
    all_weeks = pd.date_range(start=df_unique_users_per_week.index.min(), end=df_unique_users_per_week.index.max(), freq='W-MON')

    # 📌 Créer un DataFrame pour toutes les semaines avec total_unique_users initialisé à 0
    df_all_weeks_unique_users = pd.DataFrame(index=all_weeks)
    df_all_weeks_unique_users['total_unique_users'] = 0

    # 📌 Mettre à jour les valeurs pour les semaines qui ont des données
    for idx in df_unique_users_per_week.index:
        if idx in df_all_weeks_unique_users.index:
            df_all_weeks_unique_users.loc[idx, 'total_unique_users'] = df_unique_users_per_week.loc[idx, 'total_unique_users']

    # 📌 Afficher le tableau des utilisateurs uniques par semaine
    st.title("Partie Weekly active users")
    st.subheader("📊 Tableau utilisateurs uniques par semaine")
    st.dataframe(df_all_weeks_unique_users)

    # 📌 Créer une courbe interactive avec Plotly
    fig = px.line(df_all_weeks_unique_users, 
                x=df_all_weeks_unique_users.index, 
                y="total_unique_users", 
                title="📈 Évolution des utilisateurs uniques par semaine",
                labels={"week_start": "Semaine", "total_unique_users": "Utilisateurs uniques"},
                markers=True)
    st.subheader("📈 Évolution des utilisateurs uniques par semaine")
    st.plotly_chart(fig)

if page == "Bug Report":
    st.title("Partie Bug Report")
    # 📌 Définition de la semaine actuelle (du lundi au dimanche)
    today = datetime.now()
    current_week_start = (today - timedelta(days=today.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    current_week_end = current_week_start + timedelta(days=6, hours=23, minutes=59, seconds=59)

    # 📌 Requête MongoDB pour récupérer les paniers abandonnés de la semaine en cours
    non_finalized_carts = list(orders_collection.find({
        'isPaid': False,
        'createdAt': {'$gte': current_week_start, '$lte': current_week_end},
        'scanItems': {'$exists': True, '$ne': []}  
    }))

    # 📌 Nombre total de paniers abandonnés
    total_non_finalized = len(non_finalized_carts)

    # 📌 Mapping des magasins
    store_mapping = {
        "65e6388eb6667e3400b5b8d8": "Supermarché Match",
        "65d3631ff2cd066ab75434fa": "Intermarché Saint Julien",
        "662bb3234c362c6e79e27020": "Netto Troyes",
        "64e48f4697303382f745cb11": "Carrefour Contact Buchères",
        "65ce4e565a9ffc7e5fe298bb": "Carrefour Market Romilly",
        "65b8bde65a0ef81ff30473bf": "Jils Food",
        "67a8fef293a9fcb4dec991b4": "Intermarché EXPRESS Clamart"
    }

    # 📌 Comptage des paniers abandonnés par magasin
    non_finalized_counts = defaultdict(int)
    for cart in non_finalized_carts:
        store_id = cart.get('storeId')
        store_name = store_mapping.get(str(store_id), "Inconnu")  # Utiliser le nom du store si trouvé, sinon "Inconnu"
        non_finalized_counts[store_name] += 1

    # 📌 Conversion en DataFrame et tri des résultats
    non_finalized_df = pd.DataFrame(list(non_finalized_counts.items()), columns=['Magasin', 'Paniers Abandonnés'])
    non_finalized_df = non_finalized_df.sort_values(by='Paniers Abandonnés', ascending=False)

    # 📌 Affichage des résultats
    st.subheader("🛒 Paniers abandonnés de la semaine")
    st.write(f"Nombre total de paniers abandonnés : {total_non_finalized}")
    st.write(non_finalized_df)