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
# Vérifier si la connexion MongoDB est déjà stockée dans `st.session_state`
if "mongo_client" not in st.session_state:
    st.session_state.mongo_client = MongoClient(MONGO_URI)
    st.session_state.db = st.session_state.mongo_client["storesDatabase"]
    st.session_state.users_collection = st.session_state.db["usertests"]
    st.session_state.orders_collection = st.session_state.db["ordertests"]

# Vérifier que la connexion a bien été stockée
users_collection = st.session_state.users_collection
orders_collection = st.session_state.orders_collection

# Définition des magasins avec leur date de lancement
store_mapping = {
    "65e6388eb6667e3400b5b8d8": {"name": "Supermarché Match", "launch_date": datetime(2024, 6, 10)},
    "65d3631ff2cd066ab75434fa": {"name": "Intermarché Saint Julien", "launch_date": datetime(2024, 4, 1)},
    "662bb3234c362c6e79e27020": {"name": "Netto Troyes", "launch_date": datetime(2024, 5, 6)},
    "64e48f4697303382f745cb11": {"name": "Carrefour Contact Buchères", "launch_date": datetime(2023, 11, 5)},
    "65ce4e565a9ffc7e5fe298bb": {"name": "Carrefour Market Romilly", "launch_date": datetime(2024, 2, 16)},
    "65b8bde65a0ef81ff30473bf": {"name": "Jils Food", "launch_date": datetime(2024, 2, 12)},
    "67a8fef293a9fcb4dec991b4": {"name": "Intermarché EXPRESS Clamart", "launch_date": datetime(2025, 3, 3)}
}

# 📌 Connexion MongoDB
client = MongoClient(MONGO_URI)
db = client['storesDatabase']
users_collection = db['usertests']
orders_collection = db['ordertests']


# 📌 Ajout du menu de navigation dans la barre latérale
st.sidebar.title("📊 Dashboard de suivi")

# Sélection du magasin (obligatoire)
displayed_store_id = st.sidebar.selectbox(
    "Sélectionnez un magasin :",
    options=list(store_mapping.keys()),
    format_func=lambda x: store_mapping[x]["name"]
)

date_end = datetime.now()
date_start = store_mapping[displayed_store_id]["launch_date"]
store_filter = ObjectId(displayed_store_id)



page = st.sidebar.radio(
    "Choisissez une section :", 
    ["Rétention", "Acquisition", "Active Users", "Bug Report"]
)

# ========================
# Partie Rétention
# ========================
if page == "Rétention":
    pipeline_new_users_week = [
        {"$unwind": "$receipt"},
        {"$match": {
            "receipt.isPaid": True,
            "receipt.storeId": store_filter,
            "receipt.paidAt": {"$gte": date_start, "$lte": date_end}
        }},
        {"$sort": {"receipt.paidAt": 1}},
        {"$group": {
            "_id": "$_id",
            "firstPaidAt": {"$first": "$receipt.paidAt"}
        }},
        {"$addFields": {
            "firstPaidWeek": {"$isoWeek": "$firstPaidAt"},
            "firstPaidYear": {"$isoWeekYear": "$firstPaidAt"}
        }},
        {"$group": {
            "_id": {"year": "$firstPaidYear", "week": "$firstPaidWeek"},
            "new_users": {"$addToSet": "$_id"}
        }},
        {"$project": {
            "_id": 1,
            "total_new_users": {"$size": "$new_users"},
            "new_users": 1
        }},
        {"$sort": {"_id.year": 1, "_id.week": 1}}
    ]

    # Liste des testeurs à exclure
    testers_to_exclude = [
        ObjectId("66df2f59c1271156d5468044"),
        ObjectId("670f97d3f38642c54d678d26"),
        ObjectId("65c65360b03953a598253426"),
        ObjectId("65bcb0e43956788471c88e31")
    ]

    # Pipeline pour récupérer les utilisateurs actifs (hebdomadaire)
    pipeline_active_users_week = [
        {"$unwind": "$receipt"},
        {"$match": {
            "receipt.isPaid": True,
            "receipt.storeId": store_filter,
            "receipt.paidAt": {"$gte": date_start, "$lte": date_end},
            "_id": {"$nin": testers_to_exclude}
        }},
        {"$addFields": {
            "paymentWeek": {"$isoWeek": "$receipt.paidAt"},
            "paymentYear": {"$isoWeekYear": "$receipt.paidAt"}
        }},
        {"$group": {
            "_id": {"year": "$paymentYear", "week": "$paymentWeek"},
            "active_users": {"$addToSet": "$_id"}
        }},
        {"$project": {
            "_id": 1,
            "total_active_users": {"$size": "$active_users"},
            "active_users": 1
        }},
        {"$sort": {"_id.year": 1, "_id.week": 1}}
    ]

    # Exécuter les requêtes MongoDB
    cursor_new_users_week = st.session_state.users_collection.aggregate(pipeline_new_users_week)
    cursor_active_users_week = st.session_state.users_collection.aggregate(pipeline_active_users_week)

    data_new_users_week = list(cursor_new_users_week)
    data_active_users_week = list(cursor_active_users_week)

    # Vérification des données
    if not data_new_users_week or not data_active_users_week:
        st.error("❌ Aucune donnée trouvée ! Vérifiez la structure de votre base MongoDB.")
        st.stop()

    # Transformation en DataFrame
    df_new_users_week = pd.DataFrame(data_new_users_week)
    df_active_users_week = pd.DataFrame(data_active_users_week)

    # Extraction des années et semaines (pour les données hebdomadaires)
    df_new_users_week['year'] = df_new_users_week['_id'].apply(lambda x: x['year'])
    df_new_users_week['week'] = df_new_users_week['_id'].apply(lambda x: x['week'])
    df_active_users_week['year'] = df_active_users_week['_id'].apply(lambda x: x['year'])
    df_active_users_week['week'] = df_active_users_week['_id'].apply(lambda x: x['week'])

    # Générer la colonne "début de semaine" (pour la partie hebdomadaire)
    df_new_users_week['week_start'] = df_new_users_week.apply(lambda x: datetime.fromisocalendar(x['year'], x['week'], 1), axis=1)
    df_active_users_week['week_start'] = df_active_users_week.apply(lambda x: datetime.fromisocalendar(x['year'], x['week'], 1), axis=1)

    # Convertir ObjectId en str pour faciliter la manipulation
    df_new_users_week['new_users'] = df_new_users_week['new_users'].apply(lambda users: set(str(u) for u in users) if users else set())
    df_active_users_week['active_users'] = df_active_users_week['active_users'].apply(lambda users: set(str(u) for u in users) if users else set())

    # Trier et indexer par "week_start"
    df_new_users_week = df_new_users_week.sort_values(by='week_start').set_index('week_start')
    df_active_users_week = df_active_users_week.sort_values(by='week_start').set_index('week_start')

    # Générer la liste complète des semaines (plage hebdomadaire)
    all_weeks_range = pd.date_range(start=date_start, end=datetime.now(), freq='W-MON')

    # Créer un DataFrame pour toutes les semaines avec la colonne total_new_users initialisée à 0
    all_weeks_df = pd.DataFrame(index=all_weeks_range)
    all_weeks_df['total_new_users'] = 0

    # Mettre à jour les valeurs pour les semaines disposant de données
    for idx in df_new_users_week.index:
        if idx in all_weeks_df.index:
            all_weeks_df.loc[idx, 'total_new_users'] = df_new_users_week.loc[idx, 'total_new_users']

# ====================================
# Calcul de la rétention hebdomadaire
# ====================================

# Dictionnaire pour stocker la rétention par cohorte
    week_retention = {}

    # Pour chaque date de la plage complète, initialiser la colonne "+0" à 0
    for idx in all_weeks_range:
        week_retention[idx] = {"+0": 0}

    # Pour chaque cohorte (déterminée dans df_new_users_week), calculer la rétention
    for index, row in df_new_users_week.iterrows():
        new_user_set = row['new_users']
        week_retention[index]["+0"] = len(new_user_set)
        
        # S'il n'y a pas de nouveaux utilisateurs, on passe à la cohorte suivante
        if not new_user_set:
            continue

        # Sélectionner les semaines actives postérieures à la date de cohorte
        future_weeks = df_active_users_week.loc[df_active_users_week.index > index]
        for future_index, future_row in future_weeks.iterrows():
            # Calcul du décalage réel en semaines
            week_diff = (future_index - index).days // 7
            future_users = future_row['active_users']
            retained_users = len(new_user_set.intersection(future_users)) if isinstance(future_users, set) else 0
            week_retention[index][f"+{week_diff}"] = retained_users

    # Conversion du dictionnaire en DataFrame
    df_retention_week = pd.DataFrame.from_dict(week_retention, orient='index')


    # =====================================================
    # Forcer la présence des colonnes "+N" jusqu'à un horizon global
    # =====================================================

    # Définir la date actuelle et le début de la semaine courante
    current_date = datetime.now()
    current_week_start = datetime.fromisocalendar(current_date.year, current_date.isocalendar()[1], 1)

    # Calculer l'horizon global : pour chaque cohorte, le nombre maximal de semaines possibles
    global_max = 0
    for index in df_retention_week.index:
        # Calcul du nombre de semaines écoulées entre la cohorte et la semaine courante
        possible = (current_week_start - index).days // 7
        if possible > global_max:
            global_max = possible

    # Pour chaque cohorte, s'assurer que les colonnes de "+0" jusqu'à "+global_max" existent
    # et appliquer la logique suivante :
    # - Si la date (cohorte + N semaines) est dans le futur ( > current_week_start), la valeur doit rester None (NaN)
    # - Sinon, si aucune valeur n'existe, on remplit avec 0.
    for index, row in df_retention_week.iterrows():
        for week_diff in range(global_max + 1):
            col_name = f"+{week_diff}"
            future_week = index + pd.Timedelta(weeks=week_diff)
            if future_week > current_week_start:
                df_retention_week.at[index, col_name] = None
            else:
                # S'il n'existe pas encore ou si la valeur est NaN, on met 0
                if col_name not in df_retention_week.columns or pd.isna(row.get(col_name, None)):
                    df_retention_week.at[index, col_name] = 0


    # ========================================
    # Fusion avec le DataFrame de toutes les semaines
    # ========================================

    df_numeric_week = all_weeks_df.merge(df_retention_week, left_index=True, right_index=True, how='left')


    # ======================================================
    # Réordonner les colonnes de rétention dans l'ordre croissant
    # ======================================================

    retention_cols = sorted(
        [col for col in df_numeric_week.columns if col.startswith("+")],
        key=lambda x: int(x.replace("+", ""))
    )
    other_cols = [col for col in df_numeric_week.columns if not col.startswith("+")]
    ordered_cols = other_cols + retention_cols
    df_numeric_week = df_numeric_week[ordered_cols]



    # Création d'une copie pour les pourcentages (hebdomadaire)
    df_percentage_week = df_numeric_week.copy()
    for col in df_percentage_week.columns:
        if col.startswith("+") and col != "+0":
            mask = (df_percentage_week["+0"] > 0) & (df_percentage_week[col].notna())
            df_percentage_week.loc[mask, col] = (df_percentage_week.loc[mask, col] / df_percentage_week.loc[mask, "+0"] * 100).round(1)
    df_percentage_week["+0"] = df_percentage_week["+0"].apply(lambda x: 100 if x > 0 else 0)

    # Fonction de dégradé pour le style (hebdomadaire)
    def apply_red_gradient_with_future(val):
        if pd.isna(val):
            return 'background-color: #f0f0f0; color: #f0f0f0;'
        elif pd.notna(val):
            intensity = int(255 * ((1 - val / 100) ** 3))
            return f'background-color: rgba(255, {intensity}, {intensity}, 1); color: black;'
        return ''

    # Afficher les tableaux avec Streamlit (pour les données hebdomadaires)
    st.title("Partie Rétention - Analyse Hebdomadaire")
    st.header("📅 Tableau des cohortes hebdomadaires (valeurs numériques)")
    st.dataframe(df_numeric_week)
    st.subheader("📊 Tableau des cohortes hebdomadaires (%)")
    st.dataframe(df_percentage_week.style.applymap(apply_red_gradient_with_future, subset=[col for col in df_percentage_week.columns if col.startswith("+")]))

   # Calcul de la diagonale pour la semaine cible
    # On définit la semaine cible comme étant la dernière semaine présente dans df_active_users_week
    last_week_target = df_active_users_week.index.max()

    diagonal_values = []
    for cohort_date, row in df_numeric_week.iterrows():
        # On ne considère que les cohortes antérieures ou égales à la semaine cible
        if cohort_date > last_week_target:
            continue

        # Calculer l'offset en semaines entre la cohorte et la semaine cible
        offset = (last_week_target - cohort_date).days // 7
        col_name = f"+{offset}"
        
        # Si la colonne existe et contient une valeur (pas NaN)
        if col_name in row.index and pd.notna(row[col_name]):
            diagonal_values.append(row[col_name])
        else:
            st.write(f"⚠️ La colonne {col_name} est absente ou NaN pour la cohorte {cohort_date}")

    diagonal_sum = sum(diagonal_values)

    # Calcul du nombre d'utilisateurs uniques dans la semaine cible
    if last_week_target in df_active_users_week.index:
        unique_users_last_week = len(df_active_users_week.loc[last_week_target, 'active_users'])
    else:
        unique_users_last_week = 0

    st.write(f"Somme de la diagonale (hebdomadaire): {diagonal_sum}")
    st.write(f"Utilisateurs uniques de la dernière semaine (hebdomadaire): {unique_users_last_week}")


    st.header("📈 Évolution du pourcentage d'utilisateurs par cohorte (hebdomadaire)")

    # Préparer les données pour le graphique : on utilise df_percentage_week et on retire éventuellement la colonne 'total_new_users'
    df_pct = df_percentage_week.copy()
    if 'total_new_users' in df_pct.columns:
        df_pct = df_pct.drop(columns=['total_new_users'])

    # Créer la figure Plotly
    fig = go.Figure()

    # Créer une palette de couleurs distinctes pour chaque cohorte
    colormap = cm.get_cmap('tab20c', len(df_pct))

    # Tracer chaque cohorte en ligne
    for i, (cohorte, row) in enumerate(df_pct.iterrows()):
        # Garder uniquement les colonnes non-NaN
        valid_values = row.dropna()
        if valid_values.empty:
            continue
        # Générer une couleur pour cette cohorte
        rgba = colormap(i / len(df_pct))
        color = f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]})"
        
        fig.add_trace(go.Scatter(
            x=list(valid_values.index),
            y=list(valid_values.values),
            mode="lines",
            name=f"Cohorte {cohorte.strftime('%Y-%m-%d')}",
            line=dict(color=color, width=2)
        ))

    # Ajouter une courbe moyenne sur toutes les cohortes
    average_curve = df_pct.mean(axis=0, skipna=True)
    fig.add_trace(go.Scatter(
        x=list(average_curve.index),
        y=list(average_curve.values),
        mode="lines",
        name="Moyenne",
        line=dict(color="black", width=3)
    ))

    # Mise en forme du graphique
    fig.update_layout(
        title="Évolution du pourcentage d'utilisateurs par cohorte (hebdomadaire)",
        xaxis_title="Semaine après le premier achat (ex: +0, +1, +2, ...)",
        yaxis_title="Pourcentage de rétention (%)",
        template="plotly_white",
        xaxis=dict(
            tickmode='array',
            tickvals=[f"+{i}" for i in range(len(average_curve))]
        )
    )

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)

    # ========================
    # 📅 Cohortes par MOIS (nouveaux utilisateurs ayant payé ce mois-là)
    # ========================
    # 📌 Pipeline pour récupérer les nouveaux utilisateurs par mois
    pipeline_new_users = [
        {"$unwind": "$receipt"},
        {"$match": {
            "receipt.isPaid": True,
            "receipt.storeId": store_filter,
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
            "receipt.storeId": store_filter
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
    st.title("Partie Rétention - Analyse Mensuelle")
    st.header("📅 Tableau des cohortes mensuelles")
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
    st.header("📈 Évolution du pourcentage d'utilisateurs par cohorte (hebdomadaire)")
    # ✅ Afficher le graphique dans Streamlit
    st.plotly_chart(fig)

    # ========================
    # Layer Cake Chart
    # ========================

    st.title("🍰 Partie Layer Cake")
    df_layer_cake = df_numeric_week.copy().fillna(0)

    # 🔥 Supprimer la colonne "total_new_users" si elle existe
    if "total_new_users" in df_layer_cake.columns:
        df_layer_cake = df_layer_cake.drop(columns=["total_new_users"])

    # 🔥 Trier les cohortes par ordre de date (ancienne -> récente)
    df_layer_cake = df_layer_cake.sort_index(ascending=True)

    # 🔥 Initialiser la figure
    fig = go.Figure()

    # 🔥 Générer une palette de couleurs cohérente avec `tab20c`
    colormap = cm.get_cmap('tab20c', len(df_layer_cake))

    # 🔥 Parcourir chaque cohorte en ajustant son point de départ
    stacked_values = {col: 0 for col in df_layer_cake.columns}

    for i, (index, row) in enumerate(df_layer_cake.iterrows()):
        # 🔹 Décaler les semaines pour bien empiler les cohortes
        adjusted_x = list(row.index[i:])  # Commence à +i pour décaler la cohorte
        adjusted_y = [stacked_values[col] + row[col] for col in adjusted_x]  # Ajout cumulatif

        # 🔹 Générer une couleur issue de tab20c
        rgba = colormap(i / len(df_layer_cake))
        color = f"rgba({int(rgba[0] * 255)},{int(rgba[1] * 255)},{int(rgba[2] * 255)},{rgba[3]})"

        # 🔹 Ajouter la trace empilée avec la couleur uniforme
        fig.add_trace(go.Scatter(
            x=adjusted_x,
            y=adjusted_y,
            mode='lines',
            fill='tonexty',
            name=f"Cohorte {index.strftime('%Y-%m-%d')}",
            line=dict(color=color),
        ))

        # 🔹 Mettre à jour les valeurs empilées pour la cohorte suivante
        for col in adjusted_x:
            stacked_values[col] += row[col]

    # 🔥 Mise en forme du graphique
    fig.update_layout(
        title="📊 Layer Cake Chart - Rétention des utilisateurs",
        xaxis_title="Semaines après premier achat",
        yaxis_title="Nombre d'utilisateurs cumulés",
        template="plotly_white",
        xaxis=dict(
            tickmode='array',
            tickvals=[f"+{i}" for i in range(len(df_layer_cake.columns))]  # ✅ Affichage des semaines en +x
        ),
    )

    # 🔥 Afficher le Layer Cake Chart
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

if page == "Active Users": 
    # 📌 Pipeline pour récupérer le nombre total d'utilisateurs uniques par semaine
    pipeline_unique_users_per_week = [
        {"$unwind": "$receipt"},
        {"$match": {
            "receipt.isPaid": True,
            "receipt.paidAt": {"$gte": date_start, "$lte": date_end},
            "receipt.storeId": store_filter
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
    st.title("Partie Weekly Active Users")
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

    # ========================
    # Monthly Active Users
    # ========================
        # 📌 Pipeline pour récupérer le nombre total d'utilisateurs uniques par mois
    pipeline_unique_users_per_month = [
        {"$unwind": "$receipt"},
        {"$match": {
            "receipt.isPaid": True,
            "receipt.paidAt": {"$gte": date_start, "$lte": date_end},
            "receipt.storeId": store_filter
        }},
        {"$group": {
            "_id": {
                "year": {"$year": "$receipt.paidAt"},
                "month": {"$month": "$receipt.paidAt"}
            },
            "unique_users": {"$addToSet": "$_id"}
        }},
        {"$project": {
            "_id": 1,
            "total_unique_users": {"$size": "$unique_users"}
        }},
        {"$sort": {"_id.year": 1, "_id.month": 1}}
    ]

    # 📌 Exécuter la requête MongoDB
    cursor_unique_users_per_month = users_collection.aggregate(pipeline_unique_users_per_month)
    data_unique_users_per_month = list(cursor_unique_users_per_month)

    # 📌 Vérification des données
    if not data_unique_users_per_month:
        st.error("❌ Aucune donnée trouvée pour les utilisateurs uniques par mois !")
        st.stop()

    # 📌 Transformation en DataFrame
    df_unique_users_per_month = pd.DataFrame(data_unique_users_per_month)

    # 📌 Extraction des années et mois
    df_unique_users_per_month['year'] = df_unique_users_per_month['_id'].apply(lambda x: x['year'])
    df_unique_users_per_month['month'] = df_unique_users_per_month['_id'].apply(lambda x: x['month'])

    # 📌 Générer la colonne du début de mois
    df_unique_users_per_month['month_start'] = df_unique_users_per_month.apply(
        lambda x: datetime(x['year'], x['month'], 1), axis=1
    )

    # 📌 Trier et indexer les données
    df_unique_users_per_month = df_unique_users_per_month.sort_values(by='month_start').set_index('month_start')

    # 📌 Générer la liste complète des mois entre la première et la dernière
    all_months = pd.date_range(start=df_unique_users_per_month.index.min(), 
                               end=df_unique_users_per_month.index.max(), 
                               freq='MS')  # 'MS' = Month Start

    # 📌 Créer un DataFrame pour tous les mois avec total_unique_users initialisé à 0
    df_all_months_unique_users = pd.DataFrame(index=all_months)
    df_all_months_unique_users['total_unique_users'] = 0

    # 📌 Mettre à jour les valeurs pour les mois qui ont des données
    for idx in df_unique_users_per_month.index:
        if idx in df_all_months_unique_users.index:
            df_all_months_unique_users.loc[idx, 'total_unique_users'] = df_unique_users_per_month.loc[idx, 'total_unique_users']

    # 📌 Afficher le tableau des utilisateurs uniques par mois
    st.title("Partie Monthly Active Users")
    st.subheader("📊 Tableau utilisateurs uniques par mois")
    st.dataframe(df_all_months_unique_users)

    # 📌 Créer une courbe interactive avec Plotly
    fig = px.line(df_all_months_unique_users, 
                x=df_all_months_unique_users.index, 
                y="total_unique_users", 
                title="📈 Évolution des utilisateurs uniques par mois",
                labels={"month_start": "Mois", "total_unique_users": "Utilisateurs uniques"},
                markers=True)
    st.subheader("📈 Évolution des utilisateurs uniques par mois")
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