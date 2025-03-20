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

payment_mapping = {
    "Tous": "Tous",
    "apple-pay": "Apple Pay",
    "apple_pay": "Apple Pay",
    "applepay": "Apple Pay",
    "cb": "CB",
    "conecs": "Conecs",
    "edenred": "Edenred",
    "googlepay": "Google Pay",
    "https://google.com/pay": "Google Pay"
}

# 📌 Connexion MongoDB
client = MongoClient(MONGO_URI)
db = client['storesDatabase']
users_collection = db['usertests']
orders_collection = db['ordertests']

# 📌 Ajout du menu de navigation dans la barre latérale
st.sidebar.title("📊 Dashboard de suivi")

# Sélection du type d'utilisateur
user_type = st.sidebar.radio(
    "Type d'utilisateur :", 
    ["Tous", "Utilisateurs Connectés", "Invités"]
)


# Sélection du magasin (obligatoire)
displayed_store_id = st.sidebar.selectbox(
    "Sélectionnez un magasin :",
    options=list(store_mapping.keys()),
    format_func=lambda x: store_mapping[x]["name"]
)

date_end = datetime.now()
date_start = store_mapping[displayed_store_id]["launch_date"]
store_filter = ObjectId(displayed_store_id)

unique_payments = st.session_state.users_collection.distinct(
    "receipt.paymentMethod",
    {
        "receipt.storeId": store_filter,
        "receipt.isPaid": True,
        "receipt.paidAt": {"$gte": date_start, "$lte": date_end}
    }
)

normalized_payments = {payment_mapping.get(p, p) for p in unique_payments}
normalized_payments.add("Tous")
payment_options = ["Tous"] + sorted([p for p in normalized_payments if p != "Tous"])

selected_payment_method = st.sidebar.selectbox(
    "Sélectionnez un mode de paiement :",
    options=payment_options,
    index=0  
)

# Préparation du filtre de base pour les pipelines
base_filter = {
    "receipt.isPaid": True,
    "receipt.storeId": store_filter,
    "receipt.paidAt": {"$gte": date_start, "$lte": date_end}
}


# Dictionnaire pour associer la méthode normalisée à toutes ses variantes réelles possibles
payment_variants = {
    "Apple Pay": ["apple-pay", "apple_pay", "applepay"],
    "Google Pay": ["googlepay", "https://google.com/pay"],
    "CB": ["cb", "CB"],
    "Conecs": ["conecs", "Conecs"],
    "Edenred": ["edenred", "Edenred"]
}
payment_filter = {}
if selected_payment_method != "Tous":
    if selected_payment_method in payment_variants:
        variants = payment_variants[selected_payment_method]
    else:
        variants = [selected_payment_method]
    payment_filter = {"receipt.paymentMethod": {"$in": variants}}

match_filter = {**base_filter, **payment_filter}

toggle_view = st.sidebar.radio(
    "Sélectionnez la vue :", 
    ["Hebdomadaire", "Mensuel"], 
    index=0  
    )

page = st.sidebar.radio(
    "Choisissez une section :", 
    ["Rétention", "Acquisition", "Active Users", "Bug Report"]
)

# ========================
# Partie Rétention
# ========================
if page == "Rétention" and user_type == "Utilisateurs Connectés" and toggle_view == "Hebdomadaire":

            if selected_payment_method == "Tous":
                pipeline_new_users_week = [
                    {"$unwind": "$receipt"},
                    {"$match": match_filter},  # Ici, match_filter équivaut à base_filter (pas de filtre sur paymentMethod)
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


            else:
                pipeline_new_users_week = [
                    # Stage 1: Sélectionner les utilisateurs qui ont utilisé le mode de paiement choisi
                    {"$unwind": "$receipt"},
                    {"$match": match_filter},
                    {"$group": {"_id": "$_id"}},
                                
                    # Stage 2: Pour ces utilisateurs, trouver leur premier paiement correspondant au mode choisi
                    {"$lookup": {
                        "from": "usertests",
                        "localField": "_id",
                        "foreignField": "_id",
                        "as": "user_data"
                    }},
                    {"$unwind": "$user_data"},
                    {"$unwind": "$user_data.receipt"},
                    {"$match": {
                        "user_data.receipt.isPaid": True,
                        "user_data.receipt.storeId": store_filter,
                        "user_data.receipt.paymentMethod": {"$in": variants}  # Filtrer sur les variantes
                    }},
                    {"$sort": {"user_data.receipt.paidAt": 1}},
                    {"$group": {
                        "_id": "$_id",
                        "firstPaidAt": {"$first": "$user_data.receipt.paidAt"}
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

            # Use the same approach for active users
            if selected_payment_method == "Tous":
                pipeline_active_users_week = [
                    {"$unwind": "$receipt"},
                    {"$match": {**base_filter, "_id": {"$nin": testers_to_exclude}}},
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
            else:
                pipeline_active_users_week = [
                    {"$unwind": "$receipt"},
                    {"$match": {**match_filter, "_id": {"$nin": testers_to_exclude}}},
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

            # Définir la date actuelle et le début de la semaine courante
            current_date = datetime.now()
            current_week_start = datetime.fromisocalendar(current_date.year, current_date.isocalendar()[1], 1)

            # Générer la liste complète des semaines (plage hebdomadaire), en excluant la semaine en cours
            all_weeks_range = pd.date_range(start=date_start, end=current_week_start - pd.Timedelta(days=1), freq='W-MON')

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

            if df_new_users_week.empty:
                st.warning("Aucun nouvel utilisateur trouvé pour ce mode de paiement.")
                df_numeric_week = pd.DataFrame(columns=['total_new_users', '+0'])
                df_percentage_week = pd.DataFrame(columns=['total_new_users', '+0'])
            else:
                # ====================================
                # Calcul de la rétention hebdomadaire
                # ====================================
                
                # Dictionnaire pour stocker la rétention par cohorte
                week_retention = {}
                
                # Combiner les indices de all_weeks_range et df_new_users_week pour éviter les KeyError
                # Exclure tous les index supérieurs ou égaux à la semaine en cours
                filtered_df_new_users_week = df_new_users_week[df_new_users_week.index < current_week_start]
                all_cohort_dates = set(all_weeks_range) | set(filtered_df_new_users_week.index)
                
                # Pour chaque date de la plage complète, initialiser la colonne "+0" à 0
                for idx in all_cohort_dates:
                    week_retention[idx] = {"+0": 0}
                
                # Pour chaque cohorte (déterminée dans df_new_users_week), calculer la rétention
                for index, row in filtered_df_new_users_week.iterrows():
                    new_user_set = row['new_users']
                    week_retention[index]["+0"] = len(new_user_set)
                    
                    # S'il n'y a pas de nouveaux utilisateurs, on passe à la cohorte suivante
                    if not new_user_set:
                        continue
                
                    # Sélectionner les semaines actives postérieures à la date de cohorte, mais avant la semaine en cours
                    if not df_active_users_week.empty:
                        future_weeks = df_active_users_week.loc[(df_active_users_week.index > index) & 
                                                            (df_active_users_week.index < current_week_start)]
                        for future_index, future_row in future_weeks.iterrows():
                            # Calcul du décalage réel en semaines
                            week_diff = (future_index - index).days // 7
                            future_users = future_row['active_users']
                            retained_users = len(new_user_set.intersection(future_users)) if isinstance(future_users, set) else 0
                            week_retention[index][f"+{week_diff}"] = retained_users
                
                # Conversion du dictionnaire en DataFrame
                df_retention_week = pd.DataFrame.from_dict(week_retention, orient='index')
                
                # Calculer l'horizon global : pour chaque cohorte, le nombre maximal de semaines possibles
                # Exclure la semaine en cours
                global_max = 0
                for index in df_retention_week.index:
                    # Calcul du nombre de semaines écoulées entre la cohorte et la semaine précédente
                    possible = (current_week_start - pd.Timedelta(days=7) - index).days // 7
                    if possible > global_max:
                        global_max = possible
                
                # Pour chaque cohorte, s'assurer que les colonnes de "+0" jusqu'à "+global_max" existent
                # et appliquer la logique suivante :
                # - Si la date (cohorte + N semaines) est dans le futur ou dans la semaine en cours, la valeur doit rester None (NaN)
                # - Sinon, si aucune valeur n'existe, on remplit avec 0.
                for index, row in df_retention_week.iterrows():
                    for week_diff in range(global_max + 1):
                        col_name = f"+{week_diff}"
                        future_week = index + pd.Timedelta(weeks=week_diff)
                        if future_week >= current_week_start:
                            df_retention_week.at[index, col_name] = None
                        else:
                            # S'il n'existe pas encore ou si la valeur est NaN, on met 0
                            if col_name not in df_retention_week.columns or pd.isna(row.get(col_name, None)):
                                df_retention_week.at[index, col_name] = 0
                
                # Créer un DataFrame pour toutes les semaines avec la colonne total_new_users initialisée à 0
                all_weeks_df = pd.DataFrame(index=sorted(all_cohort_dates))
                all_weeks_df['total_new_users'] = 0
                
                # Mettre à jour les valeurs pour les semaines disposant de données
                for idx in filtered_df_new_users_week.index:
                    if idx in all_weeks_df.index:
                        all_weeks_df.loc[idx, 'total_new_users'] = filtered_df_new_users_week.loc[idx, 'total_new_users']
                
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
            today = pd.Timestamp.today()
            # Calcul du début de la semaine (lundi) et normalisation à minuit
            target_week = (today - pd.Timedelta(weeks=1)).normalize()  # ✅ Prend la semaine précédente

            # Si l'index de df_active_users_week est un PeriodIndex, convertir target_week en période
            if isinstance(df_active_users_week.index, pd.PeriodIndex):
                target_week = pd.Period(target_week, freq='W-MON')

            diagonal_values = []
            for cohort_date, row in df_numeric_week.iterrows():
                # On ne considère que les cohortes antérieures ou égales à la semaine cible
                if cohort_date > target_week:
                    continue

                # Calcul de l'offset en semaines entre la cohorte et la semaine cible
                offset = (target_week - cohort_date).days // 7
                col_name = f"+{offset}"
                
                # Si la colonne existe et contient une valeur (pas NaN)
                if col_name in row.index and pd.notna(row[col_name]):
                    diagonal_values.append(row[col_name])
                else:
                    st.write(f"⚠️ La colonne {col_name} est absente ou NaN pour la cohorte {cohort_date}")

            diagonal_sum = sum(diagonal_values)

            # Récupération du nombre d'utilisateurs actifs uniques dans la semaine cible
            if target_week in df_active_users_week.index:
                active_users_data = df_active_users_week.loc[target_week, 'active_users']
                # S'il s'agit d'une liste ou d'un ensemble, on compte le nombre d'éléments
                if isinstance(active_users_data, (list, tuple, set)):
                    unique_users_target = len(active_users_data)
                else:
                    # Sinon, on suppose que la valeur est déjà un nombre
                    unique_users_target = active_users_data
            else:
                unique_users_target = 0

            st.write(f"Somme de la diagonale (hebdomadaire): {diagonal_sum}")

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
            # Layer Cake Chart Hebdomadaire
            # ========================
            st.title("🍰 Partie Layer Cake")

            # Vérifier que df_numeric_week existe et n'est pas vide
            if 'df_numeric_week' not in locals() or df_numeric_week.empty:
                st.warning("❌ Aucune donnée de rétention hebdomadaire disponible pour générer le Layer Cake.")
                st.stop()

            # Copier df_numeric_week et remplacer NaN par 0
            df_layer_cake = df_numeric_week.copy().fillna(0)

            # Supprimer la colonne "total_new_users" si elle existe
            if "total_new_users" in df_layer_cake.columns:
                df_layer_cake = df_layer_cake.drop(columns=["total_new_users"])

            # Conserver uniquement les colonnes de rétention (celles qui commencent par "+")
            retention_cols = [col for col in df_layer_cake.columns if col.startswith("+")]
            # Assurer un ordre croissant, par exemple "+0", "+1", "+2", ...
            retention_cols = sorted(retention_cols, key=lambda c: int(c.replace("+", "")))
            df_layer_cake = df_layer_cake[retention_cols]

            # On suppose que l'index (les cohortes) est déjà chronologique (du plus ancien au plus récent)
            df_layer_cake.sort_index(ascending=True, inplace=True)

            num_weeks = len(retention_cols)
            x_axis = np.arange(num_weeks)  # Axe x : [0, 1, 2, ..., num_weeks-1]

            # Initialiser la figure Plotly et la palette de couleurs
            fig = go.Figure()
            num_cohorts = len(df_layer_cake)
            colormap = cm.get_cmap('tab20c', num_cohorts)

            # Parcourir chaque cohorte (du plus ancien au plus récent)
            for i, (cohort_date, row) in enumerate(df_layer_cake.iterrows()):
                # Extraire les valeurs de rétention de la cohorte
                cohort_values = np.array(row.tolist(), dtype=float)
                
                # Créer une série décalée : i positions initiales seront None pour ne pas afficher de valeurs avant le début
                shifted = [None] * i + list(cohort_values[:num_weeks - i])
                
                rgba = colormap(i / num_cohorts)
                color = f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]})"
                
                fig.add_trace(go.Scatter(
                    x=x_axis,
                    y=shifted,
                    mode='lines',
                    stackgroup='one',  # Empilement automatique avec Plotly
                    name=f"Cohorte {cohort_date.strftime('%Y-%m-%d')}",
                    line=dict(color=color)
                ))

            # Configurer l'axe x pour afficher "+0", "+1", ... etc.
            fig.update_xaxes(
                tickmode='array',
                tickvals=list(x_axis),
                ticktext=[f"+{i}" for i in x_axis]
            )

            # Mettre à jour la mise en page pour isoler la trace cliquée dans la légende
            fig.update_layout(
                title="📊 Layer Cake Chart - Rétention des utilisateurs",
                xaxis_title="Semaines après premier achat",
                yaxis_title="Nombre d'utilisateurs cumulés",
                template="plotly_white",
                legend_title="Cohortes hebdomadaires",
            )

            st.plotly_chart(fig)
            # -------------------------------
            # Section : Export de cohorte
            # -------------------------------
            st.subheader("Exporter une cohorte")

            # Vérifier que df_new_users_week existe et n'est pas vide
            if 'df_new_users_week' not in locals() or df_new_users_week.empty:
                st.warning("❌ Aucune cohorte disponible pour l'export.")
            else:
                # Préparer les options de cohorte (utiliser l'index qui est le week_start)
                cohort_options = df_new_users_week.index.strftime("%Y-%m-%d").tolist()
                selected_cohorts = st.multiselect("Sélectionnez la/les cohorte(s) à exporter", options=cohort_options)
                
                if selected_cohorts:
                    # Convertir les chaînes sélectionnées en datetime
                    selected_cohort_dates = [pd.to_datetime(date_str) for date_str in selected_cohorts]
                    
                    # Récupérer les _id des utilisateurs pour chaque cohorte sélectionnée
                    user_ids = set()
                    for cohort_date in selected_cohort_dates:
                        if cohort_date in df_new_users_week.index:
                            user_ids.update(df_new_users_week.loc[cohort_date, "new_users"])

                    user_ids_converted = [ObjectId(uid) for uid in user_ids]
                    
                    # Définir la projection pour récupérer les coordonnées et le tableau receipt
                    projection = {
                        "nom": 1,
                        "prenom": 1,
                        "email": 1,
                        "phoneNumber": 1,
                        "gender": 1,
                        "address": 1,
                        "birthDate": 1,
                        "city": 1,
                        "zipCode": 1,
                        "receipt": 1
                    }
                    
                    # Récupérer les détails des utilisateurs depuis MongoDB
                    user_details = list(st.session_state.users_collection.find({"_id": {"$in": user_ids_converted}}, projection))
                    
                    # Pour chaque utilisateur, déterminer le dernier paiement (trier les reçus par "paidAt")
                    for user in user_details:
                        receipts = user.get("receipt", [])
                        if receipts:
                            receipts_sorted = sorted(receipts, key=lambda r: r.get("paidAt", ""), reverse=True)
                            user["last_receipt"] = receipts_sorted[0]
                        else:
                            user["last_receipt"] = None
                    
                    # Préparer les données pour l'export
                    export_rows = []
                    for user in user_details:
                        row = {
                            "_id": str(user.get("_id")),
                            "nom": user.get("nom", ""),
                            "prenom": user.get("prenom", ""),
                            "email": user.get("email", ""),
                            "phoneNumber": user.get("phoneNumber", ""),
                            "gender": user.get("gender", ""),
                            "address": user.get("address", ""),
                            "birthDate": user.get("birthDate", ""),
                            "city": user.get("city", ""),
                            "zipCode": user.get("zipCode", "")
                        }
                        last_receipt = user.get("last_receipt")
                        if last_receipt:
                            row["lastPaidAt"] = last_receipt.get("paidAt", "")
                            row["lastPaymentMethod"] = last_receipt.get("paymentMethod", "")
                            # Vous pouvez ajouter d'autres champs de receipt si besoin
                        else:
                            row["lastPaidAt"] = ""
                            row["lastPaymentMethod"] = ""
                        export_rows.append(row)
                    
                    df_export = pd.DataFrame(export_rows)
                    
                    # Convertir le DataFrame en CSV
                    csv_data = df_export.to_csv(index=False).encode('utf-8')
                    
                    st.download_button(
                        label="Télécharger la cohorte en CSV",
                        data=csv_data,
                        file_name="cohorte_export.csv",
                        mime="text/csv"
                    )




elif page == "Rétention" and toggle_view == "Mensuel" and user_type == "Utilisateurs Connectés":
    # ========================
    # 📅 Cohortes par MOIS (nouveaux utilisateurs ayant payé ce mois-là)
    # ========================
    # Définir le filtre de base pour tous les paiements dans la période et pour le magasin
    # ----------------------------
    # 1) Construction des pipelines
    # ---------------------------

    # Pipeline pour récupérer les nouveaux utilisateurs par mois
    if selected_payment_method == "Tous":
        pipeline_new_users = [
            {"$unwind": "$receipt"},
            {"$match": base_filter},
            {"$sort": {"receipt.paidAt": 1}},
            {"$group": {
                "_id": "$_id",
                "firstPaidAt": {"$first": "$receipt.paidAt"}
            }},
            {"$group": {
                "_id": {
                    "year": {"$year": "$firstPaidAt"},
                    "month": {"$month": "$firstPaidAt"}
                },
                "new_users": {"$addToSet": "$_id"}
            }},
            {"$project": {
                "_id": 1,
                "total_new_users": {"$size": "$new_users"},
                "new_users": 1
            }},
            {"$sort": {"_id.year": 1, "_id.month": 1}}
        ]
    else:
        variants = payment_variants.get(selected_payment_method, [selected_payment_method])
        pipeline_new_users = [
            # Étape 1 : Sélectionner les utilisateurs ayant utilisé le mode choisi
            {"$unwind": "$receipt"},
            {"$match": {
                "receipt.isPaid": True,
                "receipt.storeId": store_filter,
                "receipt.paymentMethod": {"$in": variants}
            }},
            {"$group": {"_id": "$_id"}},

            # Étape 2 : Trouver le premier paiement (toutes méthodes confondues)
            {"$lookup": {
                "from": "usertests",
                "localField": "_id",
                "foreignField": "_id",
                "as": "user_data"
            }},
            {"$unwind": "$user_data"},
            {"$unwind": "$user_data.receipt"},
            {"$match": {
                "user_data.receipt.isPaid": True,
                "user_data.receipt.storeId": store_filter
            }},
            {"$sort": {"user_data.receipt.paidAt": 1}},
            {"$group": {
                "_id": "$_id",
                "firstPaidAt": {"$first": "$user_data.receipt.paidAt"}
            }},
            {"$match": {
                "firstPaidAt": {"$gte": date_start, "$lte": date_end}
            }},
            {"$group": {
                "_id": {
                    "year": {"$year": "$firstPaidAt"},
                    "month": {"$month": "$firstPaidAt"}
                },
                "new_users": {"$addToSet": "$_id"}
            }},
            {"$project": {
                "_id": 1,
                "total_new_users": {"$size": "$new_users"},
                "new_users": 1
            }},
            {"$sort": {"_id.year": 1, "_id.month": 1}}
        ]
    testers_to_exclude = [
        ObjectId("66df2f59c1271156d5468044"),
        ObjectId("670f97d3f38642c54d678d26"),
        ObjectId("65c65360b03953a598253426"),
        ObjectId("65bcb0e43956788471c88e31")
    ]

    # Pipeline pour récupérer les utilisateurs actifs par mois
    if selected_payment_method == "Tous":
        pipeline_active_users = [
            {"$unwind": "$receipt"},
            {"$match": {**base_filter, "_id": {"$nin": testers_to_exclude}}},
            {"$addFields": {
                "paymentYear": {"$year": "$receipt.paidAt"},
                "paymentMonth": {"$month": "$receipt.paidAt"}
            }},
            {"$group": {
                "_id": {"year": "$paymentYear", "month": "$paymentMonth"},
                "active_users": {"$addToSet": "$_id"}
            }},
            {"$project": {
                "_id": 1,
                "total_active_users": {"$size": "$active_users"},
                "active_users": 1
            }},
            {"$sort": {"_id.year": 1, "_id.month": 1}}
        ]
    else:
        pipeline_active_users = [
            {"$unwind": "$receipt"},
            {"$match": {
                "receipt.isPaid": True,
                "receipt.storeId": store_filter,
                "receipt.paidAt": {"$gte": date_start, "$lte": date_end},
                "receipt.paymentMethod": {"$in": variants},  # Utiliser la même logique que pour les nouveaux utilisateurs
                "_id": {"$nin": testers_to_exclude}
            }},
            {"$addFields": {
                "paymentYear": {"$year": "$receipt.paidAt"},
                "paymentMonth": {"$month": "$receipt.paidAt"}
            }},
            {"$group": {
                "_id": {"year": "$paymentYear", "month": "$paymentMonth"},
                "active_users": {"$addToSet": "$_id"}
            }},
            {"$project": {
                "_id": 1,
                "total_active_users": {"$size": "$active_users"},
                "active_users": 1
            }},
            {"$sort": {"_id.year": 1, "_id.month": 1}}
        ]

    # ----------------------------
    # 2) Exécution des pipelines
    # ----------------------------
    cursor_new_users = users_collection.aggregate(pipeline_new_users)
    cursor_active_users = users_collection.aggregate(pipeline_active_users)

    data_new_users = list(cursor_new_users)
    data_active_users = list(cursor_active_users)

    if not data_new_users or not data_active_users:
        st.error("❌ Aucune donnée trouvée ! Vérifiez la structure de votre base MongoDB.")
        st.stop()

    # ----------------------------
    # 3) Construction des DataFrames
    # ----------------------------
    df_new_users = pd.DataFrame(data_new_users)
    df_active_users = pd.DataFrame(data_active_users)

    # Extraire l'année et le mois depuis l'_id
    df_new_users['year'] = df_new_users['_id'].apply(lambda x: x['year'])
    df_new_users['month'] = df_new_users['_id'].apply(lambda x: x['month'])
    df_active_users['year'] = df_active_users['_id'].apply(lambda x: x['year'])
    df_active_users['month'] = df_active_users['_id'].apply(lambda x: x['month'])

    # Créer la colonne "month_start"
    df_new_users['month_start'] = df_new_users.apply(lambda x: datetime(x['year'], x['month'], 1), axis=1)
    df_active_users['month_start'] = df_active_users.apply(lambda x: datetime(x['year'], x['month'], 1), axis=1)

    # Convertir les ensembles d'IDs en chaînes
    df_new_users['new_users'] = df_new_users['new_users'].apply(lambda users: set(str(u) for u in users) if users else set())
    df_active_users['active_users'] = df_active_users['active_users'].apply(lambda users: set(str(u) for u in users) if users else set())

    # Trier et indexer
    df_new_users = df_new_users.sort_values(by='month_start').set_index('month_start')
    df_active_users = df_active_users.sort_values(by='month_start').set_index('month_start')

    today = datetime.now()
    first_day_current_month = datetime(today.year, today.month, 1)
    df_new_users = df_new_users[df_new_users.index < first_day_current_month]
    df_active_users = df_active_users[df_active_users.index < first_day_current_month]


    # ----------------------------
    # 4) Calcul de la rétention mensuelle
    # ----------------------------
    monthly_retention = {}
    for index, row in df_new_users.iterrows():
        new_user_set = row['new_users']
        monthly_retention[index] = {"+0": row["total_new_users"]}

        if not new_user_set:
            continue

        # Pour chaque mois >= la cohorte, on vérifie combien sont toujours actifs
        future_months = df_active_users.loc[df_active_users.index >= index]
        for month_diff, (future_index, future_row) in enumerate(future_months.iterrows()):
            future_users = future_row['active_users']
            retained_users = len(new_user_set.intersection(future_users)) if isinstance(future_users, set) else 0
            monthly_retention[index][f"+{month_diff}"] = retained_users

    # On convertit ce dictionnaire en DataFrame
    df_monthly_retention = pd.DataFrame.from_dict(monthly_retention, orient='index')

    # ✅ Correction : Forcer la colonne "+0" à être identique à "total_new_users"
    df_monthly_retention["+0"] = df_new_users["total_new_users"]

    # Fusionner avec df_new_users pour récupérer total_new_users
    df_final = df_new_users[['total_new_users']].merge(
        df_monthly_retention, left_index=True, right_index=True, how='left'
    )

    # ----------------------------
    # 5) Afficher tous les mois de la plage, même s'ils sont à 0
    # ----------------------------

    # a) Générer la liste complète des mois
    today = datetime.now()
    first_day_current_month = datetime(today.year, today.month, 1)
    last_month_completed = first_day_current_month - pd.DateOffset(months=1)
    # Aligner date_start sur le premier jour du mois
    date_start = datetime(date_start.year, date_start.month, 1)

    all_months_range = pd.date_range(start=date_start, end=last_month_completed, freq='MS')
    all_months_df = pd.DataFrame(index=all_months_range)
    all_months_df['total_new_users'] = 0  # Valeur par défaut

    # b) Fusionner pour inclure tous les mois
    df_merged = all_months_df.merge(df_final, left_index=True, right_index=True, how='left', suffixes=('', '_old'))

    # Récupérer les bonnes valeurs de total_new_users
    if 'total_new_users_old' in df_merged.columns:
        df_merged['total_new_users'] = df_merged['total_new_users_old'].fillna(df_merged['total_new_users'])
        df_merged.drop(columns=['total_new_users_old'], inplace=True)

    df_final = df_merged

    # c) Recalcul complet de global_max
    # Nombre maximum de mois possibles entre la plus ancienne cohorte et aujourd'hui
    oldest_cohort = df_final.index.min() if not df_final.empty else first_day_current_month
    months_diff = (last_month_completed.year - oldest_cohort.year) * 12 + (last_month_completed.month - oldest_cohort.month)
    global_max = months_diff

    # d) Recréer toutes les colonnes +0 à +global_max
    for offset in range(global_max + 1):
        col_name = f"+{offset}"
        if col_name not in df_final.columns:
            df_final[col_name] = None

    # e) Remplir TOUTES les valeurs manquantes avec 0 pour les périodes passées
    for index, row in df_final.iterrows():
        cohort_date = index
        for offset in range(global_max + 1):
            col_name = f"+{offset}"
            future_date = cohort_date + pd.DateOffset(months=offset)
            
            # Si le mois est passé (jusqu'au mois dernier inclus), on doit avoir une valeur
            if future_date <= last_month_completed:
                # Vérifier si la valeur existe déjà et n'est pas nulle
                current_value = df_final.at[index, col_name]
                if pd.isna(current_value):
                    df_final.at[index, col_name] = 0

    # ----------------------------
    # 6) Affichage des cohortes mensuelles (valeurs numériques)
    # ----------------------------
    st.title("Partie Rétention - Analyse Mensuelle")
    st.header("📅 Tableau des cohortes mensuelles")
    st.subheader("📊 Cohortes mensuelles (valeurs numériques)")
    st.dataframe(df_final)

    # ----------------------------
    # 7) Calcul des pourcentages
    # ----------------------------
    df_percentage = df_final.copy()
    df_percentage_calcul = df_final.copy()

    for col in df_percentage.columns:
        if col.startswith("+"):
            if col == "+0":
                df_percentage[col] = df_percentage_calcul["+0"].apply(lambda x: 100 if x > 0 else 0)
            else:
                mask = (df_percentage_calcul["+0"] > 0) & (df_percentage_calcul[col].notna())
                df_percentage.loc[mask, col] = (
                    df_percentage_calcul.loc[mask, col] / df_percentage_calcul.loc[mask, "+0"] * 100
                ).round(1)

    def apply_red_gradient_with_future(val):
        if pd.isna(val):
            return 'background-color: #f0f0f0; color: #f0f0f0;'
        elif pd.notna(val):
            intensity = int(255 * ((1 - val / 100) ** 3))
            return f'background-color: rgba(255, {intensity}, {intensity}, 1); color: black;'
        return ''

    st.subheader("📊 Cohortes mensuelles (%)")
    st.dataframe(
        df_percentage.style.applymap(apply_red_gradient_with_future, subset=[c for c in df_percentage.columns if c.startswith("+")])
    )

    # ----------------------------
    # 8) (Optionnel) Graphique de rétention mensuelle
    # ----------------------------
    # Supprimer "total_new_users" avant le tracé
    df_plot = df_percentage.drop(columns=["total_new_users"]) if "total_new_users" in df_percentage.columns else df_percentage.copy()

    fig = go.Figure()
    colormap = cm.get_cmap('tab20c', len(df_plot))

    for i, (idx, row) in enumerate(df_plot.iterrows()):
        valid_values = row.dropna()
        if "+0" not in valid_values.index:
            continue
        rgba = colormap(i / len(df_plot))
        color = f'rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]})'
        fig.add_trace(go.Scatter(
            x=valid_values.index,
            y=valid_values.values,
            mode='lines',
            name=f"Cohorte {idx.strftime('%Y-%m-%d')}",
            line=dict(width=2, color=color),
            hoverinfo='x+y',
            opacity=0.8
        ))

    # Courbe moyenne
    average_curve = df_plot.mean(axis=0, skipna=True)
    fig.add_trace(go.Scatter(
        x=average_curve.index,
        y=average_curve.values,
        mode='lines',
        name='Moyenne par +x',
        line=dict(width=3, color='black'),
        opacity=1.0
    ))

    fig.update_layout(
        title="📊 Rétention des utilisateurs par mois (%)",
        xaxis_title="Mois après premier achat",
        yaxis_title="Pourcentage de rétention",
        template="plotly_white",
        xaxis=dict(
            tickmode='array',
            tickvals=[f'+{i}' for i in range(len(average_curve))]
        ),
        yaxis=dict(
            tickformat=".1f",
            range=[0, 110]
        )
    )

    st.header("📈 Évolution du pourcentage d'utilisateurs par cohorte (mensuelle)")
    st.plotly_chart(fig)

    st.title("🍰 Partie Layer Cake - Mensuel (Valeurs numériques)")

    # Vérifier que df_final existe et n'est pas vide (ou utilisez df_numeric_month si c'est votre DataFrame source)
    if 'df_final' not in locals() or df_final.empty:
        st.warning("❌ Aucune donnée de rétention mensuelle disponible pour générer le Layer Cake.")
        st.stop()

    # Copier le DataFrame de rétention mensuelle et remplacer les NaN par 0
    df_layer_cake = df_final.copy().fillna(0)

    # Supprimer la colonne "total_new_users" si elle existe
    if "total_new_users" in df_layer_cake.columns:
        df_layer_cake = df_layer_cake.drop(columns=["total_new_users"])

    # Conserver uniquement les colonnes de rétention (celles qui commencent par "+")
    retention_cols = [col for col in df_layer_cake.columns if col.startswith("+")]
    # Assurer un ordre croissant : "+0", "+1", "+2", ...
    retention_cols = sorted(retention_cols, key=lambda c: int(c.replace("+", "")))
    df_layer_cake = df_layer_cake[retention_cols]

    # S'assurer que l'index (les cohortes mensuelles) est chronologique
    df_layer_cake.sort_index(ascending=True, inplace=True)

    # Initialiser la figure Plotly et la palette de couleurs
    fig = go.Figure()
    num_cohorts = len(df_layer_cake)
    colormap = cm.get_cmap('tab20c', num_cohorts)

    # Le maximum de mois disponible est défini par le nombre de colonnes (par exemple, 3 colonnes -> +0 à +2)
    max_retention_month = len(retention_cols) - 1

    # Pour chaque cohorte mensuelle (du plus ancien au plus récent)
    # On aligne les séries de manière que, par exemple, la 2ème cohorte commence à x=1,
    # de sorte que son "+0" s'empile sur le "+1" de la cohorte précédente.
    for i, (cohort_date, row) in enumerate(df_layer_cake.iterrows()):
        # Extraire les valeurs numériques de rétention pour la cohorte
        cohort_values = np.array(row.tolist(), dtype=float)
        
        # Définir les x pour cette cohorte : on commence à l'offset i
        x_values = np.arange(i, i + len(cohort_values))
        # Limiter les x à ne pas dépasser le maximum de mois (max_retention_month)
        mask = x_values <= max_retention_month
        x_values = x_values[mask]
        y_values = cohort_values[mask]
        
        rgba = colormap(i / num_cohorts)
        color = f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]})"
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='lines',
            stackgroup='one',  # Permet l'empilement avec Plotly
            name=f"Cohorte {cohort_date.strftime('%Y-%m-%d')}",
            line=dict(color=color)
        ))

    # Définir l'axe x pour qu'il s'arrête au nombre maximal de mois
    all_x = np.arange(0, max_retention_month + 1)

    fig.update_xaxes(
        tickmode='array',
        tickvals=list(all_x),
        ticktext=[f"+{i}" for i in all_x],
        range=[0, max_retention_month]
    )

    fig.update_layout(
        title="📊 Layer Cake Chart - Rétention mensuelle des utilisateurs (Valeurs numériques)",
        xaxis_title="Mois après premier achat",
        yaxis_title="Nombre d'utilisateurs cumulés",
        template="plotly_white",
        legend_title="Cohortes mensuelles"
    )

    st.plotly_chart(fig)

    # -------------------------------
    # Section : Export de cohorte
    # -------------------------------
    st.subheader("Exporter une cohorte")

    # Vérifier que df_new_users existe et n'est pas vide
    if 'df_new_users' not in locals() or df_new_users.empty:
        st.warning("❌ Aucune cohorte disponible pour l'export.")
    else:
        # Préparer les options de cohorte en utilisant l'index de df_new_users (par exemple, month_start)
        cohort_options = df_new_users.index.strftime("%Y-%m-%d").tolist()
        selected_cohorts = st.multiselect("Sélectionnez la/les cohorte(s) à exporter", options=cohort_options)
        
        if selected_cohorts:
            # Convertir les chaînes sélectionnées en datetime
            selected_cohort_dates = [pd.to_datetime(date_str) for date_str in selected_cohorts]
            
            # Récupérer les _id des utilisateurs pour chaque cohorte sélectionnée depuis df_new_users
            user_ids = set()
            for cohort_date in selected_cohort_dates:
                if cohort_date in df_new_users.index:
                    # La colonne "new_users" contient un ensemble d'_id pour la cohorte
                    user_ids.update(df_new_users.loc[cohort_date, "new_users"])
            
            # IMPORTANT : Convertir les _id (qui sont des chaînes) en ObjectId pour la requête Mongo
            from bson import ObjectId
            user_ids_converted = [ObjectId(uid) for uid in user_ids]
            
            # Définir la projection pour récupérer les coordonnées et le tableau receipt
            projection = {
                "nom": 1,
                "prenom": 1,
                "email": 1,
                "phoneNumber": 1,
                "gender": 1,
                "address": 1,
                "birthDate": 1,
                "city": 1,
                "zipCode": 1,
                "receipt": 1,
                "createdAt": 1  # Ajout de createdAt si souhaité
            }
            
            # Récupérer les détails des utilisateurs depuis MongoDB
            user_details = list(st.session_state.users_collection.find({"_id": {"$in": user_ids_converted}}, projection))
            
            # Pour chaque utilisateur, déterminer le dernier paiement (trié par "paidAt" décroissant)
            for user in user_details:
                receipts = user.get("receipt", [])
                if receipts:
                    receipts_sorted = sorted(receipts, key=lambda r: r.get("paidAt", ""), reverse=True)
                    user["last_receipt"] = receipts_sorted[0]
                else:
                    user["last_receipt"] = None
            
            # Préparer les données pour l'export en aplatissant le dernier paiement
            export_rows = []
            for user in user_details:
                row = {
                    "_id": str(user.get("_id")),
                    "nom": user.get("nom", ""),
                    "prenom": user.get("prenom", ""),
                    "email": user.get("email", ""),
                    "phoneNumber": user.get("phoneNumber", ""),
                    "gender": user.get("gender", ""),
                    "address": user.get("address", ""),
                    "birthDate": user.get("birthDate", ""),
                    "city": user.get("city", ""),
                    "zipCode": user.get("zipCode", ""),
                    "createdAt": user.get("createdAt", "")
                }
                last_receipt = user.get("last_receipt")
                if last_receipt:
                    row["lastPaidAt"] = last_receipt.get("paidAt", "")
                    row["lastPaymentMethod"] = last_receipt.get("paymentMethod", "")
                else:
                    row["lastPaidAt"] = ""
                    row["lastPaymentMethod"] = ""
                export_rows.append(row)
            
            df_export = pd.DataFrame(export_rows)
            
            # Convertir le DataFrame en CSV
            csv_data = df_export.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="Télécharger la cohorte en CSV",
                data=csv_data,
                file_name="cohorte_export.csv",
                mime="text/csv"
            )

elif page == "Rétention" and toggle_view == "Hebdomadaire" and user_type == "Invités":
    st.write("✅ Mode Rétention Hebdomadaire - Invités sélectionné")

    # 📌 Pipeline MongoDB pour récupérer les **nouveaux invités payants** par semaine
    guest_pipeline_new_users_week = [
        # 🏷️ Étape 1 : Filtrer les documents où `userId` commence par "GUEST_" et est payé
        {"$match": {
            "userId": {"$exists": True, "$ne": None, "$regex": "^GUEST_"},  
            "isPaid": True,  
            "storeId": store_filter  
        }},
        # 🏷️ Étape 2 : Trouver **la première transaction payée** de chaque invité
        {"$group": {
            "_id": "$userId",  
            "firstPaidAt": {"$min": "$paidAt"}  
        }},
        # 🏷️ Étape 3 : Extraire **l'année et la semaine** du premier paiement
        {"$addFields": {
            "guest_firstPaidWeek": {"$isoWeek": "$firstPaidAt"},
            "guest_firstPaidYear": {"$isoWeekYear": "$firstPaidAt"}
        }},
        # 🏷️ Étape 4 : Grouper les invités par **année et semaine du premier paiement**
        {"$group": {
            "_id": {"year": "$guest_firstPaidYear", "week": "$guest_firstPaidWeek"},
            "guest_new_users": {"$addToSet": "$_id"}  
        }},
        # 🏷️ Étape 5 : Calculer le nombre total de nouveaux invités par semaine
        {"$project": {
            "_id": 1,
            "guest_total_new_users": {"$size": "$guest_new_users"},  
            "guest_new_users": 1  
        }},
        # 🏷️ Étape 6 : Trier les résultats par **année et semaine**
        {"$sort": {"_id.year": 1, "_id.week": 1}}
    ]

    # 📌 Pipeline MongoDB pour récupérer les **invités actifs** par semaine
    guest_pipeline_active_users_week = [
        # 🏷️ Étape 1 : Filtrer les documents où `userId` commence par "GUEST_" et est payé
        {"$match": {
            "userId": {"$exists": True, "$ne": None, "$regex": "^GUEST_"},  
            "isPaid": True,  
            "storeId": store_filter  
        }},
        # 🏷️ Étape 2 : Extraire **l'année et la semaine** du paiement
        {"$addFields": {
            "guest_paymentWeek": {"$isoWeek": "$paidAt"},
            "guest_paymentYear": {"$isoWeekYear": "$paidAt"}
        }},
        # 🏷️ Étape 3 : Grouper par **année et semaine** et rassembler les invités actifs
        {"$group": {
            "_id": {"year": "$guest_paymentYear", "week": "$guest_paymentWeek"},
            "guest_active_users": {"$addToSet": "$userId"}  
        }},
        # 🏷️ Étape 4 : Calculer le nombre total d'invités actifs par semaine
        {"$project": {
            "_id": 1,
            "guest_total_active_users": {"$size": "$guest_active_users"},  
            "guest_active_users": 1  
        }},
        # 🏷️ Étape 5 : Trier les résultats par **année et semaine**
        {"$sort": {"_id.year": 1, "_id.week": 1}}
    ]

    # 🛠️ Exécution des requêtes MongoDB
    guest_cursor_new_users_week = st.session_state.users_collection.aggregate(guest_pipeline_new_users_week)
    guest_cursor_active_users_week = st.session_state.users_collection.aggregate(guest_pipeline_active_users_week)

    guest_data_new_users_week = list(guest_cursor_new_users_week)
    guest_data_active_users_week = list(guest_cursor_active_users_week)

    # 🚨 Vérification des données avant d'aller plus loin
    if not guest_data_new_users_week or not guest_data_active_users_week:
        st.error("❌ Aucune donnée trouvée pour les invités ! Vérifiez la structure de votre base MongoDB.")
        st.stop()

    # 🗂️ Transformation en DataFrame
    df_guest_new_users_week = pd.DataFrame(guest_data_new_users_week)
    df_guest_active_users_week = pd.DataFrame(guest_data_active_users_week)

    # 📊 Extraction des années et semaines pour structurer les données
    df_guest_new_users_week['guest_year'] = df_guest_new_users_week['_id'].apply(lambda x: x['year'])
    df_guest_new_users_week['guest_week'] = df_guest_new_users_week['_id'].apply(lambda x: x['week'])
    df_guest_active_users_week['guest_year'] = df_guest_active_users_week['_id'].apply(lambda x: x['year'])
    df_guest_active_users_week['guest_week'] = df_guest_active_users_week['_id'].apply(lambda x: x['week'])

    # 📆 Génération de la colonne "début de semaine"
    df_guest_new_users_week['guest_week_start'] = df_guest_new_users_week.apply(lambda x: datetime.fromisocalendar(x['guest_year'], x['guest_week'], 1), axis=1)
    df_guest_active_users_week['guest_week_start'] = df_guest_active_users_week.apply(lambda x: datetime.fromisocalendar(x['guest_year'], x['guest_week'], 1), axis=1)

    # 🔄 Convertir les ensembles d'IDs en chaînes pour faciliter la manipulation
    df_guest_new_users_week['guest_new_users'] = df_guest_new_users_week['guest_new_users'].apply(lambda users: set(str(u) for u in users) if users else set())
    df_guest_active_users_week['guest_active_users'] = df_guest_active_users_week['guest_active_users'].apply(lambda users: set(str(u) for u in users) if users else set())

    # 📌 Trier et indexer par "guest_week_start"
    df_guest_new_users_week = df_guest_new_users_week.sort_values(by='guest_week_start').set_index('guest_week_start')
    df_guest_active_users_week = df_guest_active_users_week.sort_values(by='guest_week_start').set_index('guest_week_start')

    # 📊 Calcul de la rétention hebdomadaire
    guest_week_retention = {}

    for index, row in df_guest_new_users_week.iterrows():
        new_user_set = row['guest_new_users']
        guest_week_retention[index] = {"+0": len(new_user_set)}

        if not new_user_set:
            continue

        # 🔍 Vérifier la rétention des invités actifs dans les semaines suivantes
        future_weeks = df_guest_active_users_week.loc[df_guest_active_users_week.index > index]
        for future_index, future_row in future_weeks.iterrows():
            week_diff = (future_index - index).days // 7
            future_users = future_row['guest_active_users']
            retained_users = len(new_user_set.intersection(future_users)) if isinstance(future_users, set) else 0
            guest_week_retention[index][f"+{week_diff}"] = retained_users

    # 📊 Conversion en DataFrame et affichage
    df_guest_retention_week = pd.DataFrame.from_dict(guest_week_retention, orient='index').fillna(0)

    st.title("📊 Rétention Hebdomadaire - Invités")
    st.subheader("Tableau des cohortes (valeurs numériques)")
    st.dataframe(df_guest_retention_week)

    # 📈 Graphique d'évolution de la rétention
    fig = go.Figure()
    for cohort, row in df_guest_retention_week.iterrows():
        fig.add_trace(go.Scatter(
            x=row.index,
            y=row.values,
            mode="lines",
            name=f"Cohorte {cohort.strftime('%Y-%m-%d')}"
        ))

    fig.update_layout(
        title="Évolution du pourcentage d'utilisateurs invités par cohorte (hebdomadaire)",
        xaxis_title="Semaine après le premier achat",
        yaxis_title="Nombre d'utilisateurs actifs",
        template="plotly_white"
    )

    st.plotly_chart(fig)

elif page == "Rétention" and toggle_view == "Mensuel" and user_type == "Invités":
    print("")

# ========================
# Acquisition des utilisateurs
# ========================
if page == "Acquisition" and toggle_view == "Hebdomadaire":
    
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

    # 📌 Exclure la semaine en cours
    today = datetime.now()
    current_week = today.isocalendar()[1]  # Numéro de la semaine actuelle
    current_year = today.year  # Année actuelle

    df_new_users_per_week = df_new_users_per_week[
        ~((df_new_users_per_week['year'] == current_year) & (df_new_users_per_week['week'] == current_week))
    ]

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

elif page == "Acquisition" and toggle_view == "Mensuel":

    # 📌 Pipeline MongoDB pour récupérer le nombre d'utilisateurs créés par mois
    pipeline_new_users_per_month = [
        {"$match": { 
            "createdAt": {"$gte": date_start, "$lte": date_end}
        }},
        {"$group": {
            "_id": {
                "year": {"$year": "$createdAt"},
                "month": {"$month": "$createdAt"}
            },
            "new_users": {"$sum": 1}
        }},
        {"$sort": {"_id.year": 1, "_id.month": 1}}
    ]

    # 📌 Exécuter la requête MongoDB
    cursor_new_users_per_month = users_collection.aggregate(pipeline_new_users_per_month)
    data_new_users_per_month = list(cursor_new_users_per_month)

    # 📌 Vérification des données
    if not data_new_users_per_month:
        st.error("❌ Aucune donnée trouvée pour les nouveaux utilisateurs par mois !")
        st.stop()

    # 📌 Transformation en DataFrame
    df_new_users_per_month = pd.DataFrame(data_new_users_per_month)

    # 📌 Extraction des années et mois
    df_new_users_per_month['year'] = df_new_users_per_month['_id'].apply(lambda x: x['year'])
    df_new_users_per_month['month'] = df_new_users_per_month['_id'].apply(lambda x: x['month'])

    # 📌 Générer la colonne du début du mois
    df_new_users_per_month['month_start'] = df_new_users_per_month.apply(
        lambda x: datetime(x['year'], x['month'], 1), axis=1
    )

    # 📌 Trier et indexer les données
    df_new_users_per_month = df_new_users_per_month.sort_values(by='month_start').set_index('month_start')

    today = datetime.now()
    current_month = today.month
    current_year = today.year

    df_new_users_per_month = df_new_users_per_month[
        ~((df_new_users_per_month['year'] == current_year) & (df_new_users_per_month['month'] == current_month))
    ]

    st.title("📊 Partie Acquisition (Mensuelle)")
    # 📌 Affichage du tableau des nouveaux utilisateurs par mois
    st.subheader("📅 Nombre de nouveaux utilisateurs par mois")
    st.dataframe(df_new_users_per_month[['new_users']])

    # 📌 Créer une courbe interactive avec Plotly
    fig = px.line(df_new_users_per_month,
                x=df_new_users_per_month.index,
                y="new_users",
                title="📈 Évolution des nouveaux utilisateurs par mois",
                labels={"month_start": "Mois", "new_users": "Nouveaux utilisateurs"},
                markers=True)
    st.subheader("📈 Évolution des nouveaux utilisateurs par mois")
    st.plotly_chart(fig)

    # ========================
    # Weekly Active Users 
    # ========================

if page == "Active Users" and toggle_view == "Hebdomadaire":
    # 🔹 Définir le filtre de base
    match_filter = {
        "receipt.isPaid": True,
        "receipt.paidAt": {"$gte": date_start, "$lte": date_end},
        "receipt.storeId": store_filter
    }

    # 🔹 Ajouter un filtre pour le mode de paiement si un mode spécifique est sélectionné
    if selected_payment_method != "Tous":
        variants = payment_variants.get(selected_payment_method, [selected_payment_method])
        match_filter["receipt.paymentMethod"] = {"$in": variants}

    # ========================
    # 📌 Pipeline pour utilisateurs uniques par SEMAINE
    # ========================
    pipeline_unique_users_per_week = [
        {"$unwind": "$receipt"},
        {"$match": match_filter},
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

    # 🔹 Exécuter la requête MongoDB
    cursor_unique_users_per_week = users_collection.aggregate(pipeline_unique_users_per_week)
    data_unique_users_per_week = list(cursor_unique_users_per_week)

    # 🔹 Vérification des données
    if not data_unique_users_per_week:
        st.error("❌ Aucune donnée trouvée pour les utilisateurs uniques par semaine !")
        st.stop()

    # 🔹 Transformation en DataFrame
    df_unique_users_per_week = pd.DataFrame(data_unique_users_per_week)

    # 🔹 Extraction des années et semaines
    df_unique_users_per_week['year'] = df_unique_users_per_week['_id'].apply(lambda x: x['year'])
    df_unique_users_per_week['week'] = df_unique_users_per_week['_id'].apply(lambda x: x['week'])

    # 🔹 Générer la colonne du début de semaine
    df_unique_users_per_week['week_start'] = df_unique_users_per_week.apply(
        lambda x: datetime.fromisocalendar(x['year'], x['week'], 1), axis=1
    )

    # 🔹 Trier et indexer les données
    df_unique_users_per_week = df_unique_users_per_week.sort_values(by='week_start').set_index('week_start')

    # 🔹 Générer toutes les semaines entre la première et la dernière transaction
    all_weeks = pd.date_range(start=date_start, end=datetime.now(), freq='W-MON')
    df_all_weeks_unique_users = pd.DataFrame(index=all_weeks)
    df_all_weeks_unique_users['total_unique_users'] = 0

    # 🔹 Mettre à jour les valeurs des semaines
    df_all_weeks_unique_users.update(df_unique_users_per_week)

    # 📌 Ajouter toutes les semaines manquantes
    df_all_weeks_unique_users = df_all_weeks_unique_users.reindex(all_weeks, fill_value=0)

    # 📌 Ajouter la dernière semaine si elle est absente avant d'exclure la semaine actuelle
    last_week_start = (datetime.now() - timedelta(weeks=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    if last_week_start not in df_all_weeks_unique_users.index:
        df_all_weeks_unique_users.loc[last_week_start] = 0

    # 📌 Exclure la semaine en cours
    today = datetime.now()
    current_week = today.isocalendar()[1]
    current_year = today.year

    df_all_weeks_unique_users = df_all_weeks_unique_users[
        ~((df_all_weeks_unique_users.index.year == current_year) &
          (df_all_weeks_unique_users.index.to_series().apply(lambda x: x.isocalendar()[1]) == current_week))
    ]

    # 🔹 Affichage
    st.title("Partie Weekly Active Users")
    st.subheader("📊 Tableau utilisateurs uniques par semaine")
    st.dataframe(df_all_weeks_unique_users)

    # 🔹 Graphique
    fig = px.line(df_all_weeks_unique_users,
                  x=df_all_weeks_unique_users.index,
                  y="total_unique_users",
                  title="📈 Évolution des utilisateurs uniques par semaine",
                  labels={"week_start": "Semaine", "total_unique_users": "Utilisateurs uniques"},
                  markers=True)
    st.subheader("📈 Évolution des utilisateurs uniques par semaine")
    st.plotly_chart(fig)

elif page == "Active Users" and toggle_view == "Mensuel":
        # ========================
        # 📌 Pipeline pour utilisateurs uniques par MOIS
        # ========================
        pipeline_unique_users_per_month = [
            {"$unwind": "$receipt"},
            {"$match": match_filter},
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

        # 🔹 Exécuter la requête MongoDB
        cursor_unique_users_per_month = users_collection.aggregate(pipeline_unique_users_per_month)
        data_unique_users_per_month = list(cursor_unique_users_per_month)

        # 🔹 Vérification des données
        if not data_unique_users_per_month:
            st.error("❌ Aucune donnée trouvée pour les utilisateurs uniques par mois !")
            st.stop()

        # 🔹 Transformation en DataFrame
        df_unique_users_per_month = pd.DataFrame(data_unique_users_per_month)

        # 🔹 Extraction des années et mois
        df_unique_users_per_month['year'] = df_unique_users_per_month['_id'].apply(lambda x: x['year'])
        df_unique_users_per_month['month'] = df_unique_users_per_month['_id'].apply(lambda x: x['month'])

        # 🔹 Générer la colonne du début de mois
        df_unique_users_per_month['month_start'] = df_unique_users_per_month.apply(
            lambda x: datetime(x['year'], x['month'], 1), axis=1
        )

        # 🔹 Trier et indexer les données
        df_unique_users_per_month = df_unique_users_per_month.sort_values(by='month_start').set_index('month_start')

        # 🔹 Générer toutes les mois
        all_months = pd.date_range(start=df_unique_users_per_month.index.min(), 
                                end=df_unique_users_per_month.index.max(), 
                                freq='MS')
        df_all_months_unique_users = pd.DataFrame(index=all_months)
        df_all_months_unique_users['total_unique_users'] = 0

        # 🔹 Mettre à jour les valeurs des mois
        df_all_months_unique_users.update(df_unique_users_per_month)
        
        # 📌 Exclure le mois en cours
        today = datetime.now()
        current_month = today.month
        current_year = today.year

        df_all_months_unique_users = df_all_months_unique_users[
            ~((df_all_months_unique_users.index.year == current_year) & 
            (df_all_months_unique_users.index.month == current_month))
        ]


        # 🔹 Affichage
        st.title("Partie Monthly Active Users")
        st.subheader("📊 Tableau utilisateurs uniques par mois")
        st.dataframe(df_all_months_unique_users)

        # 🔹 Graphique
        fig = px.line(df_all_months_unique_users, 
                    x=df_all_months_unique_users.index, 
                    y="total_unique_users", 
                    title="📈 Évolution des utilisateurs uniques par mois",
                    labels={"month_start": "Mois", "total_unique_users": "Utilisateurs uniques"},
                    markers=True)
        st.subheader("📈 Évolution des utilisateurs uniques par mois")
        st.plotly_chart(fig)


# ========================
# Bug Report
# ========================

if page == "Bug Report":
    st.title("Partie Bug Report")
    # 🔹 Définition de la période actuelle (semaine en cours)
    today = datetime.now()
    current_week_start = (today - timedelta(days=today.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    current_week_end = current_week_start + timedelta(days=6, hours=23, minutes=59, seconds=59)

    # 🔹 Récupération des paniers abandonnés
    non_finalized_carts = list(orders_collection.find({
        'isPaid': False,
        'createdAt': {'$gte': current_week_start, '$lte': current_week_end},
        'scanItems': {'$exists': True, '$ne': []}  
    }))

    # 🔹 Récupération des paniers finalisés
    finalized_carts = list(orders_collection.find({
        'isPaid': True,
        'paidAt': {'$gte': current_week_start, '$lte': current_week_end}
    }))

    # 🔹 Nombre total de paniers abandonnés et finalisés
    total_non_finalized = len(non_finalized_carts)
    total_finalized = len(finalized_carts)

    # 🔹 Mapping des magasins
    store_mapping = {
        "65e6388eb6667e3400b5b8d8": "Supermarché Match",
        "65d3631ff2cd066ab75434fa": "Intermarché Saint Julien",
        "662bb3234c362c6e79e27020": "Netto Troyes",
        "64e48f4697303382f745cb11": "Carrefour Contact Buchères",
        "65ce4e565a9ffc7e5fe298bb": "Carrefour Market Romilly",
        "65b8bde65a0ef81ff30473bf": "Jils Food",
        "67a8fef293a9fcb4dec991b4": "Intermarché EXPRESS Clamart"
    }

    # 🔹 Comptage des paniers abandonnés par magasin
    non_finalized_counts = defaultdict(int)
    for cart in non_finalized_carts:
        store_id = cart.get('storeId')
        store_name = store_mapping.get(str(store_id), "Inconnu") 
        non_finalized_counts[store_name] += 1

    # 🔹 Comptage des paniers finalisés par magasin
    finalized_counts = defaultdict(int)
    for cart in finalized_carts:
        store_id = cart.get('storeId')
        store_name = store_mapping.get(str(store_id), "Inconnu") 
        finalized_counts[store_name] += 1

    # 🔹 Conversion en DataFrame et tri des résultats
    non_finalized_df = pd.DataFrame(list(non_finalized_counts.items()), columns=['Magasin', 'Paniers Abandonnés'])
    non_finalized_df = non_finalized_df.sort_values(by='Paniers Abandonnés', ascending=False).reset_index(drop=True)

    finalized_df = pd.DataFrame(list(finalized_counts.items()), columns=['Magasin', 'Paniers Finalisés'])
    finalized_df = finalized_df.sort_values(by='Paniers Finalisés', ascending=False).reset_index(drop=True)

    # 📌 Affichage côte à côte avec espacement amélioré
    tab1, tab2 = st.tabs(["✅ Paniers Finalisés", "🛒 Paniers Abandonnés"])

    with tab1:
        st.subheader("✅ Paniers finalisés de la semaine")
        st.write(f"Nombre total de paniers finalisés : {total_finalized}")
        st.dataframe(finalized_df)  # Tableau des paniers finalisés

    with tab2:
        st.subheader("🛒 Paniers abandonnés de la semaine")
        st.write(f"Nombre total de paniers abandonnés : {total_non_finalized}")
        st.dataframe(non_finalized_df)  # Tableau des paniers abandonnés


