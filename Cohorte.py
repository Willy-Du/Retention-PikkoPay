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

# ------------------------------------------------------
# Partie Rétention Hebdomadaire Utilisateurs Connectés 
# ------------------------------------------------------
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

        # ------------------------------------------------------------
        # Calcul de la rétention hebdomadaire Utilisateurs Connectés
        # ------------------------------------------------------------

            if df_new_users_week.empty:
                st.warning("Aucun nouvel utilisateur trouvé pour ce mode de paiement.")
                df_numeric_week = pd.DataFrame(columns=['total_new_users', '+0'])
                df_percentage_week = pd.DataFrame(columns=['total_new_users', '+0'])
            else:
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
                
                df_numeric_week = all_weeks_df.merge(df_retention_week, left_index=True, right_index=True, how='left')
                
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

            st.plotly_chart(fig)
        
            # ----------------------------------------------------------------
            # Section : Layer Cake Chart Hebdomadaire Utilisateurs Connectés
            # ----------------------------------------------------------------
            st.title("🍰 Partie Layer Cake")

            if 'df_numeric_week' not in locals() or df_numeric_week.empty:
                st.warning("❌ Aucune donnée de rétention hebdomadaire disponible pour générer le Layer Cake.")
                st.stop()

            df_layer_cake = df_numeric_week.copy().fillna(0)

            # Supprimer la colonne "total_new_users" si elle existe
            if "total_new_users" in df_layer_cake.columns:
                df_layer_cake = df_layer_cake.drop(columns=["total_new_users"])

            # Récupérer les colonnes de type "+0", "+1", ...
            retention_cols = [col for col in df_layer_cake.columns if col.startswith("+")]
            retention_cols = sorted(retention_cols, key=lambda c: int(c.replace("+", "")))
            df_layer_cake = df_layer_cake[retention_cols]

            df_layer_cake.sort_index(ascending=True, inplace=True)

            num_weeks = len(retention_cols)
            x_axis = np.arange(num_weeks)

            fig = go.Figure()
            num_cohorts = len(df_layer_cake)
            colormap = cm.get_cmap('tab20c', num_cohorts)

            # Créer une matrice vide pour y placer les courbes décalées
            stacked_matrix = np.full((num_cohorts, num_weeks), np.nan)

            for i, row in enumerate(df_layer_cake.itertuples(index=False)):
                values = np.array(row)
                shifted = [np.nan] * i + list(values[:num_weeks - i])
                stacked_matrix[i, :len(shifted)] = shifted

            # Générer les traces
            for i, (cohort_date, row) in enumerate(df_layer_cake.iterrows()):
                cohort_values = np.array(row.tolist(), dtype=float)
                shifted = [None] * i + list(cohort_values[:num_weeks - i])
                customdata = []
                for j in range(num_weeks):
                    if j < i:
                        customdata.append(None)
                    else:
                        total = np.nansum(stacked_matrix[:i+1, j])
                        customdata.append(total)

                rgba = colormap(i / num_cohorts)
                color = f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]})"

                fig.add_trace(go.Scatter(
                    x=x_axis,
                    y=shifted,
                    mode='lines',
                    stackgroup='one',
                    name=f"Cohorte {cohort_date.strftime('%Y-%m-%d')}",
                    line=dict(color=color),
                    customdata=customdata,
                    hovertemplate=(
                        "<b>Cohorte</b> : %{fullData.name}<br>" +
                        "<b>Semaine</b> : %{x}<br>" +
                        "<b>Utilisateurs de la cohorte</b> : %{y:.0f}<br>" +
                        "<b>Total empilé</b> : %{customdata:.0f}" +
                        "<extra></extra>"
                    )
                ))

            fig.update_xaxes(
                tickmode='array',
                tickvals=list(x_axis),
                ticktext=[f"+{i}" for i in x_axis]
            )

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

# ------------------------------------------------------
# Partie Rétention Mensuel Utilisateurs Connectés 
# ------------------------------------------------------

elif page == "Rétention" and toggle_view == "Mensuel" and user_type == "Utilisateurs Connectés":
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

    cursor_new_users = users_collection.aggregate(pipeline_new_users)
    cursor_active_users = users_collection.aggregate(pipeline_active_users)

    data_new_users = list(cursor_new_users)
    data_active_users = list(cursor_active_users)

    if not data_new_users or not data_active_users:
        st.error("❌ Aucune donnée trouvée ! Vérifiez la structure de votre base MongoDB.")
        st.stop()

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
    # Calcul de la rétention mensuelle
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

    st.title("Partie Rétention - Analyse Mensuelle")
    st.header("📅 Tableau des cohortes mensuelles")
    st.subheader("📊 Cohortes mensuelles (valeurs numériques)")
    st.dataframe(df_final)

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

    # -----------------------------------------------------------
    # Section : Layer Cake Chart Mensuel Utilisateurs Connectés
    # -----------------------------------------------------------
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
# ------------------------------------------------------
# Partie Rétention Hebdomadaire Invités
# ------------------------------------------------------
if page == "Rétention" and toggle_view == "Hebdomadaire" and user_type == "Invités":
    st.title("Partie Rétention - Invités (Hebdomadaire)")

    if isinstance(store_filter, str):
        store_filter = ObjectId(store_filter)

    base_filter_guest = {
        "isPaid": True,
        "storeId": store_filter,
        "paidAt": {"$gte": date_start, "$lte": date_end},
        "userId": { "$exists": True },
        "$or": [
            { "userId": None },
            { "userId": "" },
            { "userId": { "$regex": "^GUEST_", "$options": "i" } }
        ]
    }

    if selected_payment_method != "Tous":
        variants = payment_variants.get(selected_payment_method, [selected_payment_method])
        base_filter_guest["paymentMethod"] = {"$in": variants}

    # 🔹 Nouveaux invités
    pipeline_new_users_week = [
        { "$match": base_filter_guest },
        { "$addFields": {
            "guestKey": {
                "$cond": [
                    { "$or": [
                        { "$eq": ["$userId", None] },
                        { "$eq": ["$userId", ""] }
                    ]},
                    { "$toString": "$_id" },
                    "$userId"
                ]
            }
        }},
        { "$group": {
            "_id": "$guestKey",
            "firstPaidAt": { "$min": "$paidAt" }
        }},
        { "$project": {
            "_id": 0,
            "guestKey": "$_id",
            "year": { "$isoWeekYear": "$firstPaidAt" },
            "week": { "$isoWeek": "$firstPaidAt" }
        }}
    ]

    # 🔹 Actifs par semaine
    pipeline_active_users_week = [
        { "$match": base_filter_guest },
        { "$addFields": {
            "guestKey": {
                "$cond": [
                    { "$or": [
                        { "$eq": ["$userId", None] },
                        { "$eq": ["$userId", ""] }
                    ]},
                    { "$toString": "$_id" },
                    "$userId"
                ]
            }
        }},
        { "$group": {
            "_id": {
                "year": { "$isoWeekYear": "$paidAt" },
                "week": { "$isoWeek": "$paidAt" }
            },
            "active_users": { "$addToSet": "$guestKey" }
        }}
    ]

    # Exécuter les requêtes
    new_users_data = list(st.session_state.orders_collection.aggregate(pipeline_new_users_week))
    active_users_data = list(st.session_state.orders_collection.aggregate(pipeline_active_users_week))

    if not new_users_data or not active_users_data:
        st.error("❌ Aucune donnée trouvée !")
        st.stop()

    # Transformation en DataFrames
    df_new_users_week = pd.DataFrame(new_users_data)
    df_active_users_week = pd.DataFrame(active_users_data)

    df_new_users_week['week_start'] = df_new_users_week.apply(
        lambda x: datetime.fromisocalendar(x['year'], x['week'], 1), axis=1)
    df_new_users_week = df_new_users_week.groupby('week_start')['guestKey'].apply(set).reset_index()
    df_new_users_week = df_new_users_week.rename(columns={'guestKey': 'new_users'})
    df_new_users_week = df_new_users_week.set_index('week_start')

    df_active_users_week['week_start'] = df_active_users_week['_id'].apply(
        lambda x: datetime.fromisocalendar(x['year'], x['week'], 1))
    df_active_users_week['active_users'] = df_active_users_week['active_users'].apply(
        lambda users: set(str(u) for u in users) if users else set())
    df_active_users_week = df_active_users_week[['week_start', 'active_users']].set_index('week_start')

    # ➕ Calcul de la rétention
    current_date = datetime.now()
    current_week_start = datetime.fromisocalendar(current_date.year, current_date.isocalendar()[1], 1)

    start_date = datetime(2025, 3, 17)
    all_weeks_range = pd.date_range(start=start_date, end=current_week_start - timedelta(days=1), freq='W-MON')

    week_retention = {}
    filtered_df_new_users_week = df_new_users_week[df_new_users_week.index < current_week_start]
    all_cohort_dates = set(all_weeks_range) | set(filtered_df_new_users_week.index)

    for idx in all_cohort_dates:
        week_retention[idx] = {"+0": 0}

    for index, row in filtered_df_new_users_week.iterrows():
        new_user_set = row['new_users']
        week_retention[index]["+0"] = len(new_user_set)
        if not new_user_set:
            continue
        future_weeks = df_active_users_week.loc[(df_active_users_week.index > index) & 
                                                (df_active_users_week.index < current_week_start)]
        for future_index, future_row in future_weeks.iterrows():
            week_diff = (future_index - index).days // 7
            retained_users = len(new_user_set.intersection(future_row['active_users']))
            week_retention[index][f"+{week_diff}"] = retained_users

    df_retention_week = pd.DataFrame.from_dict(week_retention, orient='index')
    global_max = max((current_week_start - timedelta(days=7) - idx).days // 7 for idx in df_retention_week.index)

    for index, row in df_retention_week.iterrows():
        for week_diff in range(global_max + 1):
            col_name = f"+{week_diff}"
            future_week = index + timedelta(weeks=week_diff)
            if future_week >= current_week_start:
                df_retention_week.at[index, col_name] = None
            elif pd.isna(row.get(col_name, None)):
                df_retention_week.at[index, col_name] = 0

    all_weeks_df = pd.DataFrame(index=sorted(all_cohort_dates))
    all_weeks_df['total_new_users'] = 0
    for idx in filtered_df_new_users_week.index:
        if idx in all_weeks_df.index:
            all_weeks_df.loc[idx, 'total_new_users'] = len(filtered_df_new_users_week.loc[idx, 'new_users'])

    df_numeric_week = all_weeks_df.merge(df_retention_week, left_index=True, right_index=True, how='left')

    retention_cols = sorted([col for col in df_numeric_week.columns if col.startswith("+")],
                            key=lambda x: int(x.replace("+", "")))
    other_cols = [col for col in df_numeric_week.columns if not col.startswith("+")]
    df_numeric_week = df_numeric_week[other_cols + retention_cols]

    # Pourcentage
    df_percentage_week = df_numeric_week.copy()
    for col in df_percentage_week.columns:
        if col.startswith("+") and col != "+0":
            mask = (df_percentage_week["+0"] > 0) & (df_percentage_week[col].notna())
            df_percentage_week.loc[mask, col] = (
                df_percentage_week.loc[mask, col] / df_percentage_week.loc[mask, "+0"] * 100
            ).round(1)
    df_percentage_week["+0"] = df_percentage_week["+0"].apply(lambda x: 100 if x > 0 else 0)

    def apply_red_gradient_with_future(val):
        if pd.isna(val):
            return 'background-color: #f0f0f0; color: #f0f0f0;'
        elif pd.notna(val):
            intensity = int(255 * ((1 - val / 100) ** 3))
            return f'background-color: rgba(255, {intensity}, {intensity}, 1); color: black;'
        return ''

    # 🔸 AFFICHAGE TABLEAUX
    st.header("📅 Tableau des cohortes hebdomadaires - Invités (valeurs)")
    st.dataframe(df_numeric_week)

    st.subheader("📊 Tableau des cohortes hebdomadaires - Invités (%)")
    st.dataframe(
        df_percentage_week.style.applymap(
            apply_red_gradient_with_future,
            subset=[col for col in df_percentage_week.columns if col.startswith("+")]
        )
    )

    # 📈 Courbe de rétention hebdomadaire - Invités (style cohorte connecté)
    st.subheader("📈 Courbe de rétention hebdomadaire des invités")

    df_plot = df_percentage_week.copy()
    fig = go.Figure()
    colormap = cm.get_cmap('tab20c', len(df_plot))

    for i, (idx, row) in enumerate(df_plot.iterrows()):
        valid_values = row[[col for col in row.index if col.startswith("+")]].dropna()
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

    # 🔸 Ajouter la courbe moyenne
    average_curve = df_plot[[col for col in df_plot.columns if col.startswith("+")]].mean(axis=0, skipna=True)
    fig.add_trace(go.Scatter(
        x=average_curve.index,
        y=average_curve.values,
        mode='lines',
        name='Moyenne par +x',
        line=dict(width=3, color='black'),
        opacity=1.0
    ))

    fig.update_layout(
        title="📊 Rétention hebdomadaire des invités (%)",
        xaxis_title="Semaine après premier achat",
        yaxis_title="Pourcentage de rétention",
        template="plotly_white",
        xaxis=dict(
            tickmode='array',
            tickvals=[f'+{i}' for i in range(len(average_curve))]
        ),
        yaxis=dict(
            tickformat=".1f",
            range=[0, 110]
        ),
        height=500
    )

    st.plotly_chart(fig)


# ------------------------------------------------------
# Partie Rétention Mensuel Invités 
# ------------------------------------------------------

if page == "Rétention" and toggle_view == "Mensuel" and user_type == "Invités":
    st.title("Partie Rétention - Invités (Mensuelle)")

    if isinstance(store_filter, str):
        store_filter = ObjectId(store_filter)

    base_filter_guest = {
        "isPaid": True,
        "storeId": store_filter,
        "paidAt": {"$gte": date_start, "$lte": date_end},
        "userId": { "$exists": True },
        "$or": [
            { "userId": None },
            { "userId": "" },
            { "userId": { "$regex": "^GUEST_", "$options": "i" } }
        ]
    }

    if selected_payment_method != "Tous":
        variants = payment_variants.get(selected_payment_method, [selected_payment_method])
        base_filter_guest["paymentMethod"] = {"$in": variants}

    # 🔹 Nouveaux invités mensuels
    pipeline_new_users_month = [
        { "$match": base_filter_guest },
        { "$addFields": {
            "guestKey": {
                "$cond": [
                    { "$or": [
                        { "$eq": ["$userId", None] },
                        { "$eq": ["$userId", ""] }
                    ]},
                    { "$toString": "$_id" },
                    "$userId"
                ]
            }
        }},
        { "$group": {
            "_id": "$guestKey",
            "firstPaidAt": { "$min": "$paidAt" }
        }},
        { "$project": {
            "_id": 0,
            "guestKey": "$_id",
            "year": { "$year": "$firstPaidAt" },
            "month": { "$month": "$firstPaidAt" }
        }}
    ]

    pipeline_active_users_month = [
        { "$match": base_filter_guest },
        { "$addFields": {
            "guestKey": {
                "$cond": [
                    { "$or": [
                        { "$eq": ["$userId", None] },
                        { "$eq": ["$userId", ""] }
                    ]},
                    { "$toString": "$_id" },
                    "$userId"
                ]
            }
        }},
        { "$group": {
            "_id": {
                "year": { "$year": "$paidAt" },
                "month": { "$month": "$paidAt" }
            },
            "active_users": { "$addToSet": "$guestKey" }
        }}
    ]

    new_data = list(st.session_state.orders_collection.aggregate(pipeline_new_users_month))
    active_data = list(st.session_state.orders_collection.aggregate(pipeline_active_users_month))

    if not new_data or not active_data:
        st.error("❌ Aucune donnée trouvée.")
        st.stop()

    # ➕ Transform into DataFrames
    df_new = pd.DataFrame(new_data)
    df_active = pd.DataFrame(active_data)

    df_new['month_start'] = df_new.apply(lambda x: datetime(x['year'], x['month'], 1), axis=1)
    df_new = df_new.groupby('month_start')['guestKey'].apply(set).reset_index()
    df_new = df_new.rename(columns={'guestKey': 'new_users'}).set_index('month_start')

    df_active['month_start'] = df_active['_id'].apply(lambda x: datetime(x['year'], x['month'], 1))
    df_active['active_users'] = df_active['active_users'].apply(
        lambda users: set(str(u) for u in users) if users else set()
    )
    df_active = df_active[['month_start', 'active_users']].set_index('month_start')

    # 📅 Cohortes mensuelles
    today = datetime.now()
    current_month_start = datetime(today.year, today.month, 1)
    all_months = pd.date_range(start=datetime(2025, 3, 1), end=current_month_start - timedelta(days=1), freq='MS')

    retention = {}
    df_new_filtered = df_new[df_new.index < current_month_start]
    all_cohort_dates = set(all_months) | set(df_new_filtered.index)

    for idx in all_cohort_dates:
        retention[idx] = {"+0": 0}

    for index, row in df_new_filtered.iterrows():
        new_users = row['new_users']
        retention[index]["+0"] = len(new_users)
        if not new_users:
            continue
        future_months = df_active.loc[(df_active.index > index) & (df_active.index < current_month_start)]
        for future_index, future_row in future_months.iterrows():
            month_diff = (future_index.year - index.year) * 12 + (future_index.month - index.month)
            retained = len(new_users.intersection(future_row['active_users']))
            retention[index][f"+{month_diff}"] = retained

    df_retention = pd.DataFrame.from_dict(retention, orient='index')
    global_max = max(
        [(current_month_start.year - idx.year) * 12 + (current_month_start.month - idx.month - 1)
        for idx in df_retention.index if idx < current_month_start],
        default=0
    )


    for index, row in df_retention.iterrows():
        for i in range(global_max + 1):
            col = f"+{i}"
            future_month = datetime(index.year, index.month, 1) + pd.DateOffset(months=i)
            if future_month >= current_month_start:
                df_retention.at[index, col] = None
            elif pd.isna(row.get(col, None)):
                df_retention.at[index, col] = 0

    df_all_months = pd.DataFrame(index=sorted(all_cohort_dates))
    df_all_months['total_new_users'] = 0
    for idx in df_new_filtered.index:
        df_all_months.loc[idx, 'total_new_users'] = len(df_new_filtered.loc[idx, 'new_users'])

    df_numeric_month = df_all_months.merge(df_retention, left_index=True, right_index=True, how='left')
    retention_cols = sorted([c for c in df_numeric_month.columns if c.startswith("+")], key=lambda x: int(x[1:]))
    other_cols = [c for c in df_numeric_month.columns if not c.startswith("+")]
    df_numeric_month = df_numeric_month[other_cols + retention_cols]

    # % Rétention
    df_percentage_month = df_numeric_month.copy()
    if "+0" in df_percentage_month.columns:
        for col in df_percentage_month.columns:
            if col.startswith("+") and col != "+0":
                mask = (df_percentage_month["+0"] > 0) & (df_percentage_month[col].notna())
                df_percentage_month.loc[mask, col] = (
                    df_percentage_month.loc[mask, col] / df_percentage_month.loc[mask, "+0"] * 100
                ).round(1)
        df_percentage_month["+0"] = df_percentage_month["+0"].apply(lambda x: 100 if x > 0 else 0)

    def apply_red_gradient(val):
        if pd.isna(val):
            return 'background-color: #f0f0f0; color: #f0f0f0;'
        intensity = int(255 * ((1 - val / 100) ** 3))
        return f'background-color: rgba(255, {intensity}, {intensity}, 1); color: black;'

    # 📊 Tableau brut
    st.subheader("📅 Tableau des cohortes mensuelles - Invités (valeurs)")
    st.dataframe(df_numeric_month)

    st.subheader("📊 Tableau des cohortes mensuelles - Invités (%)")
    st.dataframe(
        df_percentage_month.style.applymap(
            apply_red_gradient,
            subset=[col for col in df_percentage_month.columns if col.startswith("+")]
        )
    )

    # 📈 Courbe de rétention (style cohorte connecté)
    st.subheader("📈 Courbe de rétention mensuelle des invités")
    df_plot = df_percentage_month.copy()
    fig = go.Figure()
    colormap = cm.get_cmap('tab20c', len(df_plot))

    for i, (idx, row) in enumerate(df_plot.iterrows()):
        valid_values = row[[col for col in row.index if col.startswith("+")]].dropna()
        if "+0" not in valid_values.index:
            continue
        rgba = colormap(i / len(df_plot))
        color = f'rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]})'
        fig.add_trace(go.Scatter(
            x=valid_values.index,
            y=valid_values.values,
            mode='lines',
            name=f"Cohorte {idx.strftime('%Y-%m')}",
            line=dict(width=2, color=color),
            hoverinfo='x+y',
            opacity=0.8
        ))

    # Moyenne
    average_curve = df_plot[[col for col in df_plot.columns if col.startswith("+")]].mean(axis=0, skipna=True)
    fig.add_trace(go.Scatter(
        x=average_curve.index,
        y=average_curve.values,
        mode='lines',
        name='Moyenne par +x',
        line=dict(width=3, color='black'),
        opacity=1.0
    ))

    fig.update_layout(
        title="📊 Rétention mensuelle des invités (%)",
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
        ),
        height=500
    )

    st.plotly_chart(fig)

# ------------------------------------------------------
# Partie Rétention Hebdomadaire Tous
# ------------------------------------------------------

if page == "Rétention" and toggle_view == "Hebdomadaire" and user_type == "Tous":
    st.title("Partie Rétention - Tous (Connectés + Invités)")

    # ----------- 🔁 PIPELINES INVITÉS -----------

    # Base filter invités
    base_filter_guest = {
        "isPaid": True,
        "storeId": store_filter,
        "paidAt": {"$gte": date_start, "$lte": date_end},
        "userId": {"$exists": True},
        "$or": [
            {"userId": None},
            {"userId": ""},
            {"userId": {"$regex": "^GUEST_", "$options": "i"}}
        ]
    }
    if selected_payment_method != "Tous":
        variants = payment_variants.get(selected_payment_method, [selected_payment_method])
        base_filter_guest["paymentMethod"] = {"$in": variants}

    # Nouveaux invités
    pipeline_new_users_week_guests = [
        {"$match": base_filter_guest},
        {"$addFields": {
            "guestKey": {
                "$cond": [
                    {"$or": [
                        {"$eq": ["$userId", None]},
                        {"$eq": ["$userId", ""]}
                    ]},
                    {"$toString": "$_id"},
                    "$userId"
                ]
            }
        }},
        {"$group": {
            "_id": "$guestKey",
            "firstPaidAt": {"$min": "$paidAt"}
        }},
        {"$project": {
            "_id": 0,
            "guestKey": "$_id",
            "year": {"$isoWeekYear": "$firstPaidAt"},
            "week": {"$isoWeek": "$firstPaidAt"}
        }}
    ]

    # Actifs invités
    pipeline_active_users_week_guests = [
        {"$match": base_filter_guest},
        {"$addFields": {
            "guestKey": {
                "$cond": [
                    {"$or": [
                        {"$eq": ["$userId", None]},
                        {"$eq": ["$userId", ""]}
                    ]},
                    {"$toString": "$_id"},
                    "$userId"
                ]
            }
        }},
        {"$group": {
            "_id": {
                "year": {"$isoWeekYear": "$paidAt"},
                "week": {"$isoWeek": "$paidAt"}
            },
            "active_users": {"$addToSet": "$guestKey"}
        }}
    ]

    # ----------- 🔁 PIPELINES CONNECTÉS -----------

    # Tester exclusion
    testers_to_exclude = [
        ObjectId("66df2f59c1271156d5468044"),
        ObjectId("670f97d3f38642c54d678d26"),
        ObjectId("65c65360b03953a598253426"),
        ObjectId("65bcb0e43956788471c88e31")
    ]

    if selected_payment_method == "Tous":
        match_filter = base_filter
        pipeline_new_users_week_connected = [
            {"$unwind": "$receipt"},
            {"$match": match_filter},
            {"$sort": {"receipt.paidAt": 1}},
            {"$group": {"_id": "$_id", "firstPaidAt": {"$first": "$receipt.paidAt"}}},
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
            }}
        ]
        pipeline_active_users_week_connected = [
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
        ]
    else:
        variants = payment_variants.get(selected_payment_method, [selected_payment_method])
        match_filter = {**base_filter, "receipt.paymentMethod": {"$in": variants}}
        pipeline_new_users_week_connected = [
            {"$unwind": "$receipt"},
            {"$match": match_filter},
            {"$group": {"_id": "$_id"}},
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
                "user_data.receipt.paymentMethod": {"$in": variants}
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
            }}
        ]
        pipeline_active_users_week_connected = [
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
        ]

    # ----------- 🧾 EXÉCUTION DES QUERIES -----------

    guest_new_users = list(st.session_state.orders_collection.aggregate(pipeline_new_users_week_guests))
    guest_active_users = list(st.session_state.orders_collection.aggregate(pipeline_active_users_week_guests))
    connected_new_users = list(st.session_state.users_collection.aggregate(pipeline_new_users_week_connected))
    connected_active_users = list(st.session_state.users_collection.aggregate(pipeline_active_users_week_connected))

    # ----------- 🧬 TRANSFORMATION DATAFRAMES -----------

    # Nouveaux invités
    df_guests_new = pd.DataFrame(guest_new_users)
    df_guests_new['new_users'] = df_guests_new['guestKey'].apply(lambda x: set([str(x)]))
    df_guests_new['week_start'] = df_guests_new.apply(lambda x: datetime.fromisocalendar(x['year'], x['week'], 1), axis=1)
    df_guests_new = df_guests_new[['week_start', 'new_users']].set_index('week_start')

    # Nouveaux connectés
    df_connected_new = pd.DataFrame(connected_new_users)
    df_connected_new['new_users'] = df_connected_new['new_users'].apply(lambda x: set(str(uid) for uid in x))
    df_connected_new['week_start'] = df_connected_new.apply(lambda x: datetime.fromisocalendar(x['_id']['year'], x['_id']['week'], 1), axis=1)
    df_connected_new = df_connected_new[['week_start', 'new_users']].set_index('week_start')

    # Fusion des nouveaux utilisateurs
    combined_new_users_week = pd.concat([df_guests_new, df_connected_new], axis=0)
    combined_new_users_week = combined_new_users_week.groupby(combined_new_users_week.index).agg({'new_users': lambda sets: set().union(*sets)})
    combined_new_users_week['total_new_users'] = combined_new_users_week['new_users'].apply(len)

    # Utilisateurs actifs invités
    df_guests_active = pd.DataFrame(guest_active_users)
    df_guests_active['week_start'] = df_guests_active['_id'].apply(lambda x: datetime.fromisocalendar(x['year'], x['week'], 1))
    df_guests_active['active_users'] = df_guests_active['active_users'].apply(lambda x: set(str(uid) for uid in x))
    df_guests_active = df_guests_active[['week_start', 'active_users']].set_index('week_start')

    # Utilisateurs actifs connectés
    df_connected_active = pd.DataFrame(connected_active_users)
    df_connected_active['week_start'] = df_connected_active['_id'].apply(lambda x: datetime.fromisocalendar(x['year'], x['week'], 1))
    df_connected_active['active_users'] = df_connected_active['active_users'].apply(lambda x: set(str(uid) for uid in x))
    df_connected_active = df_connected_active[['week_start', 'active_users']].set_index('week_start')

    # Fusion des utilisateurs actifs
    combined_active_users_week = pd.concat([df_guests_active, df_connected_active], axis=0)
    combined_active_users_week = combined_active_users_week.groupby(combined_active_users_week.index).agg({'active_users': lambda sets: set().union(*sets)})

    # ----------- 🔁 Rétention Hebdomadaire - Connectés + Invités -----------

    current_date = datetime.now()
    current_week_start = datetime.fromisocalendar(current_date.year, current_date.isocalendar()[1], 1)

    # Générer la plage complète de semaines, en excluant la semaine actuelle
    all_weeks_range = pd.date_range(start=date_start, end=current_week_start - timedelta(days=1), freq='W-MON')

    # Filtrer les nouvelles cohortes avant la semaine en cours
    filtered_combined_new_users = combined_new_users_week[combined_new_users_week.index < current_week_start]
    all_cohort_dates = set(all_weeks_range) | set(filtered_combined_new_users.index)

    # Initialiser le dictionnaire de rétention
    week_retention = {idx: {"+0": 0} for idx in all_cohort_dates}

    # Calcul +0 et intersections
    for index, row in filtered_combined_new_users.iterrows():
        new_user_set = row['new_users']
        week_retention[index]["+0"] = len(new_user_set)
        if not new_user_set:
            continue

        future_weeks = combined_active_users_week.loc[
            (combined_active_users_week.index > index) &
            (combined_active_users_week.index < current_week_start)
        ]

        for future_index, future_row in future_weeks.iterrows():
            week_diff = (future_index - index).days // 7
            retained_users = len(new_user_set.intersection(future_row['active_users']))
            week_retention[index][f"+{week_diff}"] = retained_users

    # Convertir en DataFrame
    df_retention_week = pd.DataFrame.from_dict(week_retention, orient='index')

    # Compléter les colonnes manquantes
    global_max = max((current_week_start - timedelta(days=7) - idx).days // 7 for idx in df_retention_week.index)

    for index, row in df_retention_week.iterrows():
        for week_diff in range(global_max + 1):
            col_name = f"+{week_diff}"
            future_week = index + timedelta(weeks=week_diff)
            if future_week >= current_week_start:
                df_retention_week.at[index, col_name] = None
            elif pd.isna(row.get(col_name, None)):
                df_retention_week.at[index, col_name] = 0

    # 🧮 Total nouveaux utilisateurs par semaine
    all_weeks_df = pd.DataFrame(index=sorted(all_cohort_dates))
    all_weeks_df['total_new_users'] = 0
    for idx in filtered_combined_new_users.index:
        if idx in all_weeks_df.index:
            all_weeks_df.loc[idx, 'total_new_users'] = len(filtered_combined_new_users.loc[idx, 'new_users'])

    # Fusion avec la rétention
    df_numeric_week = all_weeks_df.merge(df_retention_week, left_index=True, right_index=True, how='left')

    # Organisation colonnes
    retention_cols = sorted([col for col in df_numeric_week.columns if col.startswith("+")],
                            key=lambda x: int(x.replace("+", "")))
    other_cols = [col for col in df_numeric_week.columns if not col.startswith("+")]
    df_numeric_week = df_numeric_week[other_cols + retention_cols]

    # Pourcentage
    df_percentage_week = df_numeric_week.copy()
    for col in df_percentage_week.columns:
        if col.startswith("+") and col != "+0":
            mask = (df_percentage_week["+0"] > 0) & (df_percentage_week[col].notna())
            df_percentage_week.loc[mask, col] = (
                df_percentage_week.loc[mask, col] / df_percentage_week.loc[mask, "+0"] * 100
            ).round(1)
    df_percentage_week["+0"] = df_percentage_week["+0"].apply(lambda x: 100 if x > 0 else 0)

    # Gradient style
    def apply_red_gradient_with_future(val):
        if pd.isna(val):
            return 'background-color: #f0f0f0; color: #f0f0f0;'
        elif pd.notna(val):
            intensity = int(255 * ((1 - val / 100) ** 3))
            return f'background-color: rgba(255, {intensity}, {intensity}, 1); color: black;'
        return ''

    # 🔎 AFFICHAGE TABLEAUX
    st.header("📅 Tableau des cohortes hebdomadaires - Tous (valeurs)")
    st.dataframe(df_numeric_week)

    st.subheader("📊 Tableau des cohortes hebdomadaires - Tous (%)")
    st.dataframe(
        df_percentage_week.style.applymap(
            apply_red_gradient_with_future,
            subset=[col for col in df_percentage_week.columns if col.startswith("+")]
        )
    )

    # 📈 COURBE DE RÉTENTION
    st.subheader("📈 Courbe de rétention hebdomadaire - Tous")

    df_plot = df_percentage_week.copy()
    fig = go.Figure()
    colormap = cm.get_cmap('tab20c', len(df_plot))

    for i, (idx, row) in enumerate(df_plot.iterrows()):
        valid_values = row[[col for col in row.index if col.startswith("+")]].dropna()
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

    # Moyenne
    average_curve = df_plot[[col for col in df_plot.columns if col.startswith("+")]].mean(axis=0, skipna=True)
    fig.add_trace(go.Scatter(
        x=average_curve.index,
        y=average_curve.values,
        mode='lines',
        name='Moyenne par +x',
        line=dict(width=3, color='black'),
        opacity=1.0
    ))

    fig.update_layout(
        title="📊 Rétention hebdomadaire (%) - Utilisateurs Connectés + Invités",
        xaxis_title="Semaine après le premier achat",
        yaxis_title="Pourcentage de rétention",
        template="plotly_white",
        xaxis=dict(
            tickmode='array',
            tickvals=[f'+{i}' for i in range(len(average_curve))]
        ),
        yaxis=dict(
            tickformat=".1f",
            range=[0, 110]
        ),
        height=500
    )

    st.plotly_chart(fig)

    st.title("🍰 Layer Cake - Tous (Utilisateurs Connectés + Invités)")

    if 'df_numeric_week' not in locals() or df_numeric_week.empty:
        st.warning("❌ Aucune donnée de rétention hebdomadaire disponible pour générer le Layer Cake.")
        st.stop()

    df_layer_cake = df_numeric_week.copy().fillna(0)

    # Supprimer la colonne "total_new_users" si elle existe
    if "total_new_users" in df_layer_cake.columns:
        df_layer_cake = df_layer_cake.drop(columns=["total_new_users"])

    # Garder uniquement les colonnes "+0", "+1", etc.
    retention_cols = [col for col in df_layer_cake.columns if col.startswith("+")]
    retention_cols = sorted(retention_cols, key=lambda c: int(c.replace("+", "")))
    df_layer_cake = df_layer_cake[retention_cols]

    # S'assurer que les index sont triés (par semaine de cohorte)
    df_layer_cake.sort_index(ascending=True, inplace=True)

    num_weeks = len(retention_cols)
    x_axis = np.arange(num_weeks)

    fig = go.Figure()
    num_cohorts = len(df_layer_cake)
    colormap = cm.get_cmap('tab20c', num_cohorts)

    # Créer une matrice pour stocker les courbes décalées (utilisée pour les totaux empilés)
    stacked_matrix = np.full((num_cohorts, num_weeks), np.nan)

    for i, row in enumerate(df_layer_cake.itertuples(index=False)):
        values = np.array(row)
        shifted = [np.nan] * i + list(values[:num_weeks - i])
        stacked_matrix[i, :len(shifted)] = shifted

    # Générer les traces pour chaque cohorte
    for i, (cohort_date, row) in enumerate(df_layer_cake.iterrows()):
        cohort_values = np.array(row.tolist(), dtype=float)
        shifted = [None] * i + list(cohort_values[:num_weeks - i])
        
        customdata = []
        for j in range(num_weeks):
            if j < i:
                customdata.append(None)
            else:
                total = np.nansum(stacked_matrix[:i+1, j])
                customdata.append(total)

        rgba = colormap(i / num_cohorts)
        color = f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]})"

        fig.add_trace(go.Scatter(
            x=x_axis,
            y=shifted,
            mode='lines',
            stackgroup='one',
            name=f"Cohorte {cohort_date.strftime('%Y-%m-%d')}",
            line=dict(color=color),
            customdata=customdata,
            hovertemplate=(
                "<b>Cohorte</b> : %{fullData.name}<br>" +
                "<b>Semaine</b> : %{x}<br>" +
                "<b>Utilisateurs de la cohorte</b> : %{y:.0f}<br>" +
                "<b>Total empilé</b> : %{customdata:.0f}" +
                "<extra></extra>"
            )
        ))

    fig.update_xaxes(
        tickmode='array',
        tickvals=list(x_axis),
        ticktext=[f"+{i}" for i in x_axis]
    )

    fig.update_layout(
        title="📊 Layer Cake Chart - Rétention Hebdomadaire - Utilisateurs Connectés + Invités",
        xaxis_title="Semaines après le premier achat",
        yaxis_title="Nombre d'utilisateurs cumulés",
        template="plotly_white",
        legend_title="Cohortes hebdomadaires"
    )

    st.plotly_chart(fig)

# ------------------------------------------------------
# Partie Rétention Mensuel Tous
# ------------------------------------------------------
if page == "Rétention" and toggle_view == "Mensuel" and user_type == "Tous":
    st.title("Partie Rétention - Tous (Connectés + Invités - Mensuel)")

    # ------------------ INVITÉS ------------------
    base_filter_guest = {
        "isPaid": True,
        "storeId": store_filter,
        "paidAt": {"$gte": date_start, "$lte": date_end},
        "userId": {"$exists": True},
        "$or": [
            {"userId": None},
            {"userId": ""},
            {"userId": {"$regex": "^GUEST_", "$options": "i"}}
        ]
    }
    if selected_payment_method != "Tous":
        variants = payment_variants.get(selected_payment_method, [selected_payment_method])
        base_filter_guest["paymentMethod"] = {"$in": variants}

    pipeline_new_users_month_guests = [
        {"$match": base_filter_guest},
        {"$addFields": {
            "guestKey": {
                "$cond": [
                    {"$or": [
                        {"$eq": ["$userId", None]},
                        {"$eq": ["$userId", ""]}
                    ]},
                    {"$toString": "$_id"},
                    "$userId"
                ]
            }
        }},
        {"$group": {
            "_id": "$guestKey",
            "firstPaidAt": {"$min": "$paidAt"}
        }},
        {"$project": {
            "guestKey": "$_id",
            "year": {"$year": "$firstPaidAt"},
            "month": {"$month": "$firstPaidAt"}
        }}
    ]

    pipeline_active_users_month_guests = [
        {"$match": base_filter_guest},
        {"$addFields": {
            "guestKey": {
                "$cond": [
                    {"$or": [
                        {"$eq": ["$userId", None]},
                        {"$eq": ["$userId", ""]}
                    ]},
                    {"$toString": "$_id"},
                    "$userId"
                ]
            }
        }},
        {"$group": {
            "_id": {
                "year": {"$year": "$paidAt"},
                "month": {"$month": "$paidAt"}
            },
            "active_users": {"$addToSet": "$guestKey"}
        }}
    ]

    # ------------------ CONNECTÉS ------------------
    testers_to_exclude = [
        ObjectId("66df2f59c1271156d5468044"),
        ObjectId("670f97d3f38642c54d678d26"),
        ObjectId("65c65360b03953a598253426"),
        ObjectId("65bcb0e43956788471c88e31")
    ]

    if selected_payment_method == "Tous":
        match_filter = base_filter
        pipeline_new_users_month_connected = [
            {"$unwind": "$receipt"},
            {"$match": match_filter},
            {"$sort": {"receipt.paidAt": 1}},
            {"$group": {"_id": "$_id", "firstPaidAt": {"$first": "$receipt.paidAt"}}},
            {"$addFields": {
                "year": {"$year": "$firstPaidAt"},
                "month": {"$month": "$firstPaidAt"}
            }},
            {"$group": {
                "_id": {"year": "$year", "month": "$month"},
                "new_users": {"$addToSet": "$_id"}
            }}
        ]
        pipeline_active_users_month_connected = [
            {"$unwind": "$receipt"},
            {"$match": {**base_filter, "_id": {"$nin": testers_to_exclude}}},
            {"$addFields": {
                "year": {"$year": "$receipt.paidAt"},
                "month": {"$month": "$receipt.paidAt"}
            }},
            {"$group": {
                "_id": {"year": "$year", "month": "$month"},
                "active_users": {"$addToSet": "$_id"}
            }}
        ]
    else:
        variants = payment_variants.get(selected_payment_method, [selected_payment_method])
        match_filter = {**base_filter, "receipt.paymentMethod": {"$in": variants}}
        pipeline_new_users_month_connected = [
            {"$unwind": "$receipt"},
            {"$match": match_filter},
            {"$group": {"_id": "$_id"}},
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
                "user_data.receipt.paymentMethod": {"$in": variants}
            }},
            {"$sort": {"user_data.receipt.paidAt": 1}},
            {"$group": {
                "_id": "$_id",
                "firstPaidAt": {"$first": "$user_data.receipt.paidAt"}
            }},
            {"$addFields": {
                "year": {"$year": "$firstPaidAt"},
                "month": {"$month": "$firstPaidAt"}
            }},
            {"$group": {
                "_id": {"year": "$year", "month": "$month"},
                "new_users": {"$addToSet": "$_id"}
            }}
        ]
        pipeline_active_users_month_connected = [
            {"$unwind": "$receipt"},
            {"$match": {**match_filter, "_id": {"$nin": testers_to_exclude}}},
            {"$addFields": {
                "year": {"$year": "$receipt.paidAt"},
                "month": {"$month": "$receipt.paidAt"}
            }},
            {"$group": {
                "_id": {"year": "$year", "month": "$month"},
                "active_users": {"$addToSet": "$_id"}
            }}
        ]

    # 🧾 Exécution des requêtes
    guest_new_users = list(st.session_state.orders_collection.aggregate(pipeline_new_users_month_guests))
    guest_active_users = list(st.session_state.orders_collection.aggregate(pipeline_active_users_month_guests))
    connected_new_users = list(st.session_state.users_collection.aggregate(pipeline_new_users_month_connected))
    connected_active_users = list(st.session_state.users_collection.aggregate(pipeline_active_users_month_connected))

    # ----------- 📆 TRANSFORMATION EN DATAFRAMES MENSUELS -----------

    # 🔹 Nouveaux utilisateurs - Invités
    df_guests_new = pd.DataFrame(guest_new_users)
    if not df_guests_new.empty:
        df_guests_new['new_users'] = df_guests_new['guestKey'].apply(lambda x: set([str(x)]))
        df_guests_new['month_start'] = df_guests_new.apply(lambda x: datetime(x['year'], x['month'], 1), axis=1)
        df_guests_new = df_guests_new[['month_start', 'new_users']].set_index('month_start')
    else:
        df_guests_new = pd.DataFrame(columns=['new_users'])

    # 🔹 Nouveaux utilisateurs - Connectés
    df_connected_new = pd.DataFrame(connected_new_users)
    if not df_connected_new.empty:
        df_connected_new['new_users'] = df_connected_new['new_users'].apply(lambda x: set(str(u) for u in x))
        df_connected_new['month_start'] = df_connected_new['_id'].apply(lambda x: datetime(x['year'], x['month'], 1))
        df_connected_new = df_connected_new[['month_start', 'new_users']].set_index('month_start')
    else:
        df_connected_new = pd.DataFrame(columns=['new_users'])

    # 🔀 Fusion des nouveaux utilisateurs
    combined_new_users_month = pd.concat([df_guests_new, df_connected_new])
    combined_new_users_month = combined_new_users_month.groupby(combined_new_users_month.index).agg({'new_users': lambda sets: set().union(*sets)})
    combined_new_users_month['total_new_users'] = combined_new_users_month['new_users'].apply(len)

    # 🔹 Utilisateurs actifs - Invités
    df_guests_active = pd.DataFrame(guest_active_users)
    if not df_guests_active.empty:
        df_guests_active['month_start'] = df_guests_active['_id'].apply(lambda x: datetime(x['year'], x['month'], 1))
        df_guests_active['active_users'] = df_guests_active['active_users'].apply(lambda x: set(str(u) for u in x))
        df_guests_active = df_guests_active[['month_start', 'active_users']].set_index('month_start')
    else:
        df_guests_active = pd.DataFrame(columns=['active_users'])

    # 🔹 Utilisateurs actifs - Connectés
    df_connected_active = pd.DataFrame(connected_active_users)
    if not df_connected_active.empty:
        df_connected_active['month_start'] = df_connected_active['_id'].apply(lambda x: datetime(x['year'], x['month'], 1))
        df_connected_active['active_users'] = df_connected_active['active_users'].apply(lambda x: set(str(u) for u in x))
        df_connected_active = df_connected_active[['month_start', 'active_users']].set_index('month_start')
    else:
        df_connected_active = pd.DataFrame(columns=['active_users'])

    # 🔀 Fusion des utilisateurs actifs
    combined_active_users_month = pd.concat([df_guests_active, df_connected_active])
    combined_active_users_month = combined_active_users_month.groupby(combined_active_users_month.index).agg({'active_users': lambda sets: set().union(*sets)})

    # ----------- 📈 CALCUL RÉTENTION MENSUELLE -----------

    # Date actuelle et début du mois en cours
    current_date = datetime.now()
    current_month_start = datetime(current_date.year, current_date.month, 1)

    # Plage complète de mois (exclut le mois courant)
    all_months_range = pd.date_range(start=date_start, end=current_month_start - timedelta(days=1), freq='MS')

    filtered_combined_new_users = combined_new_users_month[combined_new_users_month.index < current_month_start]
    all_cohort_months = set(all_months_range) | set(filtered_combined_new_users.index)

    # Initialisation du dict
    month_retention = {idx: {"+0": 0} for idx in all_cohort_months}

    # Remplissage des +0 et suivants
    for index, row in filtered_combined_new_users.iterrows():
        new_user_set = row['new_users']
        month_retention[index]["+0"] = len(new_user_set)
        if not new_user_set:
            continue

        future_months = combined_active_users_month.loc[
            (combined_active_users_month.index > index) &
            (combined_active_users_month.index < current_month_start)
        ]

        for future_index, future_row in future_months.iterrows():
            month_diff = (future_index.year - index.year) * 12 + (future_index.month - index.month)
            retained_users = len(new_user_set.intersection(future_row['active_users']))
            month_retention[index][f"+{month_diff}"] = retained_users

    # Conversion en DataFrame
    df_retention_month = pd.DataFrame.from_dict(month_retention, orient='index')

    # Compléter colonnes
    max_offset = max((current_month_start - timedelta(days=1) - idx).days // 30 for idx in df_retention_month.index)

    for index, row in df_retention_month.iterrows():
        for month_diff in range(max_offset + 1):
            col_name = f"+{month_diff}"
            future_month = index + pd.DateOffset(months=month_diff)
            if future_month >= current_month_start:
                df_retention_month.at[index, col_name] = None
            elif pd.isna(row.get(col_name, None)):
                df_retention_month.at[index, col_name] = 0

    # Totaux
    all_months_df = pd.DataFrame(index=sorted(all_cohort_months))
    all_months_df['total_new_users'] = 0
    for idx in filtered_combined_new_users.index:
        if idx in all_months_df.index:
            all_months_df.loc[idx, 'total_new_users'] = len(filtered_combined_new_users.loc[idx, 'new_users'])

    # Fusion avec retention
    df_numeric_month = all_months_df.merge(df_retention_month, left_index=True, right_index=True, how='left')

    # Organisation colonnes
    retention_cols = sorted([col for col in df_numeric_month.columns if col.startswith("+")], key=lambda x: int(x[1:]))
    df_numeric_month = df_numeric_month[['total_new_users'] + retention_cols]

    # Pourcentages
    df_percentage_month = df_numeric_month.copy()
    for col in retention_cols:
        if col != "+0":
            mask = (df_percentage_month["+0"] > 0) & (df_percentage_month[col].notna())
            df_percentage_month.loc[mask, col] = (
                df_percentage_month.loc[mask, col] / df_percentage_month.loc[mask, "+0"] * 100
            ).round(1)
    df_percentage_month["+0"] = df_percentage_month["+0"].apply(lambda x: 100 if x > 0 else 0)

    def apply_red_gradient(val):
        if pd.isna(val):
            return 'background-color: #f0f0f0; color: #f0f0f0;'
        intensity = int(255 * ((1 - val / 100) ** 3))
        return f'background-color: rgba(255, {intensity}, {intensity}, 1); color: black;'

    # TABLEAUX
    st.header("📅 Tableau des cohortes mensuelles - Tous (valeurs)")
    st.dataframe(df_numeric_month)

    st.subheader("📊 Tableau des cohortes mensuelles - Tous (%)")
    st.dataframe(
        df_percentage_month.style.applymap(
            apply_red_gradient,
            subset=[col for col in df_percentage_month.columns if col.startswith("+")]
        )
    )

    # COURBE
    st.subheader("📈 Courbe de rétention mensuelle - Tous")

    df_plot = df_percentage_month.copy()
    fig = go.Figure()
    colormap = cm.get_cmap('tab20c', len(df_plot))

    for i, (idx, row) in enumerate(df_plot.iterrows()):
        valid_values = row[[col for col in row.index if col.startswith("+")]].dropna()
        if "+0" not in valid_values.index:
            continue
        rgba = colormap(i / len(df_plot))
        color = f'rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]})'
        fig.add_trace(go.Scatter(
            x=valid_values.index,
            y=valid_values.values,
            mode='lines',
            name=f"Cohorte {idx.strftime('%Y-%m')}",
            line=dict(width=2, color=color),
            hoverinfo='x+y',
            opacity=0.8
        ))

    # Moyenne
    average_curve = df_plot[[col for col in df_plot.columns if col.startswith("+")]].mean(axis=0, skipna=True)
    fig.add_trace(go.Scatter(
        x=average_curve.index,
        y=average_curve.values,
        mode='lines',
        name='Moyenne',
        line=dict(width=3, color='black'),
        opacity=1.0
    ))

    fig.update_layout(
        title="📊 Rétention mensuelle (%) - Tous (Connectés + Invités)",
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
        ),
        height=500
    )

    st.plotly_chart(fig)


# ----------------------------------------------------------
# Partie Acquisition Hebdomadaire des Utilisateurs Connectés 
# ----------------------------------------------------------

if page == "Acquisition" and toggle_view == "Hebdomadaire" and user_type == "Utilisateurs Connectés":
    
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

    st.title("📊 Partie Acquisition")
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

# ----------------------------------------------------------
# Partie Acquisition Mensuelle des Utilisateurs Connectés 
# ----------------------------------------------------------

elif page == "Acquisition" and toggle_view == "Mensuel" and user_type == "Utilisateurs Connectés":

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

    st.title("📊 Partie Acquisition")
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

# ------------------------------------------------------
# Partie Acquisition Hebdomadaire Invités
# ------------------------------------------------------

elif page == "Acquisition" and toggle_view == "Hebdomadaire" and user_type == "Invités":
    st.title("Partie Acquisition - Utilisateurs Invités")
    pipeline_new_guests_per_week = [
        {
        "$match": {
            "createdAt": { "$gte": date_start, "$lte": date_end },
            "userId": { "$exists": True },
            "$or": [
                { "userId": None },
                { "userId": "" },
                { "userId": { "$regex": "^GUEST_", "$options": "i" } }
            ]
        }

        },
        {
            "$addFields": {
                "guestKey": {
                    "$cond": [
                        { "$eq": ["$userId", None] },
                        { "$toString": "$_id" }, 
                        "$userId"                
                    ]
                }
            }
        },
        {
            "$group": {
                "_id": {
                    "year": { "$isoWeekYear": "$createdAt" },
                    "week": { "$isoWeek": "$createdAt" }
                },
                "guests": { "$addToSet": "$guestKey" }
            }
        },
        {
            "$project": {
                "_id": 1,
                "guest_count": { "$size": "$guests" }
            }
        },
        { "$sort": { "_id.year": 1, "_id.week": 1 } }
    ]


    # 📌 Exécuter la requête
    cursor_new_guests_per_week = orders_collection.aggregate(pipeline_new_guests_per_week)
    data_new_guests_per_week = list(cursor_new_guests_per_week)

    if not data_new_guests_per_week:
        st.error("❌ Aucune donnée trouvée pour les nouveaux invités par semaine !")
        st.stop()

    # 📌 Transformation en DataFrame
    df_new_guests_per_week = pd.DataFrame(data_new_guests_per_week)

    # 📌 Extraction des années et semaines
    df_new_guests_per_week['year'] = df_new_guests_per_week['_id'].apply(lambda x: x['year'])
    df_new_guests_per_week['week'] = df_new_guests_per_week['_id'].apply(lambda x: x['week'])

    # 📌 Générer la colonne de début de semaine
    df_new_guests_per_week['week_start'] = df_new_guests_per_week.apply(
        lambda x: datetime.fromisocalendar(x['year'], x['week'], 1), axis=1
    )

    # 📌 Trier et indexer
    df_new_guests_per_week = df_new_guests_per_week.sort_values(by='week_start').set_index('week_start')

    # 📌 Exclure la semaine actuelle
    today = datetime.now()
    current_week = today.isocalendar()[1]
    current_year = today.year

    df_new_guests_per_week = df_new_guests_per_week[
        ~((df_new_guests_per_week['year'] == current_year) & (df_new_guests_per_week['week'] == current_week))
    ]

    # 📌 Affichage tableau
    st.subheader("📅 Nombre de nouveaux utilisateurs invités par semaine")
    st.dataframe(df_new_guests_per_week['guest_count'])

    # 📌 Courbe Plotly
    fig = px.line(
        df_new_guests_per_week,
        x=df_new_guests_per_week.index,
        y="guest_count",
        title="📈 Évolution des nouveaux utilisateurs invités par semaine",
        labels={"week_start": "Semaine", "guest_count": "Nouveaux invités"},
        markers=True
    )

    st.subheader("📈 Évolution des nouveaux invités par semaine")
    st.plotly_chart(fig)



    current_date = datetime.now()
    current_week_start = datetime.fromisocalendar(current_date.year, current_date.isocalendar()[1], 1)

    # Générer la plage complète de semaines, en excluant la semaine actuelle
    all_weeks_range = pd.date_range(start=date_start, end=current_week_start - timedelta(days=1), freq='W-MON')

    # Filtrer les nouvelles cohortes avant la semaine en cours
    filtered_combined_new_users = combined_new_users_week[combined_new_users_week.index < current_week_start]
    all_cohort_dates = set(all_weeks_range) | set(filtered_combined_new_users.index)

    # Initialiser le dictionnaire de rétention
    week_retention = {idx: {"+0": 0} for idx in all_cohort_dates}

    # Calcul +0 et intersections
    for index, row in filtered_combined_new_users.iterrows():
        new_user_set = row['new_users']
        week_retention[index]["+0"] = len(new_user_set)
        if not new_user_set:
            continue

        future_weeks = combined_active_users_week.loc[
            (combined_active_users_week.index > index) &
            (combined_active_users_week.index < current_week_start)
        ]

        for future_index, future_row in future_weeks.iterrows():
            week_diff = (future_index - index).days // 7
            retained_users = len(new_user_set.intersection(future_row['active_users']))
            week_retention[index][f"+{week_diff}"] = retained_users

    # Convertir en DataFrame
    df_retention_week = pd.DataFrame.from_dict(week_retention, orient='index')

    # Compléter les colonnes manquantes
    global_max = max((current_week_start - timedelta(days=7) - idx).days // 7 for idx in df_retention_week.index)

    for index, row in df_retention_week.iterrows():
        for week_diff in range(global_max + 1):
            col_name = f"+{week_diff}"
            future_week = index + timedelta(weeks=week_diff)
            if future_week >= current_week_start:
                df_retention_week.at[index, col_name] = None
            elif pd.isna(row.get(col_name, None)):
                df_retention_week.at[index, col_name] = 0

    # 🧮 Total nouveaux utilisateurs par semaine
    all_weeks_df = pd.DataFrame(index=sorted(all_cohort_dates))
    all_weeks_df['total_new_users'] = 0
    for idx in filtered_combined_new_users.index:
        if idx in all_weeks_df.index:
            all_weeks_df.loc[idx, 'total_new_users'] = len(filtered_combined_new_users.loc[idx, 'new_users'])

    # Fusion avec la rétention
    df_numeric_week = all_weeks_df.merge(df_retention_week, left_index=True, right_index=True, how='left')

    # Organisation colonnes
    retention_cols = sorted([col for col in df_numeric_week.columns if col.startswith("+")],
                            key=lambda x: int(x.replace("+", "")))
    other_cols = [col for col in df_numeric_week.columns if not col.startswith("+")]
    df_numeric_week = df_numeric_week[other_cols + retention_cols]

    # Pourcentage
    df_percentage_week = df_numeric_week.copy()
    for col in df_percentage_week.columns:
        if col.startswith("+") and col != "+0":
            mask = (df_percentage_week["+0"] > 0) & (df_percentage_week[col].notna())
            df_percentage_week.loc[mask, col] = (
                df_percentage_week.loc[mask, col] / df_percentage_week.loc[mask, "+0"] * 100
            ).round(1)
    df_percentage_week["+0"] = df_percentage_week["+0"].apply(lambda x: 100 if x > 0 else 0)

    # Gradient style
    def apply_red_gradient_with_future(val):
        if pd.isna(val):
            return 'background-color: #f0f0f0; color: #f0f0f0;'
        elif pd.notna(val):
            intensity = int(255 * ((1 - val / 100) ** 3))
            return f'background-color: rgba(255, {intensity}, {intensity}, 1); color: black;'
        return ''

    # 🔎 AFFICHAGE TABLEAUX
    st.header("📅 Tableau des cohortes hebdomadaires - Tous (valeurs)")
    st.dataframe(df_numeric_week)

    st.subheader("📊 Tableau des cohortes hebdomadaires - Tous (%)")
    st.dataframe(
        df_percentage_week.style.applymap(
            apply_red_gradient_with_future,
            subset=[col for col in df_percentage_week.columns if col.startswith("+")]
        )
    )

    # 📈 COURBE DE RÉTENTION
    st.subheader("📈 Courbe de rétention hebdomadaire - Tous")

    df_plot = df_percentage_week.copy()
    fig = go.Figure()
    colormap = cm.get_cmap('tab20c', len(df_plot))

    for i, (idx, row) in enumerate(df_plot.iterrows()):
        valid_values = row[[col for col in row.index if col.startswith("+")]].dropna()
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

    # Moyenne
    average_curve = df_plot[[col for col in df_plot.columns if col.startswith("+")]].mean(axis=0, skipna=True)
    fig.add_trace(go.Scatter(
        x=average_curve.index,
        y=average_curve.values,
        mode='lines',
        name='Moyenne par +x',
        line=dict(width=3, color='black'),
        opacity=1.0
    ))

    fig.update_layout(
        title="📊 Rétention hebdomadaire (%) - Utilisateurs Connectés + Invités",
        xaxis_title="Semaine après le premier achat",
        yaxis_title="Pourcentage de rétention",
        template="plotly_white",
        xaxis=dict(
            tickmode='array',
            tickvals=[f'+{i}' for i in range(len(average_curve))]
        ),
        yaxis=dict(
            tickformat=".1f",
            range=[0, 110]
        ),
        height=500
    )

    st.plotly_chart(fig)


# ------------------------------------------------------
# Partie Acquisition Mensuelle Invités
# ------------------------------------------------------

if page == "Acquisition" and toggle_view == "Mensuel" and user_type == "Invités":
    st.title("Partie Acquisition - Utilisateurs Invités (Mensuel)")

    # 📌 Pipeline MongoDB mensuel
    pipeline_new_guests_per_month = [
        {
            "$match": {
                "createdAt": {"$gte": date_start, "$lte": date_end},
                "$or": [
                    {"userId": None},
                    {"userId": {"$regex": "^GUEST_"}}
                ]
            }
        },
        {
            "$addFields": {
                "guestKey": {
                    "$cond": [
                        { "$eq": ["$userId", None] },
                        { "$toString": "$_id" },
                        "$userId"
                    ]
                }
            }
        },
        {
            "$group": {
                "_id": {
                    "year": { "$year": "$createdAt" },
                    "month": { "$month": "$createdAt" }
                },
                "guests": { "$addToSet": "$guestKey" }
            }
        },
        {
            "$project": {
                "_id": 1,
                "guest_count": { "$size": "$guests" }
            }
        },
        { "$sort": { "_id.year": 1, "_id.month": 1 } }
    ]

    # 📌 Exécuter la requête
    cursor_new_guests_per_month = orders_collection.aggregate(pipeline_new_guests_per_month)
    data_new_guests_per_month = list(cursor_new_guests_per_month)

    if not data_new_guests_per_month:
        st.error("❌ Aucune donnée trouvée pour les nouveaux invités par mois !")
        st.stop()

    # 📌 Transformation en DataFrame
    df_new_guests_per_month = pd.DataFrame(data_new_guests_per_month)

    # 📌 Extraire les champs
    df_new_guests_per_month['year'] = df_new_guests_per_month['_id'].apply(lambda x: x['year'])
    df_new_guests_per_month['month'] = df_new_guests_per_month['_id'].apply(lambda x: x['month'])

    # 📌 Créer la date de début de mois
    df_new_guests_per_month['month_start'] = df_new_guests_per_month.apply(
        lambda x: datetime(x['year'], x['month'], 1), axis=1
    )
    # 📌 Trier et indexer
    df_new_guests_per_month = df_new_guests_per_month.sort_values(by='month_start').set_index('month_start')

    # 📌 Appliquer la date de début manuelle
    start_date = datetime(2025, 3, 1)
    df_new_guests_per_month = df_new_guests_per_month[df_new_guests_per_month.index >= start_date]

    # 📌 Exclure le mois en cours
    today = datetime.now()
    df_new_guests_per_month = df_new_guests_per_month[
        ~((df_new_guests_per_month['year'] == today.year) & (df_new_guests_per_month['month'] == today.month))
    ]


    # 📌 Affichage tableau
    st.subheader("📅 Nombre de nouveaux utilisateurs invités par mois")
    st.dataframe(df_new_guests_per_month['guest_count'])

    # 📌 Graphique Plotly
    fig = px.line(
        df_new_guests_per_month,
        x=df_new_guests_per_month.index,
        y="guest_count",
        title="📈 Évolution des nouveaux utilisateurs invités par mois",
        labels={"month_start": "Mois", "guest_count": "Nouveaux invités"},
        markers=True
    )

    st.subheader("📈 Évolution mensuelle des nouveaux invités")
    st.plotly_chart(fig)

# ------------------------------------------------------
# Partie Acquisition Tous les Utilisateurs
# ------------------------------------------------------

elif page == "Acquisition" and toggle_view == "Hebdomadaire" and user_type == "Tous":
    st.title("Partie Acquisition - Tous les Utilisateurs (Hebdomadaire)")
    # Recalculer les deux DataFrames si on est en mode "Tous"

    # Pipeline utilisateurs connectés
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
    today = datetime.now()
    current_week = today.isocalendar()[1]
    current_year = today.year

    data_new_users_per_week = list(users_collection.aggregate(pipeline_new_users_per_week))
    df_new_users_per_week = pd.DataFrame(data_new_users_per_week)
    df_new_users_per_week['year'] = df_new_users_per_week['_id'].apply(lambda x: x['year'])
    df_new_users_per_week['week'] = df_new_users_per_week['_id'].apply(lambda x: x['week'])
    df_new_users_per_week['week_start'] = df_new_users_per_week.apply(
        lambda x: datetime.fromisocalendar(x['year'], x['week'], 1), axis=1
    )
    df_new_users_per_week = df_new_users_per_week.sort_values(by='week_start').set_index('week_start')
    df_new_users_per_week = df_new_users_per_week[
        ~((df_new_users_per_week['year'] == current_year) & (df_new_users_per_week['week'] == current_week))
    ]

    # Pipeline invités
    pipeline_new_guests_per_week = [
        {
            "$match": {
                "createdAt": {"$gte": date_start, "$lte": date_end},
                "$or": [
                    {"userId": None},
                    {"userId": {"$regex": "^GUEST_"}}
                ]
            }
        },
        {
            "$addFields": {
                "guestKey": {
                    "$cond": [
                        { "$eq": ["$userId", None] },
                        { "$toString": "$_id" },
                        "$userId"
                    ]
                }
            }
        },
        {
            "$group": {
                "_id": {
                    "year": { "$isoWeekYear": "$createdAt" },
                    "week": { "$isoWeek": "$createdAt" }
                },
                "guests": { "$addToSet": "$guestKey" }
            }
        },
        {
            "$project": {
                "_id": 1,
                "guest_count": { "$size": "$guests" }
            }
        },
        { "$sort": { "_id.year": 1, "_id.week": 1 } }
    ]
    data_new_guests_per_week = list(orders_collection.aggregate(pipeline_new_guests_per_week))
    df_new_guests_per_week = pd.DataFrame(data_new_guests_per_week)
    df_new_guests_per_week['year'] = df_new_guests_per_week['_id'].apply(lambda x: x['year'])
    df_new_guests_per_week['week'] = df_new_guests_per_week['_id'].apply(lambda x: x['week'])
    df_new_guests_per_week['week_start'] = df_new_guests_per_week.apply(
        lambda x: datetime.fromisocalendar(x['year'], x['week'], 1), axis=1
    )
    df_new_guests_per_week = df_new_guests_per_week.sort_values(by='week_start').set_index('week_start')
    df_new_guests_per_week = df_new_guests_per_week[
        ~((df_new_guests_per_week['year'] == current_year) & (df_new_guests_per_week['week'] == current_week))
    ]


    # 📌 Assurer les deux DataFrames existent et sont bien indexés
    if 'df_new_users_per_week' not in locals() or 'df_new_guests_per_week' not in locals():
        st.error("❌ Les données des utilisateurs connectés ou invités sont manquantes.")
        st.stop()

    # 📌 Aligner les deux DataFrames
    df_total_users_per_week = pd.DataFrame(index=df_new_users_per_week.index.union(df_new_guests_per_week.index))
    df_total_users_per_week['connectés'] = df_new_users_per_week['new_users']
    df_total_users_per_week['invités'] = df_new_guests_per_week['guest_count']

    df_total_users_per_week = df_total_users_per_week.fillna(0)
    df_total_users_per_week['total'] = df_total_users_per_week['connectés'] + df_total_users_per_week['invités']

    # 📌 Affichage tableau
    st.subheader("📅 Nombre total de nouveaux utilisateurs (connectés + invités) par semaine")
    st.dataframe(df_total_users_per_week[['connectés', 'invités', 'total']])

    # 📈 Courbe Plotly
    fig = px.line(
        df_total_users_per_week,
        x=df_total_users_per_week.index,
        y=['connectés', 'invités', 'total'],
        title="📈 Évolution hebdomadaire de l'acquisition (tous utilisateurs)",
        labels={"value": "Utilisateurs", "variable": "Type"},
        markers=True
    )
    st.subheader("📈 Évolution hebdomadaire des nouveaux utilisateurs")
    st.plotly_chart(fig)



# ------------------------------------------------------
# Partie Active Users - Hebdomadaire Utilisateurs Connectés
# ------------------------------------------------------

if page == "Active Users" and toggle_view == "Hebdomadaire" and user_type == "Utilisateurs Connectés":
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

    # 📌 Exclure la semaine actuelle en définissant clairement son début
    today = datetime.now()
    current_week_start = datetime.fromisocalendar(today.year, today.isocalendar()[1], 1)

    # 📌 Regénérer la liste des semaines valides (excluant la semaine actuelle)
    all_weeks = pd.date_range(start=date_start, end=current_week_start - timedelta(days=1), freq='W-MON')
    df_all_weeks_unique_users = df_all_weeks_unique_users[df_all_weeks_unique_users.index.isin(all_weeks)]


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

# ------------------------------------------------------
# Partie Active Users - Mensuel Utilisateurs Connectés 
# ------------------------------------------------------

elif page == "Active Users" and toggle_view == "Mensuel" and user_type == "Utilisateurs Connectés":
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


# ------------------------------------------------------
# Partie Active Users - Hebdomadaire Invités
# ------------------------------------------------------

if page == "Active Users" and toggle_view == "Hebdomadaire" and user_type == "Invités":
    st.title("Partie Weekly Active Guests")

    # 🔹 Si besoin, convertir en ObjectId
    if isinstance(store_filter, str):
        store_filter = ObjectId(store_filter)

    # 🔹 Construire le filtre MongoDB
    base_filter_guests = {
        "isPaid": True,
        "paidAt": {"$gte": date_start, "$lte": date_end},
        "storeId": store_filter,
        "userId": {"$exists": True},
        "$or": [
            {"userId": None},
            {"userId": ""},
            {"userId": {"$regex": "^GUEST_", "$options": "i"}}
        ]
    }

    # 🔹 Si un mode de paiement est sélectionné, l'ajouter au filtre
    if selected_payment_method != "Tous":
        variants = payment_variants.get(selected_payment_method, [selected_payment_method])
        base_filter_guests["paymentMethod"] = {"$in": variants}

    # 🔹 Pipeline MongoDB
    pipeline_active_guests_per_week = [
        {"$match": base_filter_guests},
        {"$addFields": {
            "guestKey": {
                "$cond": [
                    {"$or": [
                        {"$eq": ["$userId", None]},
                        {"$eq": ["$userId", ""]}
                    ]},
                    {"$toString": "$_id"},
                    "$userId"
                ]
            }
        }},
        {"$group": {
            "_id": {
                "year": {"$isoWeekYear": "$paidAt"},
                "week": {"$isoWeek": "$paidAt"}
            },
            "unique_guests": {"$addToSet": "$guestKey"}
        }},
        {"$project": {
            "_id": 1,
            "total_unique_users": {"$size": "$unique_guests"}
        }},
        {"$sort": {"_id.year": 1, "_id.week": 1}}
    ]

    # 🔹 Exécuter la requête MongoDB
    cursor = orders_collection.aggregate(pipeline_active_guests_per_week)
    data = list(cursor)

    # 🔹 Vérification des données
    if not data:
        st.error("❌ Aucune donnée trouvée pour les utilisateurs invités actifs par semaine !")
        st.stop()

    # 🔹 Transformation en DataFrame
    df = pd.DataFrame(data)
    df['year'] = df['_id'].apply(lambda x: x['year'])
    df['week'] = df['_id'].apply(lambda x: x['week'])
    df['week_start'] = df.apply(lambda x: datetime.fromisocalendar(x['year'], x['week'], 1), axis=1)
    df = df.sort_values(by='week_start').set_index('week_start')

    # 🔹 Générer la plage des semaines (hors semaine actuelle)
    start_date = datetime(2025, 3, 17)
    today = datetime.now()
    current_week_start = datetime.fromisocalendar(today.year, today.isocalendar()[1], 1)

    all_weeks = pd.date_range(start=start_date, end=current_week_start - timedelta(days=1), freq='W-MON')
    df_all_weeks = pd.DataFrame(index=all_weeks)
    df_all_weeks['total_unique_users'] = 0
    df_all_weeks.update(df)

    # 🔹 Affichage
    st.subheader("📊 Tableau utilisateurs invités actifs par semaine")
    st.dataframe(df_all_weeks)

    # 🔹 Courbe Plotly
    fig = px.line(
        df_all_weeks,
        x=df_all_weeks.index,
        y="total_unique_users",
        title="📈 Évolution des utilisateurs invités actifs par semaine",
        labels={"week_start": "Semaine", "total_unique_users": "Utilisateurs invités actifs"},
        markers=True
    )

    st.subheader("📈 Évolution des utilisateurs invités actifs")
    st.plotly_chart(fig)


# ------------------------------------------------------
# Partie Active Users - Mensuel Invités
# ------------------------------------------------------

if page == "Active Users" and toggle_view == "Mensuel" and user_type == "Invités":
    st.title("Partie Monthly Active Guests")

    # 🔹 Si besoin, convertir le store
    if isinstance(store_filter, str):
        store_filter = ObjectId(store_filter)

    # 🔹 Base filter
    base_filter_guests = {
        "isPaid": True,
        "paidAt": {"$gte": date_start, "$lte": date_end},
        "storeId": store_filter,
        "userId": {"$exists": True},
        "$or": [
            {"userId": None},
            {"userId": ""},
            {"userId": {"$regex": "^GUEST_", "$options": "i"}}
        ]
    }

    # 🔹 Ajout du filtre sur le mode de paiement si nécessaire
    if selected_payment_method != "Tous":
        variants = payment_variants.get(selected_payment_method, [selected_payment_method])
        base_filter_guests["paymentMethod"] = {"$in": variants}

    # 🔹 Pipeline MongoDB
    pipeline_active_guests_per_month = [
        {"$match": base_filter_guests},
        {"$addFields": {
            "guestKey": {
                "$cond": [
                    {"$or": [
                        {"$eq": ["$userId", None]},
                        {"$eq": ["$userId", ""]}
                    ]},
                    {"$toString": "$_id"},
                    "$userId"
                ]
            }
        }},
        {"$group": {
            "_id": {
                "year": {"$year": "$paidAt"},
                "month": {"$month": "$paidAt"}
            },
            "unique_guests": {"$addToSet": "$guestKey"}
        }},
        {"$project": {
            "_id": 1,
            "total_unique_users": {"$size": "$unique_guests"}
        }},
        {"$sort": {"_id.year": 1, "_id.month": 1}}
    ]

    # 🔹 Exécuter la requête
    cursor = orders_collection.aggregate(pipeline_active_guests_per_month)
    data = list(cursor)

    if not data:
        st.error("❌ Aucune donnée trouvée pour les utilisateurs invités actifs par mois !")
        st.stop()

    # 🔹 Transformation en DataFrame
    df = pd.DataFrame(data)
    df['year'] = df['_id'].apply(lambda x: x['year'])
    df['month'] = df['_id'].apply(lambda x: x['month'])
    df['month_start'] = df.apply(lambda x: datetime(x['year'], x['month'], 1), axis=1)
    df = df.sort_values(by='month_start').set_index('month_start')

    # 🗓️ Forcer la date de départ à partir du 1er mars 2025
    start_date = datetime(2025, 3, 1)
    current_month_start = datetime(datetime.now().year, datetime.now().month, 1)

    # 🔹 Générer toutes les périodes mensuelles (hors mois en cours)
    all_months = pd.date_range(start=start_date, end=current_month_start - timedelta(days=1), freq='MS')
    df_all_months = pd.DataFrame(index=all_months)
    df_all_months['total_unique_users'] = 0

    # 🔹 Mettre à jour avec les données réelles
    df_all_months.update(df)

    # 🔹 Affichage
    st.subheader("📊 Tableau utilisateurs invités actifs par mois")
    st.dataframe(df_all_months)

    # 🔹 Courbe Plotly
    fig = px.line(
        df_all_months,
        x=df_all_months.index,
        y="total_unique_users",
        title="📈 Évolution des utilisateurs invités actifs par mois",
        labels={"month_start": "Mois", "total_unique_users": "Utilisateurs invités actifs"},
        markers=True
    )

    st.subheader("📈 Évolution des utilisateurs invités actifs")
    st.plotly_chart(fig)

# ------------------------------------------------------
# Partie Active Users - Hebdomadaire Tous
# ------------------------------------------------------

if page == "Active Users" and toggle_view == "Hebdomadaire" and user_type == "Tous":
    st.title("Partie Weekly Active Users - Tous (Connectés + Invités)")

    # ----------------- CONNECTÉS -----------------
    match_filter_connected = {
        "receipt.isPaid": True,
        "receipt.paidAt": {"$gte": date_start, "$lte": date_end},
        "receipt.storeId": store_filter
    }

    if selected_payment_method != "Tous":
        variants = payment_variants.get(selected_payment_method, [selected_payment_method])
        match_filter_connected["receipt.paymentMethod"] = {"$in": variants}

    pipeline_active_connected = [
        {"$unwind": "$receipt"},
        {"$match": match_filter_connected},
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

    data_connected = list(users_collection.aggregate(pipeline_active_connected))
    df_connected = pd.DataFrame(data_connected)
    if not df_connected.empty:
        df_connected['year'] = df_connected['_id'].apply(lambda x: x['year'])
        df_connected['week'] = df_connected['_id'].apply(lambda x: x['week'])
        df_connected['week_start'] = df_connected.apply(lambda x: datetime.fromisocalendar(x['year'], x['week'], 1), axis=1)
        df_connected = df_connected[['week_start', 'total_unique_users']].set_index('week_start')
    else:
        df_connected = pd.DataFrame(columns=['total_unique_users'])

    # ----------------- INVITÉS -----------------
    base_filter_guests = {
        "isPaid": True,
        "paidAt": {"$gte": date_start, "$lte": date_end},
        "storeId": store_filter,
        "userId": {"$exists": True},
        "$or": [
            {"userId": None},
            {"userId": ""},
            {"userId": {"$regex": "^GUEST_", "$options": "i"}}
        ]
    }

    if selected_payment_method != "Tous":
        base_filter_guests["paymentMethod"] = {"$in": variants}

    pipeline_active_guests = [
        {"$match": base_filter_guests},
        {"$addFields": {
            "guestKey": {
                "$cond": [
                    {"$or": [
                        {"$eq": ["$userId", None]},
                        {"$eq": ["$userId", ""]}
                    ]},
                    {"$toString": "$_id"},
                    "$userId"
                ]
            }
        }},
        {"$group": {
            "_id": {
                "year": {"$isoWeekYear": "$paidAt"},
                "week": {"$isoWeek": "$paidAt"}
            },
            "unique_guests": {"$addToSet": "$guestKey"}
        }},
        {"$project": {
            "_id": 1,
            "total_unique_users": {"$size": "$unique_guests"}
        }},
        {"$sort": {"_id.year": 1, "_id.week": 1}}
    ]

    data_guests = list(orders_collection.aggregate(pipeline_active_guests))
    df_guests = pd.DataFrame(data_guests)
    if not df_guests.empty:
        df_guests['year'] = df_guests['_id'].apply(lambda x: x['year'])
        df_guests['week'] = df_guests['_id'].apply(lambda x: x['week'])
        df_guests['week_start'] = df_guests.apply(lambda x: datetime.fromisocalendar(x['year'], x['week'], 1), axis=1)
        df_guests = df_guests[['week_start', 'total_unique_users']].set_index('week_start')
    else:
        df_guests = pd.DataFrame(columns=['total_unique_users'])

    # ----------------- FUSION CONNECTÉS + INVITÉS -----------------
    df_all = pd.concat([df_connected, df_guests])
    df_all = df_all.groupby(df_all.index).sum()

    # ----------------- COMPLETION ET AFFICHAGE -----------------
    today = datetime.now()
    current_week_start = datetime.fromisocalendar(today.year, today.isocalendar()[1], 1)
    all_weeks = pd.date_range(start=date_start, end=current_week_start - timedelta(days=1), freq='W-MON')

    df_all_weeks = pd.DataFrame(index=all_weeks)
    df_all_weeks['total_unique_users'] = 0
    df_all_weeks.update(df_all)

    # 🔹 Affichage tableau
    st.subheader("📊 Tableau des utilisateurs actifs par semaine - Tous")
    st.dataframe(df_all_weeks)

    # 🔹 Courbe
    fig = px.line(
        df_all_weeks,
        x=df_all_weeks.index,
        y="total_unique_users",
        title="📈 Évolution des utilisateurs actifs par semaine (Connectés + Invités)",
        labels={"week_start": "Semaine", "total_unique_users": "Utilisateurs actifs"},
        markers=True
    )
    st.subheader("📈 Évolution des utilisateurs actifs")
    st.plotly_chart(fig)

# ------------------------------------------------------
# Partie Active Users - Mensuel Tous
# ------------------------------------------------------

if page == "Active Users" and toggle_view == "Mensuel" and user_type == "Tous":
    st.title("Partie Monthly Active Users - Tous (Connectés + Invités)")

    # ---------------- CONNECTÉS ----------------
    match_filter_connected = {
        "receipt.isPaid": True,
        "receipt.paidAt": {"$gte": date_start, "$lte": date_end},
        "receipt.storeId": store_filter
    }

    if selected_payment_method != "Tous":
        variants = payment_variants.get(selected_payment_method, [selected_payment_method])
        match_filter_connected["receipt.paymentMethod"] = {"$in": variants}

    pipeline_connected = [
        {"$unwind": "$receipt"},
        {"$match": match_filter_connected},
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

    data_connected = list(users_collection.aggregate(pipeline_connected))
    df_connected = pd.DataFrame(data_connected)
    if not df_connected.empty:
        df_connected['year'] = df_connected['_id'].apply(lambda x: x['year'])
        df_connected['month'] = df_connected['_id'].apply(lambda x: x['month'])
        df_connected['month_start'] = df_connected.apply(lambda x: datetime(x['year'], x['month'], 1), axis=1)
        df_connected = df_connected[['month_start', 'total_unique_users']].set_index('month_start')
    else:
        df_connected = pd.DataFrame(columns=['total_unique_users'])

    # ---------------- INVITÉS ----------------
    base_filter_guests = {
        "isPaid": True,
        "paidAt": {"$gte": date_start, "$lte": date_end},
        "storeId": store_filter,
        "userId": {"$exists": True},
        "$or": [
            {"userId": None},
            {"userId": ""},
            {"userId": {"$regex": "^GUEST_", "$options": "i"}}
        ]
    }

    if selected_payment_method != "Tous":
        base_filter_guests["paymentMethod"] = {"$in": variants}

    pipeline_guests = [
        {"$match": base_filter_guests},
        {"$addFields": {
            "guestKey": {
                "$cond": [
                    {"$or": [
                        {"$eq": ["$userId", None]},
                        {"$eq": ["$userId", ""]}
                    ]},
                    {"$toString": "$_id"},
                    "$userId"
                ]
            }
        }},
        {"$group": {
            "_id": {
                "year": {"$year": "$paidAt"},
                "month": {"$month": "$paidAt"}
            },
            "unique_guests": {"$addToSet": "$guestKey"}
        }},
        {"$project": {
            "_id": 1,
            "total_unique_users": {"$size": "$unique_guests"}
        }},
        {"$sort": {"_id.year": 1, "_id.month": 1}}
    ]

    data_guests = list(orders_collection.aggregate(pipeline_guests))
    df_guests = pd.DataFrame(data_guests)
    if not df_guests.empty:
        df_guests['year'] = df_guests['_id'].apply(lambda x: x['year'])
        df_guests['month'] = df_guests['_id'].apply(lambda x: x['month'])
        df_guests['month_start'] = df_guests.apply(lambda x: datetime(x['year'], x['month'], 1), axis=1)
        df_guests = df_guests[['month_start', 'total_unique_users']].set_index('month_start')
    else:
        df_guests = pd.DataFrame(columns=['total_unique_users'])

    # ---------------- FUSION CONNECTÉS + INVITÉS ----------------
    df_all = pd.concat([df_connected, df_guests])
    df_all = df_all.groupby(df_all.index).sum()

    # ---------------- COMPLETION & EXCLUSION MOIS COURANT ----------------
    today = datetime.now()
    current_month_start = datetime(today.year, today.month, 1)

    # Choix d’un start_date cohérent (sinon on peut reprendre celui de la base)
    start_date = min(
        df_all.index.min() if not df_all.empty else current_month_start,
        datetime(2025, 3, 1)
    )

    all_months = pd.date_range(start=start_date, end=current_month_start - timedelta(days=1), freq='MS')
    df_all_months = pd.DataFrame(index=all_months)
    df_all_months['total_unique_users'] = 0

    df_all_months.update(df_all)

    # ---------------- AFFICHAGE ----------------
    st.subheader("📊 Tableau utilisateurs actifs mensuels - Tous")
    st.dataframe(df_all_months)

    fig = px.line(
        df_all_months,
        x=df_all_months.index,
        y="total_unique_users",
        title="📈 Évolution des utilisateurs actifs par mois (Connectés + Invités)",
        labels={"month_start": "Mois", "total_unique_users": "Utilisateurs actifs"},
        markers=True
    )

    st.subheader("📈 Évolution des utilisateurs actifs mensuels")
    st.plotly_chart(fig)


# ------------------------------------------------------
# Partie Bug Report User type Tous
# ------------------------------------------------------

if page == "Bug Report" and user_type == "Tous":
    st.title("Partie Bug Report")
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



# ------------------------------------------------------
# Partie Bug Report User type Utilisateurs Connectés
# ------------------------------------------------------

elif page == 'Bug Report' and user_type == 'Utilisateurs Connectés':
    today = datetime.now()
    current_week_start = (today - timedelta(days=today.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    current_week_end = current_week_start + timedelta(days=6, hours=23, minutes=59, seconds=59)

    user_filter = {}
    connected_user_ids = users_collection.distinct("_id")
    connected_user_ids_str = [str(uid) for uid in connected_user_ids]
    user_filter = {"userId": {"$in": connected_user_ids_str}}


    # 🔹 Récupération des paniers abandonnés
    non_finalized_carts_users = list(orders_collection.find({
        'isPaid': False,
        'createdAt': {'$gte': current_week_start, '$lte': current_week_end},
        'scanItems': {'$exists': True, '$ne': []},
        **user_filter
    }))

    # 🔹 Récupération des paniers finalisés
    finalized_carts_users = list(orders_collection.find({
        'isPaid': True,
        'paidAt': {'$gte': current_week_start, '$lte': current_week_end},
        **user_filter
    }))
        # 🔹 Nombre total de paniers abandonnés et finalisés
    total_non_finalized = len(non_finalized_carts_users)
    total_finalized = len(finalized_carts_users)

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
    non_finalized_counts_users = defaultdict(int)
    for cart in non_finalized_carts_users:
        store_id = cart.get('storeId')
        store_name = store_mapping.get(str(store_id), "Inconnu") 
        non_finalized_counts_users[store_name] += 1

    # 🔹 Comptage des paniers finalisés par magasin
    finalized_counts_users = defaultdict(int)
    for cart in finalized_carts_users:
        store_id = cart.get('storeId')
        store_name = store_mapping.get(str(store_id), "Inconnu") 
        finalized_counts_users[store_name] += 1

    # 🔹 Conversion en DataFrame et tri des résultats
    non_finalized_df_users = pd.DataFrame(list(non_finalized_counts_users.items()), columns=['Magasin', 'Paniers Abandonnés'])
    non_finalized_df_users = non_finalized_df_users.sort_values(by='Paniers Abandonnés', ascending=False).reset_index(drop=True)

    finalized_df_users = pd.DataFrame(list(finalized_counts_users.items()), columns=['Magasin', 'Paniers Finalisés'])
    finalized_df_users = finalized_df_users.sort_values(by='Paniers Finalisés', ascending=False).reset_index(drop=True)

    tab1, tab2 = st.tabs(["✅ Paniers Finalisés", "🛒 Paniers Abandonnés"])

    with tab1:
        st.subheader("✅ Paniers finalisés de la semaine")
        st.write(f"Nombre total de paniers finalisés : {total_finalized}")
        st.dataframe(finalized_df_users)  # Tableau des paniers finalisés

    with tab2:
        st.subheader("🛒 Paniers abandonnés de la semaine")
        st.write(f"Nombre total de paniers abandonnés : {total_non_finalized}")
        st.dataframe(non_finalized_df_users)  # Tableau des paniers abandonnés

# ------------------------------------------------------
# Partie Bug Report User type Invités
# ------------------------------------------------------

elif page == 'Bug Report' and user_type == 'Invités':
    today = datetime.now()
    current_week_start = (today - timedelta(days=today.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    current_week_end = current_week_start + timedelta(days=6, hours=23, minutes=59, seconds=59)

    # 🔹 Récupération des paniers abandonnés (GUEST_)
    non_finalized_carts_guests = list(orders_collection.find({
        'isPaid': False,
        'createdAt': {'$gte': current_week_start, '$lte': current_week_end},
        'scanItems': {'$exists': True, '$ne': []},
        'userId': {'$regex': '^GUEST_'}
    }))

    # 🔹 Récupération des paniers finalisés (userId == null = invité)
    finalized_carts_guests = list(orders_collection.find({
        'isPaid': True,
        'paidAt': {'$gte': current_week_start, '$lte': current_week_end},
        '$or': [
            {'userId': None},
            {'userId': {'$regex': '^GUEST_'}},
            {'userId': {'$exists': False}}
        ]
    }))


    # 🔹 Nombre total de paniers abandonnés et finalisés
    total_non_finalized_guests = len(non_finalized_carts_guests)
    total_finalized_guests = len(finalized_carts_guests)

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
    non_finalized_counts_guests = defaultdict(int)
    for cart in non_finalized_carts_guests:
        store_id = cart.get('storeId')
        store_name = store_mapping.get(str(store_id), "Inconnu") 
        non_finalized_counts_guests[store_name] += 1

    # 🔹 Comptage des paniers finalisés par magasin
    finalized_counts_guests = defaultdict(int)
    for cart in finalized_carts_guests:
        store_id = cart.get('storeId')
        store_name = store_mapping.get(str(store_id), "Inconnu") 
        finalized_counts_guests[store_name] += 1

    # 🔹 Conversion en DataFrame et tri des résultats
    non_finalized_df_guests = pd.DataFrame(list(non_finalized_counts_guests.items()), columns=['Magasin', 'Paniers Abandonnés'])
    non_finalized_df_guests = non_finalized_df_guests.sort_values(by='Paniers Abandonnés', ascending=False).reset_index(drop=True)

    finalized_df_guests = pd.DataFrame(list(finalized_counts_guests.items()), columns=['Magasin', 'Paniers Finalisés'])
    finalized_df_guests = finalized_df_guests.sort_values(by='Paniers Finalisés', ascending=False).reset_index(drop=True)

    # 📌 Affichage Streamlit
    tab1, tab2 = st.tabs(["✅ Paniers Finalisés", "🛒 Paniers Abandonnés (Invités)"])

    with tab1:
        st.subheader("✅ Paniers finalisés de la semaine")
        st.write(f"Nombre total de paniers finalisés : {total_finalized_guests}")
        st.dataframe(finalized_df_guests)

    with tab2:
        st.subheader("🛒 Paniers abandonnés de la semaine (Invités)")
        st.write(f"Nombre total de paniers abandonnés : {total_non_finalized_guests}")
        st.dataframe(non_finalized_df_guests)