import random
import string
import pandas as pd
from nettoyage_donnees  import generer_phrase
import logging
from pathlib import Path  
from sklearn.model_selection import train_test_split

def generer_datasets_test(df, test_size, perturb_values, dossier_data):
    # 1. Split initial
    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df['drug']
    )
    # 2. Préparation des proportions
    total_lignes = len(df_test)
    nb_erreurs_total = (int(total_lignes * 0.30) // 2) * 2
    nb_erreurs_par_type = nb_erreurs_total // 2
    nb_propres = total_lignes - nb_erreurs_total

    # Isolation du groupe propre
    df_test_propre = df_test.sample(n=nb_propres, random_state=42).copy()
    df_test_propre['error_types'] = "none"
    df_test_propre['nb_errors'] = 0
    df_source_erreurs = df_test.drop(df_test_propre.index)

    # Création du sous-dossier
    chemin_test = Path(dossier_data) / "data_test"
    chemin_test.mkdir(parents=True, exist_ok=True)
    
    datasets_test_memoire = {}

    # 3. Boucle de génération par niveau de perturbation
    for val in perturb_values:
        # Appel de votre fonction de perturbation (supposée existante ou renommée)
        df_simples, df_multiples = generer_datasets_test_perturb(
            df_final=df_source_erreurs, 
            number_prescription=nb_erreurs_par_type, 
            error_types_to_generate=[],  
            number_perturb=2,          
            perturb_dose=val           
        )
        # Assemblage et mélange
        df_final = pd.concat([df_test_propre, df_simples, df_multiples], ignore_index=True)
        df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
              
        # Sauvegarde
        nom_fichier = chemin_test / f'df_test_perturb_{str(val).replace(".", "_")}.parquet'
        df_final.to_parquet(nom_fichier, index=False)
        datasets_test_memoire[val] = df_final

    return df_train, datasets_test_memoire
 

def perturb_drug(text):
    """
    Introduit une anomalie de type typographique dans le nom d'une molécule.
    
    La fonction applique aléatoirement l'un des trois traitements suivants :
    - 'add' : Insertion d'un caractère alphabétique.
    - 'remove' : Suppression d'un caractère existant.
    - 'replace' : Substitution d'un caractère par un autre.
    
    Args:
        text (str): Le nom original du médicament.
        
    Returns:
        str: Le nom du médicament bruité
    """
    if pd.isna(text) or text == "":
        return text
    text = str(text)
    choice = random.choice(['add', 'remove', 'replace'])
    pos = random.randint(0, len(text) - 1)
    
    if choice == 'add':
        char = random.choice(string.ascii_lowercase)
        return text[:pos] + char + text[pos:]
    elif choice == 'remove' and len(text) > 1:
        return text[:pos] + text[pos+1:]
    else: 
        char = random.choice(string.ascii_lowercase)
        return text[:pos] + char + text[pos+1:]
def perturb_route_unit(texte,list_text):
    """
    Génère une erreur de classification par substitution aléatoire.
    
    Sélectionne une valeur parmi une liste de candidats en excluant systématiquement 
    la valeur originale pour garantir l'injection d'une anomalie.
    
    Args:
        texte (str): La valeur correcte actuelle.
        list_text (list): Référentiel des valeurs possibles
        
    Returns:
        str: Une valeur différente de l'originale.
    """
    candidates = [t for t in list_text if t != texte]
    if not candidates:
        return texte 
    return random.choice(candidates)


def generate_test_dataset_simple(df_input, number_prescription, map_lower, map_upper):
    """
    Génère un dataset de test contenant des erreurs unitaires réparties par catégories.
    
    Le dataset est divisé en 5 segments égaux, chacun recevant un type d'erreur unique :
    1. Voie d'administration, 2. Nom de molécule, 3. Unité de dose, 
    4. Sous-dosage, 5. Sur-dosage.
    
    Args:
        df_input (pd.DataFrame): Source de données saines.
        number_prescription (int): Taille totale du dataset à générer.
        map_lower (dict): Dictionnaire des seuils de sous-dosage par médicament.
        map_upper (dict): Dictionnaire des seuils de sur-dosage par médicament.
        
    Returns:
        pd.DataFrame: Dataset étiqueté avec le type d'erreur injecté.
    """
    # Échantillonnage
    df_drug = df_input.sample(n=number_prescription).reset_index(drop=True).copy()
    
    # Initialisation
    df_drug['error_types'] = "none"
    df_drug['nb_errors'] = 0
    df_drug = df_drug.astype(object)
    
    split_num = number_prescription // 5
    list_routes = df_input['route'].dropna().unique().tolist()
    list_units = df_input['dose_unit_rx'].dropna().unique().tolist()

    # 1. Route
    start, end = 0, split_num
    df_drug.loc[start:end, 'route'] = df_drug.loc[start:end, 'route'].apply(lambda x: perturb_route_unit(x, list_routes))
    df_drug.loc[start:end, ['nb_errors', 'error_types']] = [1, "route"]

    # 2. Drug
    start, end = end + 1, 2 * split_num
    df_drug.loc[start:end, 'drug'] = df_drug.loc[start:end, 'drug'].apply(perturb_drug)
    df_drug.loc[start:end, ['nb_errors', 'error_types']] = [1, "drug"]

    # 3. Unité
    start, end = end + 1, 3 * split_num
    df_drug.loc[start:end, 'dose_unit_rx'] = df_drug.loc[start:end, 'dose_unit_rx'].apply(lambda x: perturb_route_unit(x, list_units))
    df_drug.loc[start:end, ['nb_errors', 'error_types']] = [1, "unit_dosage"]

    # 4. Sous-dosage 
    start, end = end + 1, 4 * split_num
    df_drug.loc[start:end, 'dose_val_rx'] = df_drug.loc[start:end].apply(
        lambda r: map_lower.get(str(r['drug']), r['dose_val_rx']), axis=1)
    df_drug.loc[start:end, ['nb_errors', 'error_types']] = [1, "sous_dosage"]

    # 5. Sur-dosage 
    start, end = end + 1, number_prescription - 1
    df_drug.loc[start:end, 'dose_val_rx'] = df_drug.loc[start:end].apply(
        lambda r: map_upper.get(str(r['drug']), r['dose_val_rx']), axis=1)
    df_drug.loc[start:end, ['nb_errors', 'error_types']] = [1, "sur_dosage"]

    return df_drug

def apply_multiple_errors(row, number_perturb, map_lower, map_upper, list_routes, list_units):
    """
    Applique séquentiellement plusieurs types d'erreurs sur une même prescription.

    Args:
        row (pd.Series): Ligne de données représentant une prescription unique.
        number_perturb (int): Nombre d'erreurs distinctes à injecter.
        map_lower (dict): Référentiel des seuils de sous-dosage par médicament.
        map_upper (dict): Référentiel des seuils de sur-dosage par médicament.
        list_routes (list): Liste des voies d'administration valides pour substitution.
        list_units (list): Liste des unités de dose valides pour substitution.
        
    Returns:
        pd.Series: La ligne de prescription modifiée avec le détail des erreurs concaténées.
    """
    error_pool = ["route", "drug", "unit", "sous_dosage", "sur_dosage"]
    
    selected_errors = random.sample(error_pool, number_perturb)

    for error in selected_errors:
        if error == "route":
            row['route'] = perturb_route_unit(row['route'], list_routes)
        elif error == "drug":
            row['drug'] = perturb_drug(row['drug'])
        elif error == "unit":
            row['dose_unit_rx'] = perturb_route_unit(row['dose_unit_rx'], list_units)
        elif error == "sous_dosage":
            row['dose_val_rx'] = map_lower.get(str(row['drug']), row['dose_val_rx'])
        elif error == "sur_dosage":
            row['dose_val_rx'] = map_upper.get(str(row['drug']), row['dose_val_rx'])
    
    row['error_types'] = "|".join(selected_errors)
    row['nb_errors'] = number_perturb
    return row

def generate_test_dataset_multiple(df_input, number_prescription, number_perturb, map_lower, map_upper):
    """
    Génère un échantillon de test composé exclusivement de prescriptions multi-erreurs.
    
    Args:
        df_input (pd.DataFrame): Source de données originales saines.
        number_prescription (int): Nombre de prescriptions à échantillonner et corrompre.
        number_perturb (int): Nombre d'erreurs à appliquer sur chaque ligne.
        map_lower (dict): Seuils de sous-dosage.
        map_upper (dict): Seuils de sur-dosage.
        
    Returns:
        pd.DataFrame: Un DataFrame de prescriptions corrompues prêt pour l'inférence.
    """
    df_drug = df_input.sample(n=number_prescription).reset_index(drop=True).copy()
    df_drug = df_drug.astype(object)
    
    list_routes = df_input['route'].dropna().unique().tolist()
    list_units = df_input['dose_unit_rx'].dropna().unique().tolist()

    df_drug = df_drug.apply(
        apply_multiple_errors, 
        axis=1, 
        number_perturb=number_perturb,
        map_lower=map_lower,
        map_upper=map_upper,
        list_routes=list_routes,
        list_units=list_units
    )
    
    return df_drug


def generer_datasets_test_perturb(df_final, number_prescription, error_types_to_generate, number_perturb, perturb_dose):
    """
    Point d'entrée principal pour la création des jeux de données d'évaluation.
    
    Calcule les statistiques de dosage par molécule (moyenne et écart-type) pour définir
    les limites de normalité, puis génère deux fichiers :
    1. Un dataset d'erreurs simples (1 erreur par ligne).
    2. Un dataset d'erreurs complexes (N erreurs par ligne). [cite: 60]
    
    Returns:
        tuple: (df_test_simple, df_test_multiple)
    """
    stats = df_final.groupby('drug')['dose_val_rx'].agg(['mean', 'std'])

    map_lower = (stats['mean'] - perturb_dose * stats['std']).to_dict()
    map_upper = (stats['mean'] + perturb_dose * stats['std']).to_dict()
    
    df_validation_test = generate_test_dataset_simple(df_final, number_prescription, map_lower, map_upper)
    df_validation_test['phrase_clinique'] = df_validation_test.apply(generer_phrase, axis=1)

    df_validation_test_mult = generate_test_dataset_multiple(df_final, number_prescription, number_perturb, map_lower, map_upper)
    df_validation_test_mult['phrase_clinique'] = df_validation_test_mult.apply(generer_phrase, axis=1)

    return df_validation_test, df_validation_test_mult

