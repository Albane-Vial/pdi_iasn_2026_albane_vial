import random
import string
import pandas as pd

def perturb_drug(text):
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
    candidates = [t for t in list_text if t != texte]
    if not candidates:
        return texte 
    return random.choice(candidates)
def perturb_numeric(row, dosage_type, map_lower, map_upper):
    drug_name = str(row['drug'])
    if dosage_type == 'sous_dosage':
        return map_lower.get(drug_name, row['dose_val_rx']) 
    else:
        return map_upper.get(drug_name, row['dose_val_rx'])

def generate_test_dataset_simple(df_input, number_prescription, map_lower, map_upper):
 
    # Échantillonnage
    df_drug = df_input.sample(n=number_prescription).reset_index(drop=True).copy()
    
    # Initialisation
    df_drug['error_types'] = "none"
    df_drug['nb_errors'] = 0
    df_drug = df_drug.astype(object) # Forcer le type pour l'insertion de texte/valeurs
    
    split_num = number_prescription // 5
    list_routes = df_input['route'].dropna().unique().tolist()
    list_units = df_input['dose_unit_rx'].dropna().unique().tolist()

    # --- Application des erreurs par tranches (exclusivité des index) ---
    
    # 1. Route
    start, end = 0, split_num
    df_drug.loc[start:end, 'route'] = df_drug.loc[start:end, 'route'].apply(lambda x: perturb_route_unit(x, list_routes))
    df_drug.loc[start:end, ['nb_errors', 'error_types']] = [1, "route"]

    # 2. Drug (Notez le +1 pour éviter le chevauchement)
    start, end = end + 1, 2 * split_num
    df_drug.loc[start:end, 'drug'] = df_drug.loc[start:end, 'drug'].apply(perturb_drug)
    df_drug.loc[start:end, ['nb_errors', 'error_types']] = [1, "drug"]

    # 3. Unité
    start, end = end + 1, 3 * split_num
    df_drug.loc[start:end, 'dose_unit_rx'] = df_drug.loc[start:end, 'dose_unit_rx'].apply(lambda x: perturb_route_unit(x, list_units))
    df_drug.loc[start:end, ['nb_errors', 'error_types']] = [1, "unit_dosage"]

    # 4. Sous-dosage (Utilisation du dictionnaire map_lower)
    start, end = end + 1, 4 * split_num
    df_drug.loc[start:end, 'dose_val_rx'] = df_drug.loc[start:end].apply(
        lambda r: map_lower.get(str(r['drug']), r['dose_val_rx']), axis=1)
    df_drug.loc[start:end, ['nb_errors', 'error_types']] = [1, "sous_dosage"]

    # 5. Sur-dosage (Utilisation du dictionnaire map_upper)
    start, end = end + 1, number_prescription - 1
    df_drug.loc[start:end, 'dose_val_rx'] = df_drug.loc[start:end].apply(
        lambda r: map_upper.get(str(r['drug']), r['dose_val_rx']), axis=1)
    df_drug.loc[start:end, ['nb_errors', 'error_types']] = [1, "sur_dosage"]

    return df_drug
def generate_multi_errors_per_row(df_input, number_prescription, errors_to_generate, map_lower, map_upper):
    """
    errors_to_generate : list, ex ['route', 'drug', 'sur_dosage']
    """
    new_rows = []
    df_drug = df_input.sample(n=number_prescription).reset_index(drop=True).copy()
    
    list_routes = df_drug['route'].dropna().unique().tolist()
    list_units = df_drug['dose_unit_rx'].dropna().unique().tolist()

    # On itère sur chaque ligne de l'input original
    for _, row in df_drug.iterrows():
        
        # Pour chaque type d'erreur demandé, on crée une copie spécifique
        for error_type in errors_to_generate:
            corrupted_row = row.copy()
            corrupted_row['error_types'] = error_type
            corrupted_row['nb_errors'] = 1
            
            # Application de la perturbation spécifique
            if error_type == "route":
                corrupted_row['route'] = perturb_route_unit(row['route'], list_routes)
            elif error_type == "drug":
                corrupted_row['drug'] = perturb_drug(row['drug'])
            elif error_type == "unit":
                corrupted_row['dose_unit_rx'] = perturb_route_unit(row['dose_unit_rx'], list_units)
            elif error_type == "sous_dosage":
                corrupted_row['dose_val_rx'] = map_lower.get(str(row['drug']), row['dose_val_rx'])
            elif error_type == "sur_dosage":
                corrupted_row['dose_val_rx'] = map_upper.get(str(row['drug']), row['dose_val_rx'])
            
            new_rows.append(corrupted_row)

    # Conversion de la liste de Series en DataFrame
    return pd.DataFrame(new_rows).reset_index(drop=True)
def apply_multiple_errors(row, number_perturb, map_lower, map_upper, list_routes, list_units):
    # On définit les types d'erreurs possibles
    error_pool = ["route", "drug", "unit", "sous_dosage", "sur_dosage"]
    
    # Sélection aléatoire de N erreurs distinctes
    selected_errors = random.sample(error_pool, number_perturb)
    
    # Application séquentielle
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
    # 1. Préparation
    df_drug = df_input.sample(n=number_prescription).reset_index(drop=True).copy()
    df_drug = df_drug.astype(object)
    
    # 2. Ressources nécessaires
    list_routes = df_input['route'].dropna().unique().tolist()
    list_units = df_input['dose_unit_rx'].dropna().unique().tolist()

    # 3. Application via .apply() avec arguments nommés
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


def generer_datasets_test(df_final, number_prescription, error_types_to_generate, number_perturb):
    stats = df_final.groupby('drug')['dose_val_rx'].agg(['mean', 'std'])

    map_lower = (stats['mean'] - 2 * stats['std']).to_dict()
    map_upper = (stats['mean'] + 2 * stats['std']).to_dict()
    
    df_validation_test = generate_test_dataset_simple(df_final, number_prescription, map_lower, map_upper)
    df_validation_test_mult = generate_test_dataset_multiple(df_final, number_prescription, number_perturb, map_lower, map_upper)
    
    return df_validation_test, df_validation_test_mult