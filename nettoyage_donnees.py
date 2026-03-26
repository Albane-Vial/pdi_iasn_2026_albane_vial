import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
from sqlalchemy import create_engine, text
import os


def execute_query(sql_query, user, password, host, port, database):
    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{database}')

    try:
        df = pd.read_sql_query(sql_query, engine)
        return df
    finally:
        engine.dispose()

def extraire_donnée_diagnostics(user, password, host, port, database):
    """
    Extrait et agrège les diagnostics principaux et secondaires depuis MIMIC-IV.

    Args:
        user (str): Identifiant PostgreSQL.
        password (str): Mot de passe PostgreSQL.
        host (str): Adresse du serveur.
        port (str): Port de communication.
        database (str): Nom de la base de données.

    Returns:
        pd.DataFrame ou None: DataFrame contenant 'subject_id', 'hadm_id', 
        et 'nom_diag'. Retourne None en cas d'échec de la transaction.
    """
    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{database}')
    
    query = text("""
    SELECT 
        diag.subject_id, 
        diag.hadm_id, 
        diag.seq_num, 
        ref.long_title
    FROM hosp.diagnoses_icd diag
    LEFT JOIN hosp.d_icd_diagnoses ref 
        ON diag.icd_code = ref.icd_code 
        AND diag.icd_version = ref.icd_version
    """)
    
    try:
        with engine.connect() as conn:
            df_diag = pd.read_sql_query(query, conn)
            
        print("Extraction et nettoyage des données de diagnostics en cours...")
        
        df_diag_filtre = df_diag[df_diag['seq_num'] <= 2].copy()
        
        df_diag_filtre = df_diag_filtre[['subject_id', 'hadm_id', 'long_title']]
        
        df_diag_merged = df_diag_filtre.groupby(
            ['subject_id', 'hadm_id'], 
            as_index=False
        ).agg({
            'long_title': lambda x: ' ; '.join(x.dropna().astype(str))
        })
        
        df_diag_merged.rename(columns={'long_title': 'nom_diag'}, inplace=True)
        
        return df_diag_merged

    except Exception as e:
        print(f"Erreur critique lors de l'extraction des diagnostics : {e}")
        return None

    finally:
        engine.dispose()

def extraire_donnee_patients(user, password, host, port, database):
    """
    Extrait les données démographiques des patients depuis la base MIMIC-IV.

    Args:
        user (str): Identifiant de connexion PostgreSQL.
        password (str): Mot de passe de connexion.
        host (str): Adresse du serveur.
        port (str): Port de communication.
        database (str): Nom de la base de données.

    Returns:
        pd.DataFrame ou None: Un DataFrame contenant 'subject_id', 'gender' 
        et 'anchor_age'. Retourne None en cas d'erreur de connexion.
    """
    # Création du moteur de base de données
    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{database}')

    # Utilisation de text() et suppression de la duplication de anchor_age
    query = text("""
    SELECT 
        subject_id, 
        gender, 
        anchor_age
    FROM hosp.patients
    """)
    
    try:
        with engine.connect() as conn:
            print("Extraction et nettoyage des données du patients en cours...")
            df = pd.read_sql_query(query, conn)
        return df
        
    except Exception as e:
        print(f"Erreur lors de l'extraction des données patients : {e}")
        return None
        
    finally:
        # Libération obligatoire des ressources réseau
        engine.dispose()
def extraire_donnee_admissions(user, password, host, port, database):
    """
    Extrait les données relatives aux types d'admissions hospitalières.

    Args:
        user (str): Identifiant de connexion PostgreSQL.
        password (str): Mot de passe de connexion.
        host (str): Adresse du serveur de base de données.
        port (str): Port de connexion au serveur.
        database (str): Nom de la base de données cible.

    Returns:
        pd.DataFrame ou None: DataFrame contenant les colonnes 'hadm_id' et 
        'admission_type'
    """
    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{database}')
    
    query = text("""
    SELECT 
        hadm_id, 
        admission_type
    FROM hosp.admissions
    """)
    
    try:
        with engine.connect() as conn:
            print("Extraction et nettoyage des données d'admissions en cours...")
            df = pd.read_sql_query(query, conn)
        return df
        
    except Exception as e:
        print(f"Erreur lors de l'extraction des admissions : {e}")
        return None
        
    finally:
        engine.dispose()

def extraire_donnée_precriptions(user, password, host, port, database):
    """
    Extrait et nettoie les données de prescription depuis la base MIMIC-IV.

    Cette fonction exécute trois étapes de prétraitement :
    1. Typage strict des dosages et élimination des valeurs non numériques.
    2. Exclusion des molécules sous-représentées (seuil : <= 100 occurrences).
    3. Suppression des valeurs aberrantes de dosage (exclusion hors des 
       1er et 99e centiles, calculés spécifiquement pour chaque molécule).

    Args:
        user (str): Identifiant de connexion PostgreSQL.
        password (str): Mot de passe.
        host (str): Adresse du serveur.
        port (str): Port de connexion.
        database (str): Nom de la base de données.

    Returns:
        pd.DataFrame: DataFrame nettoyé et prêt pour la fusion des features.
    """
    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{database}')

    query = text("""
    SELECT 
        subject_id,
        hadm_id,
        drug,
        dose_val_rx,
        dose_unit_rx,
        form_val_disp,
        form_unit_disp,
        route
    FROM hosp.prescriptions
    """)
    
    try:
        with engine.connect() as conn:
            print("Extraction et nettoyage des données des prescriptions en cours...")
            df_prescriptions = pd.read_sql_query(query, conn)
        print(f"nombre de ligne avant nettoyage: {len(df_prescriptions)}")
        colonnes_utiles = ['hadm_id','subject_id','drug', 'dose_val_rx', 'dose_unit_rx', 'form_val_disp', 'form_unit_disp', 'route']
        df_prescriptions = df_prescriptions[colonnes_utiles].dropna().copy()
        
        frequence_patients = df_prescriptions['subject_id'].value_counts()
        patients_valides = frequence_patients[frequence_patients < 50].index
        df_prescriptions = df_prescriptions[df_prescriptions['subject_id'].isin(patients_valides)].copy()
        print(f"nombre de ligne après nettoyage patients: {len(df_prescriptions)}")
        
        comptage = df_prescriptions["drug"].value_counts()
        medicaments_frequents = comptage[comptage > 500].index
        df_prescriptions = df_prescriptions[df_prescriptions['drug'].isin(medicaments_frequents)].copy()
        print(f"nombre de ligne après nettoyage drug: {len(df_prescriptions)}")
        
        df_prescriptions['dose_val_rx'] = df_prescriptions['dose_val_rx'].round(2)
        
        df_prescriptions['dose_val_rx'] = pd.to_numeric(df_prescriptions['dose_val_rx'], errors='coerce')
        df_prescriptions = df_prescriptions.dropna(subset=['dose_val_rx'])
        
        q01 = df_prescriptions.groupby('drug')['dose_val_rx'].transform(lambda x: x.quantile(0.01))
        q99 = df_prescriptions.groupby('drug')['dose_val_rx'].transform(lambda x: x.quantile(0.99))

        df_prescriptions = df_prescriptions[
            (df_prescriptions['dose_val_rx'] >= q01) & 
            (df_prescriptions['dose_val_rx'] <= q99)
        ]
        
        print(f"Nombre de prescriptions après nettoyage : {len(df_prescriptions)}")
        return df_prescriptions
        
    finally:
        engine.dispose()

def extraire_donnee_biologie(user, password, host, port, database):
    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{database}')
    query = text("""
    SELECT
        le.subject_id,
        le.hadm_id,
        dl.label AS nom_examen,
        le.flag
    FROM
        hosp.labevents le
    LEFT JOIN
        hosp.d_labitems dl ON le.itemid = dl.itemid
    WHERE
        le.itemid IN (50912, 50971, 51265, 51274, 51275, 50931, 50861, 50878)
        AND le.valuenum IS NOT NULL
    """)
    
    try:
        with engine.connect() as conn:
            print("Extraction et nettoyage des données de biologie en cours...")
            df_lab_bio = pd.read_sql_query(query, conn)

        df_lab_bio['flag'] = df_lab_bio['flag'].fillna('normal')

        df_lab_bio['priority'] = df_lab_bio['flag'].map({'abnormal': 0, 'normal': 1}).fillna(2)

        df_lab_bio = df_lab_bio.sort_values(['subject_id', 'hadm_id', 'nom_examen', 'priority'])

        df_lab_bio_unique = df_lab_bio.drop_duplicates(subset=['subject_id', 'hadm_id', 'nom_examen']).copy()

        df_lab_bio_unique['label_with_flag'] = (
            df_lab_bio_unique['nom_examen'] + " (" + df_lab_bio_unique['flag'] + ")"
        )

        df_bio_merged = df_lab_bio_unique.groupby(['subject_id', 'hadm_id'], as_index=False).agg({
            'label_with_flag': lambda x: ' ; '.join(x)
        })
        
        df_bio_merged.rename(columns={'label_with_flag': 'nom_bio'}, inplace=True)
        
        return df_bio_merged

    finally:
        engine.dispose()

def generer_phrase(row):
    dose = row['dose_val_rx'] if pd.notnull(row['dose_val_rx']) else "N/A"
    
    return (f"[DRUG] {row['drug']} [DOSE] {dose} [UNIT] {row['dose_unit_rx']} "
            f"[ROUTE] {row['route']} "
            f"[GEN] {row['gender']} [ADM] {row['admission_type']} "
            #f"[DX] {row['nom_diag']} "
            f"[BIO] {row['nom_bio']}")

def executer_pipeline_nettoyage(params_connexion):
    """
    Orchestre l'extraction, la fusion et le formatage des données cliniques.

    Args:
        params_connexion (dict): Dictionnaire contenant les identifiants PostgreSQL.

    Returns:
        pd.DataFrame: Le jeu de données final prêt pour la génération d'anomalies.
    """
    print("Démarrage du nettoyage...")
    df_prescriptions = extraire_donnée_precriptions(**params_connexion)
    df_patients = extraire_donnee_patients(**params_connexion)
    df_admissions = extraire_donnee_admissions(**params_connexion)
    # df_diagnostics = extraire_donnée_diagnostics(**params_connexion)
    df_biologie = extraire_donnee_biologie(**params_connexion)


    df_final = df_prescriptions.merge(df_biologie, on=['subject_id', 'hadm_id'], how='left')
    # df_final = df_final.merge(df_diagnostics, on=['subject_id', 'hadm_id'], how='left')
    
    df_final = df_final.merge(df_patients, on='subject_id', how='left')
    df_final = df_final.merge(df_admissions, on=['hadm_id'], how='left')

    df_final = df_final.drop_duplicates()
    
    print(f"Génération de la phrase clinique pour {len(df_final)} prescriptions...")
    
    df_final['phrase_clinique'] = df_final.apply(generer_phrase, axis=1)

    n_samples = 250000

    train_ratio = n_samples / len(df_final)

    # Extraction stratifiée
    df_stratified, _ = train_test_split(
        df_final,
        train_size=n_samples,
        stratify=df_final['drug'],
        random_state=42,
        shuffle=True
    )
    print("Pipeline de nettoyage terminé avec succès.")
    return df_stratified


