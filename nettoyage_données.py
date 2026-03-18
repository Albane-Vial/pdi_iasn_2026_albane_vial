import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split


DATA_PATH = '/kaggle/input/datasets/mangeshwagle/mimic-iv-2-1/mimic-iv-2.1/hosp/'

print("Chargement des prescriptions MIMIC en cours...")


def extraire_donnée_precriptions():
    print("Extraction et nettoyage des données de prescriptions...")
    df_prescriptions = pd.read_csv(DATA_PATH + 'prescriptions.csv', nrows=500000, low_memory=False)
    colonnes_utiles = ['hadm_id','subject_id','drug', 'dose_val_rx', 'dose_unit_rx', 'form_val_disp', 'form_unit_disp', 'route']
    df_prescriptions = df_prescriptions[colonnes_utiles].dropna().copy()
    comptage = df_prescriptions["drug"].value_counts()
    medicaments_frequents = comptage[comptage > 100].index
    df_prescriptions = df_prescriptions[df_prescriptions['drug'].isin(medicaments_frequents)].copy()
    
    df_prescriptions['dose_val_rx'] = pd.to_numeric(df_prescriptions['dose_val_rx'], errors='coerce')
    quantiles = df_prescriptions.groupby('drug')['dose_val_rx'].quantile([0.01, 0.99]).unstack()
    quantiles.columns = ['q01', 'q99']
    df_prescriptions['q01'] = df_prescriptions['drug'].map(quantiles['q01'])
    df_prescriptions['q99'] = df_prescriptions['drug'].map(quantiles['q99'])

    df_prescriptions = df_prescriptions[
        (df_prescriptions['dose_val_rx'] >= df_prescriptions['q01']) & 
        (df_prescriptions['dose_val_rx'] <= df_prescriptions['q99'])
    ]
    print(f"Nombre de prescriptions après nettoyage : {len(df_prescriptions)}")
    df_prescriptions.drop(columns=['q01', 'q99'], inplace=True)
    return df_prescriptions
def extraire_donnée_patients():
    print("Extraction et nettoyage des données de patients...")
    df_patients = pd.read_csv(DATA_PATH + 'patients.csv', low_memory=False)
    colonnes_utiles = ['subject_id', 'gender', 'anchor_age']
    df_patients = df_patients[colonnes_utiles].dropna().copy()
    return df_patients
def extraire_donnée_admissions():
    print("Extraction et nettoyage des données d'admissions...")
    df_admissions = pd.read_csv(DATA_PATH + 'admissions.csv', low_memory=False)
    colonnes_utiles = ['hadm_id', 'subject_id', 'admission_type']
    df_admissions = df_admissions[colonnes_utiles].dropna().copy()
    return df_admissions
def extraire_donnée_diagnostics():
    print("Extraction et nettoyage des données de diagnostics...")
    df_diag = pd.read_csv(DATA_PATH + 'diagnoses_icd.csv')
    df_diag_name = pd.read_csv(DATA_PATH + 'd_icd_diagnoses.csv')
    df_diag_filtre = df_diag[df_diag['seq_num'] <= 2]

    df_total_diag = pd.merge(
        df_diag_filtre[['subject_id', 'hadm_id', 'icd_code', 'icd_version']], 
        df_diag_name[['icd_version', 'icd_code', 'long_title']], 
        on=['icd_version', 'icd_code'], 
        how='left'
    )

    df_total_diag = df_total_diag[['subject_id', 'hadm_id', 'long_title']]

    df_diag_merged = df_total_diag.groupby(
        ['subject_id', 'hadm_id'], 
        as_index=False
    ).agg({
        'long_title': lambda x: ' ; '.join(x.dropna().astype(str))
    })
    return df_diag_merged
def extraire_donnée_biologie():
    print("Extraction et nettoyage des données de biologie...")
    items_cibles = [50912, 50971, 51265, 51274, 51275, 50931, 50861, 50878]

    df_lab_bio = pd.read_csv(DATA_PATH + 'labevents.csv', usecols=['subject_id', 'hadm_id', 'itemid', 'flag'])
    df_d_labitems = pd.read_csv(DATA_PATH + 'd_labitems.csv', usecols=['itemid', 'label'])

    df_lab_bio = df_lab_bio[df_lab_bio['itemid'].isin(items_cibles)]

    df_lab_bio['flag'] = df_lab_bio['flag'].fillna('normal')
    df_lab_bio = df_lab_bio.dropna()

    df_lab_bio['priority'] = df_lab_bio['flag'].map({'abnormal': 0, 'normal': 1})

    df_lab_bio = pd.merge(
        df_lab_bio, 
        df_d_labitems, 
        on='itemid', 
        how='left'
    )
    df_lab_bio = df_lab_bio.drop(columns=['itemid'])

    df_lab_bio = df_lab_bio.sort_values(['subject_id', 'hadm_id', 'label', 'priority'])

    df_lab_bio_unique = df_lab_bio.drop_duplicates(subset=['subject_id', 'hadm_id', 'label'])

    df_lab_bio_unique['label_with_flag'] = (
        df_lab_bio_unique['label'] + " (" + df_lab_bio_unique['flag'] + ")"
    )

    df_bio_merged = df_lab_bio_unique.groupby(['subject_id', 'hadm_id'], as_index=False).agg({
        'label': lambda x: ' ; '.join(x)
    })
    return df_bio_merged

def generer_phrase(row):
    dose = row['dose_val_rx'] if pd.notnull(row['dose_val_rx']) else "N/A"
    
    return (f"[DRUG] {row['drug']} [DOSE] {dose} [UNIT] {row['dose_unit_rx']} "
            f"[ROUTE] {row['route']} "
            f"[GEN] {row['gender']} [ADM] {row['admission_type']} "
            f"[DX] {row['nom_diag']} "
            f"[BIO] {row['nom_bio']}")

def executer_pipeline_nettoyage():
    print("Démarrage du nettoyage...")

    df_prescriptions = extraire_donnée_precriptions()
    df_patients = extraire_donnée_patients()
    df_admissions = extraire_donnée_admissions()
    df_diagnostics = extraire_donnée_diagnostics()
    df_biologie = extraire_donnée_biologie()

    df_final = df_prescriptions.merge(df_diagnostics, on=['subject_id','hadm_id'], how='left')
    df_final = df_final.merge(df_biologie, on=['subject_id','hadm_id'], how='left')
    df_final = df_final.merge(df_patients, on='subject_id', how='left')
    df_final = df_final.merge(df_admissions, on=['subject_id','hadm_id'], how='left')

    df_final.rename(columns={'long_title': 'nom_diag'}, inplace=True)
    df_final.rename(columns={'label': 'nom_bio'}, inplace=True)

    df_final = df_final.drop_duplicates()
    print(f"Génération de la phrase clinique")
    df_final['phrase_clinique'] = df_final.apply(generer_phrase, axis=1)

    return df_final
