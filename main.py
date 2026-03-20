import sys
import pandas as pd
#from creation_dataset_test import generer_datasets_test
from modele_debertav3_bis import executer_pipeline_entrainement, executer_pipeline_inference

from evaluation_modele import evaluer_pipeline_complet

df_train = pd.read_parquet('/teamspace/studios/this_studio/df_train.parquet', engine='pyarrow')
df_test = pd.read_parquet('/teamspace/studios/this_studio/df_test.parquet', engine='pyarrow')

print("dataframe train:")
print(df_train.head())
print(df_train.shape)
print("dataframe test:")
print(df_test.head())
print(df_test.shape)

print(f"Taille finale : {len(df_test)}")
print(f"Proportion d'erreurs : {(df_test['nb_errors'] > 0).mean() * 100:.2f} %")
print("\nRépartition détaillée :")
print(df_test['error_types'].value_counts())

# DebertaV3

print("\nDémarrage entrainement DebertaV3")
df_train_reduit = df_train.sample(n=100000, random_state=42).copy()
#config_modele = executer_pipeline_entrainement(df_train_reduit)

config_modele = {
    "chemin_modele": "/teamspace/studios/this_studio/modele_mimic_debertaV3",
    "model_checkpoint": "microsoft/deberta-v3-small",
    "max_length": 91
}

print("\nDémarage de l'inférence")
df_predictions = executer_pipeline_inference(df_test, config_modele)

print("\nAperçu des résultats :")
colonnes_a_afficher = ['nb_errors', 'label_vrai', 'label_pred', 'error_types_pred']
print(df_predictions[colonnes_a_afficher].head(10))

print("\nDémarage de l'évaluation")
rapport_evaluation = evaluer_pipeline_complet(df_predictions)
exporter_resultat("modele_debertaV3", df_predictions, rapport_evaluation)


