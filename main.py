import sys
import pandas as pd
#from sklearn.model_selection import train_test_split

#from creation_dataset_test import generer_datasets_test
#from modele_debertaV3 import executer_pipeline_entrainement, executer_pipeline_inference
#from modele_debertav3_old  import modele_old, executer_pipeline_inference_old
from modele_debertav3_bis import executer_pipeline_entrainement, executer_pipeline_inference

from evaluation_modele import evaluer_pipeline_complet

#j'ai generer un df avec kaggle mais en gros j'ai utilise le fichier nettoyage_donnée et puis j'ai generer dans un df_test avec 30% d'erreur 15% avec une erreur et 15% avec 2 erreurs
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

print("\nDémarrage entrainement DebertaV3")
df_train_reduit = df_train.sample(n=100000, random_state=42).copy()
config_modele = executer_pipeline_entrainement(df_train_reduit)
#config_modele = modele_old(df_train_reduit)
"""
config_modele = {
    "chemin_modele": "/teamspace/studios/this_studio/modele_mimic_debertaV3",
    "model_checkpoint": "microsoft/deberta-v3-small",
    "max_length": 91
}
"""
print("\nConfiguration exportée :")
print(config_modele)


print("\nDémarage de l'inférence")

df_predictions = executer_pipeline_inference(df_test, config_modele)
#df_predictions = executer_pipeline_inference_old(df_test, config_modele)


print("\nAperçu des résultats :")
colonnes_a_afficher = ['nb_errors', 'label_vrai', 'label_pred', 'error_types_pred']
print(df_predictions[colonnes_a_afficher].head(10))

print("\n=Démarage de l'évaluation")

rapport_evaluation = evaluer_pipeline_complet(df_predictions)

# Affichage structuré des résultats de la détection binaire
print("\nRésultats globals:")
resultats_binaires = rapport_evaluation['Detection_Binaire']
print(f"Accuracy  : {resultats_binaires['Accuracy']:.4f}")
print(f"Precision : {resultats_binaires['Precision']:.4f}")
print(f"Recall    : {resultats_binaires['Recall']:.4f}")
print(f"F1 Score  : {resultats_binaires['F1_Score']:.4f}")
print(f"Matrice de confusion : {resultats_binaires['Matrice_Confusion']}")


print("\nRésultat Multi Label")
resultats_multi = rapport_evaluation['Caracterisation_Multilabel']

if "Erreur" in resultats_multi:
    print(resultats_multi["Erreur"])
else:
    print(f"F1 Macro : {resultats_multi['F1_Macro']:.4f}")
    print(f"F1 Micro : {resultats_multi['F1_Micro']:.4f}")
    print("\nRapport Détaillé :\n")
    print(resultats_multi['Rapport_Detaille'])
