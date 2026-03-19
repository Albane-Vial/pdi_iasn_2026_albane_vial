import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from mimic_albanevial.nettoyage_données import executer_pipeline_nettoyage
from creation_dataset_test import generer_datasets_test

df_propre = executer_pipeline_nettoyage()

df_train, df_test = train_test_split(df_propre, test_size=0.10, random_state=42)


total_lignes = len(df_test)
nb_erreurs_cible = int(total_lignes * 0.30)
nb_erreurs_par_type = nb_erreurs_cible // 2 
nb_propres = total_lignes - (nb_erreurs_par_type * 2)


df_test_propre = df_test.sample(n=nb_propres, random_state=42).copy()
df_source_erreurs = df_test.drop(df_test_propre.index)

df_test_propre['error_types'] = "none"
df_test_propre['nb_errors'] = 0

df_erreurs_simples, df_erreurs_multiples = generer_datasets_test(
    df_final=df_source_erreurs, 
    number_prescription=nb_erreurs_par_type, 
    error_types_to_generate=[], 
    number_perturb=2          
)

df_test_final = pd.concat([df_test_propre, df_erreurs_simples, df_erreurs_multiples], ignore_index=True)
df_test_final = df_test_final.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Taille finale : {len(df_test_final)}")
print(f"Proportion d'erreurs : {(df_test_final['nb_errors'] > 0).mean() * 100:.2f} %")

df_test_final.to_parquet('df_test.parquet')
df_train.to_parquet('df_train.parquet')