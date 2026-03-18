from nettoyage_données import executer_pipeline_nettoyage
from creation_dataset_test import generer_datasets_test


def main():
    print("Démarrage du nettoyage...")
    df_propre = executer_pipeline_nettoyage()

    print("Génération des datasets de test...")
    df_test_simple, df_test_multiple = generer_datasets_test(df_propre, 14, ['route', 'drug', 'unit', 'sous_dosage', 'sur_dosage'], 2)


if __name__ == "__main__":
    main()