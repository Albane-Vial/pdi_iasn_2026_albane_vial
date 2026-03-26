import argparse
import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
import joblib
from transformers import AutoTokenizer, AutoModel
from creation_dataset_test import generer_datasets_test

from nettoyage_donnees import executer_pipeline_nettoyage
from modeles.deberta.modele_debertaV3 import (
    executer_pipeline_entrainement, 
    executer_pipeline_inference, 
    charger_modele_et_tokenizer
)
from  modeles.deberta.explicabilite_deberta import (
    generer_explicabilite_ig_deberta,
    generer_explicabilite_shap_deberta, 
    generer_explicabilite_lime_deberta, 
    extraire_erreurs_pour_explicabilite,
    executer_audit_organise
    )

from modeles.lof.modele_lof import (
    iterateur_embeddings_tokens,
    entrainer_pipeline_lof_generateur,
    sauvegarder_pipeline_lof,
    charger_pipeline_lof,
    executer_pipeline_inference_lof
)

from modeles.isolation_forest.modele_if import (
    entrainer_pipeline_if,
    sauvegarder_pipeline_if,
    charger_pipeline_if,
    executer_pipeline_inference_if
)
from  modeles.isolation_forest.explicabilite_if import (
    _pipeline_prediction_boite_noire,
    generer_explicabilite_shap_if, 
    generer_explicabilite_lime_if, 
)

from modeles.evaluation_modele import (
    evaluer_pipeline_complet,
    exporter_resultat
)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

params = {
    'database': 'mimic_iv',
    'user': 'mimic_read_only_user',
    'password': 'mimiciv',
    'host': 'localhost',
    'port': '5432'
}
def run_preparation_donnees(args):
    """
    Exécute la préparation des données (extraction SQL et génération des datasets).

    Args:
        args (argparse.Namespace): Arguments contenant data_dir, test_size, etc.
        
    Returns:
        None
    """
    df = None
    chemin_data = Path(args.data_dir)
    chemin_data.mkdir(parents=True, exist_ok=True)
    fichier_propre = chemin_data / "df_mimic.parquet"

    if args.clean:
        logger.info("Démarrage de l'extraction et du nettoyage depuis PostgreSQL...")
        df = executer_pipeline_nettoyage(params)
        
        df.to_parquet(fichier_propre, index=False)
        logger.info(f"Données nettoyées sauvegardées sous : {fichier_propre}")

    if args.generate:
        logger.info("Démarrage de la génération des datasets Train/Test...")
        
        if df is None:
            if fichier_propre.exists():
                logger.info("Chargement du dataset propre existant...")
                df = pd.read_parquet(fichier_propre)
            else:
                logger.error(f"Impossible de générer : le fichier {fichier_propre} n'existe pas. Lancez d'abord --clean.")
                return

        perturbations = [0.2, 0.5, 1, 2]
        
        df_train, dict_tests = generer_datasets_test(
            df=df,
            test_size=args.test_size,
            perturb_values=perturbations,
            dossier_data=args.data_dir
        )
        
        df_train.to_parquet(Path(args.data_dir) / "df_train.parquet", index=False)
        logger.info("Génération terminée : df_train.parquet et dossiers data_test/ créés.")
       

def charger_donnees_test(repertoire_base: str) -> dict:     
    """
    Charge les datasets de test depuis le disque pour différentes perturbations.

    Args:
        repertoire_base (str): Répertoire racine contenant le dossier data_test.

    Returns:
        dict: Dictionnaire {perturbation_value: DataFrame}.
    """
    base_path = Path(repertoire_base) / "data_test"
    fichiers = {
        0.2: 'df_test_perturb_0_2.parquet',
        0.5: 'df_test_perturb_0_5.parquet',
        1:   'df_test_perturb_1.parquet',
        2:   'df_test_perturb_2.parquet',
    }
    
    datasets = {}
    
    for p_val, fichier in fichiers.items():
        chemin_complet = base_path / fichier
        if chemin_complet.exists():
            datasets[p_val] = pd.read_parquet(chemin_complet, engine='pyarrow')
        else:
            logger.warning(f"Fichier introuvable : {chemin_complet}")
            
    return datasets

def preparer_arborescence(base_dir: str, nom_modele: str) -> dict:
    """
    Crée l'arborescence de sortie pour un modèle spécifique.

    Args:
        base_dir (str): Répertoire de sortie principal.
        nom_modele (str): Nom du modèle pour sous-dossier.

    Returns:
        dict: Chemins des dossiers 'model', 'predictions' et 'evaluations'.
    """
    base_path = Path(base_dir) / nom_modele
    dossiers = {
        "model": base_path / "model_files",
        "predictions": base_path / "predictions",
        "evaluations": base_path / "evaluations"
    }
    
    for chemin in dossiers.values():
        chemin.mkdir(parents=True, exist_ok=True)
        
    return dossiers

def run_deberta(args):
    """
    Gère le cycle de vie complet du modèle DeBERTaV3 (entraînement, inférence, explicabilité).

    Args:
        args (argparse.Namespace): Arguments de configuration du modèle.

    Returns:
        None
    """
    logger.info("Démarrage du pipeline DeBERTaV3")
    
    chemins = preparer_arborescence(args.output_dir, "deberta")
    
    config_modele = {
        "chemin_modele": str(chemins["model"]), 
        "model_checkpoint": "microsoft/deberta-v3-small",
        "max_length": 91
    }

    if args.train:
        logger.info("Mode entraînement activé.")
        df_train = pd.read_parquet(Path(args.data_dir) / 'df_train.parquet', engine='pyarrow')
        df_train_reduit = df_train.sample(n=min(100000, len(df_train)), random_state=42).copy()
        config_modele = executer_pipeline_entrainement(df_train_reduit, config_modele["chemin_modele"])

    if args.test:
        logger.info("Mode inférence activé.")
        datasets_test = charger_donnees_test(args.data_dir)
        
        for p_val, df_test in datasets_test.items():
            logger.info(f"Évaluation sur le dataset de perturbation : {p_val}")
            df_predictions = executer_pipeline_inference(df_test, config_modele)
            rapport_evaluation = evaluer_pipeline_complet(df_predictions)    
            
            exporter_resultat(
                nom_modele="modele_debertaV3", 
                pertub_value=p_val, 
                df_predictions=df_predictions, 
                rapport_evaluation=rapport_evaluation,
                dossier_pred=chemins["predictions"], 
                dossier_eval=chemins["evaluations"]
            )
    if args.explain:
        logger.info(f"Mode explicabilité activé avec la méthode : {args.xai_method}")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        chemin_modele_local = str(chemins["model"])
        modele, tokenizer = charger_modele_et_tokenizer(chemin_modele_local, device)

        dossier_predictions = chemins["predictions"]
        fichiers_preds = list(dossier_predictions.glob("*.parquet"))
        
        if not fichiers_preds:
            logger.error("Aucun fichier de prédiction trouvé.")
            return
            
        for fichier_pred in fichiers_preds:
            logger.info(f"Traitement du fichier de prédiction : {fichier_pred.name}")
            
            df_audit = extraire_erreurs_pour_explicabilite(str(fichier_pred))
            print(df_audit)
            if df_audit.empty:
                logger.warning(f"Aucune erreur trouvée dans {fichier_pred.name}, passage au fichier suivant.")
                continue

            dossier_audit_racine = chemins["evaluations"] / f"audit_{fichier_pred.stem}"
            
            executer_audit_organise(
                df_audit=df_audit, 
                modele=modele, 
                tokenizer=tokenizer, 
                device=device, 
                dossier_racine=dossier_audit_racine, 
                args=args
            )
            
            torch.cuda.empty_cache()
def run_lof(args):
    """
    Gère le cycle de vie complet du modèle LOF (entraînement, inférence).

    Args:
        args (argparse.Namespace): Arguments de configuration du modèle.
        
    Returns:
        None
    """
    logger.info("Démarrage du pipeline LOF")
    chemins = preparer_arborescence(args.output_dir, "lof")

    model_lof, model_pca, model_scaler = None, None, None

    if args.train:
        logger.info("Mode entraînement LOF activé.")
        df_train = pd.read_parquet(Path(args.data_dir) / 'df_train.parquet', engine='pyarrow')
        df_train_reduit = df_train.sample(n=min(10000, len(df_train)), random_state=42).copy()
        phrases_train = df_train_reduit['phrase_clinique'].tolist()
                
        model_lof, model_pca, model_scaler = entrainer_pipeline_lof_generateur(
            phrases_train=phrases_train, 
            n_components=50, 
            n_neighbors=20, 
            batch_size=64
        )
        
        sauvegarder_pipeline_lof(model_lof, model_pca, model_scaler, chemin_dossier=str(chemins["model"]))
        

    if args.test:
        logger.info("Mode inférence LOF activé.")
        if model_lof is None:
            logger.info("Chargement des modèles depuis le disque...")
            try:
                model_lof, model_pca, model_scaler = charger_pipeline_lof(chemin_dossier=str(chemins["model"]))
            except FileNotFoundError as e:
                logger.error(f"Échec critique : {e}")
                return 

        datasets_test = charger_donnees_test(args.data_dir)

        for p_val, df_test in datasets_test.items():
            logger.info(f"Évaluation sur le dataset de perturbation : {p_val}")
            
            df_predictions = executer_pipeline_inference_lof(
                df_test=df_test,
                model_lof=model_lof,
                model_pca=model_pca,
                model_scaler=model_scaler,
                batch_size=32 
            )
            
            rapport_evaluation = evaluer_pipeline_complet(df_predictions)    
            
            exporter_resultat(
                nom_modele="modele_lof", 
                p_val=p_val, 
                df_predictions=df_predictions, 
                rapport_evaluation=rapport_evaluation,
                dossier_pred=chemins["predictions"], 
                dossier_eval=chemins["evaluations"]
            )
def run_if(args):
    """
    Gère le cycle de vie complet du modèle Isolation Forest (entraînement, inférence, explicabilité).

    Args:
        args (argparse.Namespace): Arguments de configuration du modèle.
        
    Returns:
        None
    """
    logger.info("Démarrage du pipeline Isolation Forest (IF)")
    chemins = preparer_arborescence(args.output_dir, "if")

    modele_if, model_pca, model_scaler = None, None, None
    model_id = "bert-base-multilingual-cased"

    if args.train:
        logger.info("Mode entraînement IF activé.")
        df_train = pd.read_parquet(Path(args.data_dir) / 'df_train.parquet', engine='pyarrow')
        df_train_reduit = df_train.sample(n=min(10000, len(df_train)), random_state=42).copy()
        phrases_train = df_train_reduit['phrase_clinique'].tolist()
                
        modele_if, model_pca, model_scaler, _, _ = entrainer_pipeline_if(
            phrases_train=phrases_train,
            n_components=50,
            batch_size=32,
            model_id=model_id
        )
        sauvegarder_pipeline_if(modele_if, model_pca, model_scaler, chemin_dossier=str(chemins["model"]))
        logger.info("Entraînement IF terminé.")

    if args.test:
        logger.info("Mode inférence IF activé.")
        
        try:
            modele_if, model_pca, model_scaler = charger_pipeline_if(chemin_dossier=str(chemins["model"]))
        except FileNotFoundError as e:
            logger.error(f"Échec critique : {e}")
            return 

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        bert_model = AutoModel.from_pretrained(model_id).to(device)
        special_tokens_dict = {'additional_special_tokens': ['[DRUG]', '[ROUTE]', '[UNIT]', '[DOSE]']}
        tokenizer.add_special_tokens(special_tokens_dict)
        bert_model.resize_token_embeddings(len(tokenizer))

        bert_model.eval()

        datasets_test = charger_donnees_test(args.data_dir)

        for p_val, df_test in datasets_test.items():
            logger.info(f"Évaluation sur le dataset de perturbation : {p_val}")
            
            df_predictions = executer_pipeline_inference_if(
                df_test=df_test,
                modele_if=modele_if,
                model_pca=model_pca,
                model_scaler=model_scaler,
                tokenizer=tokenizer,
                model_bert=bert_model,
                batch_size=32
            )
            
            rapport_evaluation = evaluer_pipeline_complet(df_predictions)    
            
            exporter_resultat(
                nom_modele="modele_if", 
                pertub_value=p_val, 
                df_predictions=df_predictions, 
                rapport_evaluation=rapport_evaluation,
                dossier_pred=chemins["predictions"], 
                dossier_eval=chemins["evaluations"]
            )

    if args.explain:
        logger.info(f"Mode explicabilité IF activé avec la méthode : {args.xai_method}")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_id = "bert-base-multilingual-cased"

        try:
            modele_if, model_pca, model_scaler = charger_pipeline_if(chemin_dossier=str(chemins["model"]))
        except FileNotFoundError as e:
            logger.error(f"Échec critique : Le modèle n'est pas entraîné. {e}")
            return

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        bert_model = AutoModel.from_pretrained(model_id).to(device)
        
        nouvelles_balises = ['[DRUG]', '[ROUTE]', '[UNIT]', '[DOSE]']
        tokenizer.add_tokens(nouvelles_balises)
        bert_model.resize_token_embeddings(len(tokenizer))
        bert_model.eval()

        dossier_predictions = chemins["predictions"]
        fichiers_preds = list(dossier_predictions.glob("*.parquet"))
        
        if not fichiers_preds:
            logger.error(f"Aucun fichier de prédiction trouvé dans {dossier_predictions}. Exécutez --test d'abord.")
            return
            
        for fichier_pred in fichiers_preds:
            logger.info(f"Génération de l'explicabilité pour : {fichier_pred.name}")
            
            df_erreurs = extraire_erreurs_pour_explicabilite(str(fichier_pred))
            
            if df_erreurs.empty: 
                logger.info("Aucune erreur de classification détectée sur ce dataset.")
                continue
                
            df_audit = df_erreurs.head(100) 
            chemin_base_export = chemins["evaluations"] / f"explicabilite_{args.xai_method}_if_{fichier_pred.stem}.xlsx"
            
            if args.xai_method in ['shap', 'all']:
                logger.info("Exécution de SHAP (Explainer agnostique)...")
                generer_explicabilite_shap_if(
                    df_audit=df_audit, 
                    modele_if=modele_if, 
                    pca_model=model_pca, 
                    scaler_model=model_scaler, 
                    tokenizer=tokenizer, 
                    bert_model=bert_model, 
                    chemin_export=str(chemin_base_export).replace('.xlsx', '_shap.xlsx')
                )
            
            if args.xai_method in ['lime', 'all']:
                logger.info("Exécution de LIME (TextExplainer)...")
                generer_explicabilite_lime_if(
                    df_audit=df_audit, 
                    modele_if=modele_if, 
                    pca_model=model_pca, 
                    scaler_model=model_scaler, 
                    tokenizer=tokenizer, 
                    bert_model=bert_model, 
                    chemin_export=str(chemin_base_export).replace('.xlsx', '_lime.xlsx')
                )
                
            if args.xai_method == 'ig':
                logger.warning("La méthode IG (Integrated Gradients) est mathématiquement incompatible avec l'Isolation Forest (modèle non différentiable). Ignoré.")
                
            torch.cuda.empty_cache()
    
def main():
    """
    Point d'entrée CLI pour orchestrer l'ensemble du pipeline.

    Args:
        Aucun (géré via argparse).
        
    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Pipeline d'entraînement et d'évaluation.")
    
    # Arguments de données
    parser.add_argument('--clean', action='store_true', help="Nettoyage PostgreSQL.")
    parser.add_argument('--generate', action='store_true', help="Genere datasets train et test.")
    
    # Paramètres de génération
    parser.add_argument('--perturb_dose', type=float, default=0.3)
    parser.add_argument('--test_size', type=float, default=0.10)
    parser.add_argument('--error_ratio', type=float, default=0.30)

    # Arguments de modèles 
    parser.add_argument('--model', type=str, choices=['deberta', 'lof', 'if', 'all'], 
                        help="Requis pour train/test/explain.")
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--explain', action='store_true')
    parser.add_argument('--xai_method', type=str, choices=['ig', 'shap', 'lime', 'all'], default='ig', 
                        help="Méthode d'explicabilité à utiliser.")  

    parser.add_argument('--data_dir', type=str, default='data', 
                            help="Répertoire des données")
    parser.add_argument('--output_dir', type=str, default='modeles_sorties', 
                        help="Dossier de sauvegarde des modèles.")

    args = parser.parse_args()

    logger.info(f"Les sorties seront sauvegardées dans : {Path(args.output_dir).absolute()}")


    if args.clean or args.generate:
        run_preparation_donnees(args)

    actions_modeles = [args.train, args.test, args.explain]

    if any(actions_modeles):
        if not args.model:
            parser.error("L'argument --model est requis lorsque --train, --test ou --explain est utilisé.")
        
        if args.model in ['deberta', 'all']:
            run_deberta(args)
        
        if args.model in ['lof', 'all']:
            run_lof(args)
            
        if args.model in ['if', 'all']:
            run_if(args)
    
    elif not any([args.clean, args.generate]):
        parser.print_help()
if __name__ == "__main__":
    main()

