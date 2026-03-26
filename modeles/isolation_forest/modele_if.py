import os
import torch
import numpy as np
import joblib
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

def iterateur_embeddings_tokens(phrases, model_id="bert-base-multilingual-cased", batch_size=16, tokenizer=None, model=None):
    """
    Extrait les embeddings contextuels token par token via une approche par lots (générateur).
    
    Cette fonction limite l'utilisation de la VRAM en traitant les données par fragments
    et filtre les tokens de padding et les tokens spéciaux ([CLS], [SEP]).

    Args:
        phrases (list of str): Liste des textes cliniques à analyser.
        model_id (str): Identifiant du modèle HuggingFace.
        batch_size (int): Nombre de phrases traitées simultanément sur le GPU.
        tokenizer (AutoTokenizer, optional): Tokenizer pré-instancié.
        model (AutoModel, optional): Modèle BERT pré-instancié.

    Yields:
        tuple: (batch_tokens, batch_embeddings, batch_phrase_ids)
            - batch_tokens (list): Liste des tokens textuels valides.
            - batch_embeddings (np.ndarray): Matrice des embeddings correspondants (N, 768).
            - batch_phrase_ids (np.ndarray): Indices de traçabilité liant chaque token à sa phrase d'origine.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        
        balises = ['[DRUG]', '[DOSE]', '[UNIT]', '[ROUTE]', '[GEN]', '[ADM]', '[BIO]']
        tokenizer.add_tokens(balises)       
        model.resize_token_embeddings(len(tokenizer)) 
        
        model = model.to(device)
        model.eval()

    for i in range(0, len(phrases), batch_size):
        batch = phrases[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        
        batch_tokens = []
        batch_embeddings = []
        batch_phrase_ids = []
        
        with torch.no_grad():
            outputs = model(**inputs)
            token_embeddings = outputs.last_hidden_state
            
        for j in range(len(batch)):
            absolute_phrase_index = i + j
            input_ids = inputs['input_ids'][j]
            mask = inputs['attention_mask'][j] == 1
            
            valid_input_ids = input_ids[mask]
            valid_embeddings = token_embeddings[j][mask]
            
            tokens_texte = tokenizer.convert_ids_to_tokens(valid_input_ids)
            special_tokens = tokenizer.all_special_tokens
            
            for mot, emb in zip(tokens_texte, valid_embeddings):
                if mot not in special_tokens:
                    batch_tokens.append(mot)
                    batch_embeddings.append(emb.cpu().numpy())
                    batch_phrase_ids.append(absolute_phrase_index) 
        if len(batch_embeddings) > 0:
            yield batch_tokens, np.vstack(batch_embeddings), np.array(batch_phrase_ids)
        else:
            # Retourne des structures vides respectant les types attendus
            yield [], np.array([]), np.array([])


def entrainer_pipeline_if(phrases_train, n_components=50, batch_size=32, model_id="bert-base-multilingual-cased", temp_dir="temp_data"):
    """
    Entraîne les algorithmes de normalisation (Scaler), de réduction (PCA) et de détection (Isolation Forest).
    
    Stratégie technique : Utilisation d'un fichier binaire mappé en mémoire (mmap) pour la matrice d'embeddings.

    Args:
        phrases_train (list of str): Corpus d'entraînement (prescriptions valides).
        n_components (int): Nombre de dimensions cibles pour la PCA.
        batch_size (int): Taille de lot pour l'inférence BERT.
        model_id (str): Identifiant du modèle HuggingFace.
        temp_dir (str): Répertoire de stockage temporaire de la matrice d'entraînement.

    Returns:
        tuple: Composants du pipeline entraîné (modele_if, ipca, scaler, tokenizer, model).
    """
    os.makedirs(temp_dir, exist_ok=True)
    chemin_temp = os.path.join(temp_dir, "embeddings_train_if.npy")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    
    balises = ['[DRUG]', '[DOSE]', '[UNIT]', '[ROUTE]', '[GEN]', '[ADM]', '[BIO]']
    tokenizer.add_tokens(balises)
    model.resize_token_embeddings(len(tokenizer)) 
    
    model = model.to(device)
    model.eval()

    logger.info("Étape 1/3 : Extraction des embeddings BERT...")
    all_embeddings = []

    generateur = iterateur_embeddings_tokens(phrases_train, model_id=model_id, batch_size=batch_size, tokenizer=tokenizer, model=model)
    
    for _, batch_emb, _ in generateur:
        all_embeddings.append(batch_emb)
        
    matrice_brute = np.vstack(all_embeddings)
    np.save(chemin_temp, matrice_brute)
    del all_embeddings, matrice_brute 
    
    logger.info("Étape 2/3 : Ajustement du Scaler et de la PCA...")
    data_mmap = np.load(chemin_temp, mmap_mode='r')
    
    scaler = StandardScaler()
    ipca = IncrementalPCA(n_components=n_components)
    chunk_size = 50000
    
    for i in range(0, data_mmap.shape[0], chunk_size):
        scaler.partial_fit(data_mmap[i:i+chunk_size])
        
    for i in range(0, data_mmap.shape[0], chunk_size):
        chunk_scaled = scaler.transform(data_mmap[i:i+chunk_size])
        ipca.partial_fit(chunk_scaled)

    logger.info("Étape 3/3 : Transformation et entraînement de l'Isolation Forest...")

    matrice_reduite = ipca.transform(scaler.transform(data_mmap))
    
    modele_if = IsolationForest(n_estimators=100, contamination='auto', n_jobs=-1, random_state=42)
    modele_if.fit(matrice_reduite)
    
    
    if os.path.exists(chemin_temp):
        os.remove(chemin_temp)
    
    return modele_if, ipca, scaler, tokenizer, model


def sauvegarder_pipeline_if(modele_if, model_pca, model_scaler, chemin_dossier: str):
    """
    Sérialise le pipeline complet (IF, PCA, Scaler) sur le disque.

    Args:
        modele_if: Modèle Isolation Forest entraîné.
        model_pca: Modèle PCA entraîné.
        model_scaler: Modèle Scaler entraîné.
        chemin_dossier (str): Chemin du dossier cible.

    Returns:
        None
    """
    chemin = Path(chemin_dossier)
    chemin.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model_scaler, chemin / 'scaler_if.joblib')
    joblib.dump(model_pca, chemin / 'pca_if.joblib')
    joblib.dump(modele_if, chemin / 'modele_if.joblib')
    logger.info(f"Pipeline IF sauvegardé sous : {chemin_dossier}")


def charger_pipeline_if(chemin_dossier: str):
    """
    Charge le pipeline complet depuis le disque.

    Args:
        chemin_dossier (str): Chemin du dossier source.

    Returns:
        tuple: (modele_if, model_pca, model_scaler)
    """
    chemin = Path(chemin_dossier)
    
    if not chemin.exists():
        raise FileNotFoundError(f"Le répertoire modèle est introuvable : {chemin_dossier}")
        
    model_scaler = joblib.load(chemin / 'scaler_if.joblib')
    model_pca = joblib.load(chemin / 'pca_if.joblib')
    modele_if = joblib.load(chemin / 'modele_if.joblib')
    
    logger.info(f"Pipeline IF chargé depuis : {chemin_dossier}")
    return modele_if, model_pca, model_scaler


def executer_pipeline_inference_if(df_test, modele_if, model_pca, model_scaler, tokenizer, model_bert, batch_size=32):
    """
    Exécute l'inférence vectorisée sur le jeu de test et agrège les anomalies contextuelles.
    
    Logique : Parcours séquentiel des tokens prédits. Dès qu'une balise structurelle (ex: [DRUG]) 
    est rencontrée, le contexte est mis à jour. Si une anomalie est détectée, elle est 
    affectée au contexte actif en cours.

    Args:
        df_test (pd.DataFrame): Données d'évaluation contenant la colonne 'phrase_clinique'.
        modele_if (IsolationForest): Modèle de détection d'anomalies.
        model_pca (IncrementalPCA): Modèle de réduction dimensionnelle.
        model_scaler (StandardScaler): Modèle de normalisation.
        tokenizer (AutoTokenizer): Tokenizer associé au modèle de langage.
        model_bert (AutoModel): Modèle d'extraction des caractéristiques linguistiques.
        batch_size (int): Taille du lot pour l'inférence BERT.

    Returns:
        pd.DataFrame: DataFrame enrichi avec 'label_pred' (binaire) et 'error_types_pred' (multilabel).
    """
    phrases_test = df_test['phrase_clinique'].tolist()
    
    all_tokens = []
    all_embeddings_list = []
    phrase_ids = []
    
    generateur = iterateur_embeddings_tokens(
        phrases_test, batch_size=batch_size, tokenizer=tokenizer, model=model_bert
    )
    
    for batch_tok, batch_emb, batch_ids in generateur:
        all_tokens.extend(batch_tok)
        all_embeddings_list.append(batch_emb)
        phrase_ids.extend(batch_ids)
        
    if len(all_embeddings_list) == 0:
        df_resultat = df_test.copy()
        df_resultat['label_pred'] = 0
        df_resultat['error_types_pred'] = "none"
        return df_resultat

    embeddings = np.vstack(all_embeddings_list)

    emb_scaled = model_scaler.transform(embeddings)
    emb_reduced = model_pca.transform(emb_scaled)
    preds_mots = modele_if.predict(emb_reduced)

    tag_to_label = {
        "[DRUG]": "drug",
        "[ROUTE]": "route",
        "[UNIT]": "unit_dosage",
        "[DOSE]": "dosage",
    }
    
    y_pred_phrases = np.zeros(len(phrases_test), dtype=int)
    detected_categories_per_phrase = {i: set() for i in range(len(phrases_test))}
    
    current_context = "none"
    current_phrase_id = -1

    for mot, pred, p_id in zip(all_tokens, preds_mots, phrase_ids):
        if p_id != current_phrase_id:
            current_context = "none"
            current_phrase_id = p_id

        if mot in tag_to_label:
            current_context = tag_to_label[mot]
            continue 
            
        if pred == -1:
            y_pred_phrases[p_id] = 1 
            if current_context != "none":
                detected_categories_per_phrase[p_id].add(current_context)

    labels_pred_phrases = []
    for i in range(len(phrases_test)):
        categories = detected_categories_per_phrase[i]
        if len(categories) > 0:
            labels_pred_phrases.append("|".join(list(categories)))
        else:
            labels_pred_phrases.append("none")

    df_resultat = df_test.copy()
    df_resultat['label_vrai'] = df_resultat['nb_errors'].apply(lambda x: 1 if x > 0 else 0)
    
    df_resultat['error_types_pred'] = labels_pred_phrases
    df_resultat['label_pred'] = y_pred_phrases

    return df_resultat


def analyser_phrase_if(phrase, modele_if, pca_model, scaler_model, tokenizer, bert_model):
    """
    Analyse unitaire d'une phrase pour le débogage ou l'explicabilité manuelle.

    Args:
        phrase (str): Phrase clinique à analyser.
        modele_if: Modèle Isolation Forest.
        pca_model: Modèle PCA.
        scaler_model: Modèle Scaler.
        tokenizer: Tokenizer BERT.
        bert_model: Modèle BERT.

    Returns:
        list: Liste de dictionnaires contenant l'analyse par mot.
    """
    generateur = iterateur_embeddings_tokens([phrase], batch_size=1, tokenizer=tokenizer, model=bert_model)
    
    try:
        mots, embeddings, _ = next(generateur)
    except StopIteration:
        return []

    if len(embeddings) == 0:
        return []

    emb_scaled = scaler_model.transform(embeddings)
    emb_reduced = pca_model.transform(emb_scaled)
    
    predictions = modele_if.predict(emb_reduced)
    scores = modele_if.decision_function(emb_reduced)
    
    resultats = []
    for mot, pred, score in zip(mots, predictions, scores):
        statut = "Anomalie" if pred == -1 else "Valide"
        resultats.append({
            "mot": mot, 
            "statut": statut, 
            "score_normalite": round(score, 4)
        })
        
    return resultats