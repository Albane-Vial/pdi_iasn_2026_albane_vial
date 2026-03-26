import os
import gc
import torch
import itertools
import numpy as np
from transformers import (
    AutoTokenizer, DebertaV2Config, DebertaV2ForMaskedLM, 
    DebertaV2ForTokenClassification, Trainer, TrainingArguments, 
    DataCollatorForLanguageModeling
)
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset 
import re

def configurer_environnement():
    """
    Optimise l'utilisation des ressources matérielles et nettoie la mémoire
    
    Returns:
        torch.device: Périphérique de calcul (CUDA ou CPU).
    """    
    torch.cuda.empty_cache()
    gc.collect()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Calcul alloué sur : {device}")
    return device

def initialiser_tokenizer(model_checkpoint, special_tokens):
    """
    Initialise le tokenizer et enregistre les balises cliniques personnalisées.
    
    Args:
        model_checkpoint (str): chemin local du modèle.
        special_tokens (list): Liste des balises (ex: [DRUG], [DOSE]).
        
    Returns:
        AutoTokenizer: Tokenizer configuré avec le vocabulaire étendu.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    return tokenizer

def calculer_longueur_maximale(df, tokenizer, colonne_texte="phrase_clinique", percentile=99, marge=5, plafond=512):
    """
    Détermine la longueur optimale des séquences pour minimiser le padding inutile.
    
    Args:
        df (pd.DataFrame): Données d'entraînement.
        percentile (int): Seuil de couverture de la population des phrases.
        
    Returns:
        int: Longueur maximale calculée.
    """
    print("Analyse de la distribution des tokens en cours...")
    longueurs = df[colonne_texte].apply(lambda x: len(tokenizer.encode(str(x))))
    longueur_percentile = int(np.percentile(longueurs, percentile))
    max_length = min(longueur_percentile + marge, plafond)
    print(f"Paramètre max_length défini dynamiquement sur : {max_length}")
    return max_length

def preparer_dataset(df, tokenizer, max_length, colonne_texte="phrase_clinique"):
    """
    Convertit un DataFrame Pandas en Dataset HuggingFace formaté pour PyTorch.
    
    Args:
        df (pd.DataFrame): Source de données.
        max_length (int): Longueur de troncature/padding.
        
    Returns:
        datasets.Dataset: Dataset tokenisé et prêt pour le Trainer.
    """    
    dataset = Dataset.from_pandas(df[[colonne_texte]])
    
    def tokenize_func(examples):
        return tokenizer(examples[colonne_texte], truncation=True, padding="max_length", max_length=max_length)
    
    tokenized_dataset = dataset.map(tokenize_func, batched=True)
    cols_to_remove = [colonne_texte]
    tokenized_dataset = tokenized_dataset.remove_columns([c for c in cols_to_remove if c in tokenized_dataset.column_names])
    tokenized_dataset.set_format("torch")
    return tokenized_dataset

def initialiser_modeles(vocab_size, device, gen_checkpoint="microsoft/deberta-v3-xsmall", disc_checkpoint="microsoft/deberta-v3-small"):
    """
    Instancie et configure les composants de l'architecture RTD (Replaced Token Detection).

    Cette fonction initialise deux modèles distincts pour l'apprentissage auto-supervisé :
    1. Un Générateur (MLM) chargé de corrompre les séquences d'entrée.
    2. Un Discriminateur (Classification de tokens) chargé d'identifier les tokens substitués.
    Les deux modèles subissent une extension de leurs couches d'embeddings pour supporter
    le dictionnaire clinique enrichi (balises personnalisées).

    Args:
        vocab_size (int): Dimension totale du vocabulaire (incluant les tokens spéciaux).
        device (torch.device): Périphérique cible pour le calcul (CPU ou CUDA).
        gen_checkpoint (str):Chemin local pour le générateur.
            Par défaut : "microsoft/deberta-v3-xsmall".
        disc_checkpoint (str): Chemin local pour le discriminateur.
            Par défaut : "microsoft/deberta-v3-small".

    Returns:
        tuple (DebertaV2ForMaskedLM, DebertaV2ForTokenClassification): 
            Couple (générateur, discriminateur) initialisé et transféré sur le périphérique cible.
    """
    gen_config = DebertaV2Config.from_pretrained(gen_checkpoint)
    gen_config.vocab_size = vocab_size
    generator = DebertaV2ForMaskedLM(gen_config)
    generator.resize_token_embeddings(vocab_size)
    generator.to(device)

    disc_config = DebertaV2Config.from_pretrained(disc_checkpoint, num_labels=2)
    disc_config.vocab_size = vocab_size
    discriminator = DebertaV2ForTokenClassification(disc_config)
    discriminator.resize_token_embeddings(vocab_size)
    discriminator.to(device)
    
    return generator, discriminator

class DebertaV3RTDTrainer(Trainer):
    """
    Trainer personnalisé orchestrant l'entraînement conjoint du Générateur et du Discriminateur.

    Cette classe surcharge les méthodes natives de la bibliothèque 'transformers' pour 
    implémenter la tâche de pré-entraînement RTD (Replaced Token Detection). 
    Le processus se déroule en deux étapes par passe :
    1. Le générateur (MLM) prédit des tokens pour combler les masques.
    2. Le discriminateur tente de distinguer les tokens originaux des tokens échantillonnés.
    """
    def __init__(self, generator, vocab_size, *args, **kwargs):
        """
        Initialise le trainer avec un générateur externe.

        Args:
            generator (nn.Module): Le modèle DeBERTa MaskedLM utilisé comme générateur.
            vocab_size (int): Taille du vocabulaire pour sécuriser l'échantillonnage.
            *args, **kwargs: Arguments standards passés à la classe Trainer.
        """
        super().__init__(*args, **kwargs)
        self.generator = generator
        self.vocab_size = vocab_size

    def create_optimizer(self):
        """
        Initialise un optimiseur unique pour les deux modèles.

        Surcharge la méthode native pour chaîner les paramètres du discriminateur 
        (self.model) et du générateur (self.generator) afin qu'ils soient mis à jour
        simultanément durant l'entraînement.
        
        Returns:
            torch.optim.Optimizer: L'optimiseur configuré pour l'apprentissage conjoint.
        """
        if self.optimizer is None:
            disc_params = [p for p in self.model.parameters() if p.requires_grad]
            gen_params = [p for p in self.generator.parameters() if p.requires_grad]
            combined_params = itertools.chain(disc_params, gen_params)
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(combined_params, **optimizer_kwargs)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Calcule la fonction de perte multi-tâche (Génération + Détection).

        Logique métier :
        1. Passe de génération : Le générateur prédit les tokens masqués.
        2. Échantillonnage : Les tokens sont tirés de la distribution de probabilité 
           du générateur pour corrompre la séquence d'entrée.
        3. Étiquetage RTD 
        4. Passe de discrimination : Le modèle (discriminateur) classifie chaque token.

        Args:
            model (nn.Module): Le discriminateur.
            inputs (dict): Tenseurs d'entrée (input_ids, attention_mask, labels).
            
        Returns:
            torch.Tensor: Perte totale pondérée.
        """
        gen_outputs = self.generator(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"]
        )
        
        with torch.no_grad():
            probs = torch.softmax(gen_outputs.logits, dim=-1)
            sampled_tokens = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(probs.shape[0], -1)
            sampled_tokens = torch.clamp(sampled_tokens, max=self.vocab_size - 1)
            
            mask_indices = inputs["labels"] != -100
            corrupted_input_ids = inputs["input_ids"].clone()
            corrupted_input_ids[mask_indices] = sampled_tokens[mask_indices]
            
            disc_labels = (corrupted_input_ids != inputs["labels"]) & mask_indices
            disc_labels = disc_labels.long().to(inputs["input_ids"].device)

        disc_outputs = model(
            input_ids=corrupted_input_ids,
            attention_mask=inputs["attention_mask"],
            labels=disc_labels
        )
        
        total_loss = gen_outputs.loss + 50.0 * disc_outputs.loss
        return (total_loss, disc_outputs) if return_outputs else total_loss


def lancer_entrainement(tokenizer, dataset, generator, discriminator, vocab_size, chemin_sauvegarde="modeles_sauvegardes/debertaV3"):
    """
    Configure les hyperparamètres et orchestre le cycle d'entraînement RTD.

    Cette fonction initialise les arguments d'entraînement (batch size, accumulation de gradient, etc.)
    et instancie le Trainer personnalisé. Elle assure également la persistance du modèle
    et du tokenizer sur le disque après convergence.

    Args:
        tokenizer (AutoTokenizer): Tokenizer avec vocabulaire étendu.
        dataset (Dataset): Dataset HuggingFace tokenisé.
        generator (nn.Module): Modèle MaskedLM (Générateur).
        discriminator (nn.Module): Modèle TokenClassification (Discriminateur).
        vocab_size (int): Taille totale du vocabulaire cible.
        chemin_sauvegarde (str): Répertoire de destination pour les artefacts.

    Returns:
        str: Le chemin d'accès vers le modèle sauvegardé.
    """    
    os.makedirs(chemin_sauvegarde, exist_ok=True)
    dossier_checkpoints = os.path.join(chemin_sauvegarde, "checkpoints_intermediaires")
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    training_args = TrainingArguments(
        output_dir=dossier_checkpoints, 
        per_device_train_batch_size=8,          
        gradient_accumulation_steps=4,          
        fp16=True,                              
        num_train_epochs=2,                     
        learning_rate=5e-5,
        remove_unused_columns=False,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none"
    )
    
    trainer = DebertaV3RTDTrainer(
        generator=generator,     
        vocab_size=vocab_size,
        model=discriminator,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("Début de l'entraînement...")
    generator.train() 
    trainer.train()

    trainer.save_model(chemin_sauvegarde)
    tokenizer.save_pretrained(chemin_sauvegarde)
    
    print(f"Sauvegarde terminée. Modèle et tokenizer exportés dans : {chemin_sauvegarde}")
    return chemin_sauvegarde

class DatasetInference(TorchDataset):
    """Classe utilitaire PyTorch pour gérer les tenseurs par lots (batches)."""
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

def preparer_donnees_test(df_test, colonnes_a_masquer=['error_types', 'nb_errors']):
    """
    Sépare les étiquettes de vérité des données pour l'inférence.

    Args:
        df_test (pd.DataFrame): Dataset contenant toutes les colonnes.
        colonnes_a_masquer (list): Colonnes à masquer lors de la prédiction.

    Returns:
        tuple: (DataFrame sans labels, DataFrame contenant uniquement les labels)
    """
    df_labels_caches = df_test[colonnes_a_masquer].copy()
    df_sans_labels = df_test.drop(columns=colonnes_a_masquer)
    return df_sans_labels, df_labels_caches

def executer_predictions_contextuelles(df_entree, config, batch_size=16):
    """
    Réalise l'inférence par lots et décode les erreurs par analyse de contexte.

    Le modèle identifie les tokens corrompus, et cette fonction utilise les balises 
    cliniques ([DRUG], [DOSE], etc.) comme ancres contextuelles pour imputer 
    l'erreur au champ sémantique correspondant.

    Args:
        df_entree (pd.DataFrame): Données brutes de test.
        config (dict): Paramètres de configuration (chemins, max_length).
        batch_size (int): Taille des lots pour l'inférence GPU.

    Returns:
        tuple: (Liste des flags binaires 0/1, Liste des catégories d'erreurs détectées).
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    chemin_local = config["chemin_modele"]
    tokenizer = AutoTokenizer.from_pretrained(chemin_local)
    modele = DebertaV2ForTokenClassification.from_pretrained(chemin_local)
    modele.to(device)
    modele.eval()
    
    dataset_hf = Dataset.from_pandas(df_entree[['phrase_clinique']])
    
    def tokenize_func(examples):
        return tokenizer(examples['phrase_clinique'], truncation=True, padding="max_length", max_length=config["max_length"])
    
    tokenized_dataset = dataset_hf.map(tokenize_func, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size)
    
    predictions_binaires = []
    predictions_labels = []
    
    tag_to_label = {
        "[DRUG]": "drug",
        "[ROUTE]": "route",
        "[UNIT]": "unit_dosage", 
        "[DOSE]": "dosage",
        "[GEN]": "autre",
        "[AGE]": "autre",
        "[ADM]": "autre",
        "[DX]": "autre",
        "[BIO]": "autre"
    }
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            sorties = modele(input_ids=input_ids, attention_mask=attention_mask)
            preds_tokens = torch.argmax(sorties.logits, dim=-1)
            
            for i in range(preds_tokens.shape[0]):
                tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
                preds = preds_tokens[i].cpu().tolist()
                mask = attention_mask[i].cpu().tolist()
                
                phrase_has_error = 0
                detected_categories = set() 
                current_context = "none"
                
                for token, pred, m in zip(tokens, preds, mask):
                    if m == 0: 
                        continue 
                    
                    if token in tag_to_label:
                        current_context = tag_to_label[token]
                        continue 
                        
                    if pred == 1:
                        phrase_has_error = 1
                        if current_context != "none":
                            detected_categories.add(current_context)
                
                predictions_binaires.append(phrase_has_error)
                
                if len(detected_categories) > 0:
                    predictions_labels.append("|".join(list(detected_categories)))
                else:
                    predictions_labels.append("none")
                    
    return predictions_binaires, predictions_labels



def executer_pipeline_entrainement(df_final, dossier_cible= "modeles_sauvegardes/debertaV3"):
    """
    Orchestre le cycle complet d'apprentissage, de l'initialisation à l'export final.

    Cette fonction pilote séquentiellement :
    1. La configuration de l'infrastructure de calcul (GPU/CPU).
    2. L'extension du dictionnaire par les tokens sémantiques cliniques.
    3. L'analyse statistique de la longueur des phrases pour optimiser la mémoire.
    4. L'instanciation de l'architecture double (Générateur/Discriminateur).
    5. L'exécution de l'entraînement par détection de tokens substitués (RTD).

    Args:
        df_final (pd.DataFrame): Jeu de données nettoyé et formaté.
        dossier_cible (str): Répertoire de stockage pour le modèle entraîné.

    Returns:
        dict: Configuration d'exportation contenant les métadonnées nécessaires à l'inférence.
    """
    device = configurer_environnement()
    
    special_tokens = ["[GEN]", "[AGE]", "[ADM]", "[DX]", "[BIO]", "[DRUG]", "[DOSE]", "[UNIT]", "[ROUTE]"]
    tokenizer = initialiser_tokenizer("microsoft/deberta-v3-small", special_tokens)
    
    max_length = calculer_longueur_maximale(df_final, tokenizer)
    dataset_tokenise = preparer_dataset(df_final, tokenizer, max_length)
    
    vocab_size = len(tokenizer)
    generator, discriminator = initialiser_modeles(vocab_size, device)
    
    chemin_modele = lancer_entrainement(
        tokenizer, dataset_tokenise, generator, discriminator, vocab_size, chemin_sauvegarde=dossier_cible
    )
    config_export = {
        "chemin_modele": chemin_modele,
        "max_length": max_length,
        "vocab_size": vocab_size,
        "special_tokens": special_tokens
    }
    
    return config_export

def executer_pipeline_inference(df_test, config):
    """
    Pilote la phase d'évaluation sur le jeu de test et standardise les prédictions.

    Args:
        df_test (pd.DataFrame): Dataset de test contenant les prescriptions bruitées.
        config (dict): Dictionnaire de configuration issu de la phase d'entraînement.

    Returns:
        pd.DataFrame: DataFrame enrichi des prédictions binaires et multi-labels.
    """
    colonnes_labels = ['error_types', 'nb_errors']
    
    df_sans_labels, df_labels_caches = preparer_donnees_test(df_test, colonnes_labels)
    
    preds_binaires, preds_labels = executer_predictions_contextuelles(df_sans_labels, config)
    
    df_resultat = df_test.copy() 
    df_resultat['label_pred'] = preds_binaires
    df_resultat['label_vrai'] = df_resultat['nb_errors'].apply(lambda x: 1 if x > 0 else 0)
    
    df_resultat['error_types_pred'] = preds_labels
    
    if 'error_types' in df_resultat.columns:
        df_resultat['error_types'] = df_resultat['error_types'].astype(str)
        df_resultat['error_types'] = df_resultat['error_types'].str.replace('sous_dosage', 'dosage')
        df_resultat['error_types'] = df_resultat['error_types'].str.replace('sur_dosage', 'dosage')
        
        import re
        df_resultat['error_types'] = df_resultat['error_types'].apply(
            lambda x: re.sub(r'\bunit\b', 'unit_dosage', x)
        )

    return df_resultat

def charger_modele_et_tokenizer(chemin_modele: str, device: torch.device):
    """
    Charge le modèle et le tokenizer depuis le disque pour l'inférence ou l'explicabilité.
    
    Args:
        chemin_modele (str): Chemin vers le dossier contenant les fichiers sauvegardés.
        device (torch.device): Périphérique cible (CPU ou CUDA).
        
    Returns:
        tuple: (Modèle DebertaV2ForTokenClassification, AutoTokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(chemin_modele)
    modele = DebertaV2ForTokenClassification.from_pretrained(chemin_modele)
    
    modele.to(device)
    modele.eval()
    
    return modele, tokenizer