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

def configurer_environnement():
    """Purge la VRAM et configure le périphérique cible."""
    torch.cuda.empty_cache()
    gc.collect()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Calcul alloué sur : {device}")
    return device

def initialiser_tokenizer(model_checkpoint, special_tokens):
    """Charge le tokenizer et ajoute les tokens spéciaux."""
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    return tokenizer

def calculer_longueur_maximale(df, tokenizer, colonne_texte="phrase_clinique", percentile=99, marge=5, plafond=512):
    """Calcule dynamiquement la longueur maximale de séquence à partir du dataframe."""
    print("Analyse de la distribution des tokens en cours...")
    longueurs = df[colonne_texte].apply(lambda x: len(tokenizer.encode(str(x))))
    longueur_percentile = int(np.percentile(longueurs, percentile))
    max_length = min(longueur_percentile + marge, plafond)
    print(f"Paramètre max_length défini dynamiquement sur : {max_length}")
    return max_length

def preparer_dataset(df, tokenizer, max_length, colonne_texte="phrase_clinique"):
    """Tokenize et formate le dataset pandas au format d'entrée PyTorch."""
    dataset = Dataset.from_pandas(df[[colonne_texte]])
    
    def tokenize_func(examples):
        return tokenizer(examples[colonne_texte], truncation=True, padding="max_length", max_length=max_length)
    
    tokenized_dataset = dataset.map(tokenize_func, batched=True)
    cols_to_remove = [colonne_texte]
    tokenized_dataset = tokenized_dataset.remove_columns([c for c in cols_to_remove if c in tokenized_dataset.column_names])
    tokenized_dataset.set_format("torch")
    return tokenized_dataset

def initialiser_modeles(vocab_size, device, gen_checkpoint="microsoft/deberta-v3-xsmall", disc_checkpoint="microsoft/deberta-v3-small"):
    """Instancie et redimensionne le générateur et le discriminateur selon le nouveau vocabulaire."""
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
    def __init__(self, generator, vocab_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator = generator
        self.vocab_size = vocab_size

    def create_optimizer(self):
        if self.optimizer is None:
            disc_params = [p for p in self.model.parameters() if p.requires_grad]
            gen_params = [p for p in self.generator.parameters() if p.requires_grad]
            combined_params = itertools.chain(disc_params, gen_params)
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(combined_params, **optimizer_kwargs)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
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

def lancer_entrainement(tokenizer, dataset, generator, discriminator, vocab_size, output_dir="/teamspace/studios/this_studio/"):
    """Configure les hyperparamètres, instancie le Trainer et exécute l'entraînement."""
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "deberta_rtd_medical_checkpoints"),
        per_device_train_batch_size=8,          # Modification : passage de 2 à 8
        gradient_accumulation_steps=4,          # Modification : passage de 16 à 4
        fp16=True,                              # Modification : activation de la précision 16 bits
        num_train_epochs=2,                     # Modification : passage de 5 à 2 époques
        learning_rate=5e-5,
        remove_unused_columns=False,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none"
    )
    trainer = DebertaV3RTDTrainer(
        generator=generator,     # Passage explicite
        vocab_size=vocab_size,   # Passage explicite
        model=discriminator,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("Début de l'entraînement...")
    generator.train() 
    trainer.train()

    chemin_sauvegarde = os.path.join(output_dir, "modele_mimic_debertaV3")
    trainer.save_model(chemin_sauvegarde)
    tokenizer.save_pretrained(chemin_sauvegarde)
    print("Sauvegarde terminée. Le processus peut se fermer proprement.")
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
    """Isole les labels pour éviter la fuite de données (Data Leakage)."""
    df_labels_caches = df_test[colonnes_a_masquer].copy()
    df_sans_labels = df_test.drop(columns=colonnes_a_masquer)
    return df_sans_labels, df_labels_caches

def executer_predictions_contextuelles(df_entree, config, batch_size=16):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    chemin_local = config["chemin_modele"]
    tokenizer = AutoTokenizer.from_pretrained(chemin_local)
    modele = DebertaV2ForTokenClassification.from_pretrained(chemin_local)
    modele.to(device)
    modele.eval()
    
    # Utilisation robuste du Dataset HuggingFace
    dataset_hf = Dataset.from_pandas(df_entree[['phrase_clinique']])
    
    def tokenize_func(examples):
        return tokenizer(examples['phrase_clinique'], truncation=True, padding="max_length", max_length=config["max_length"])
    
    tokenized_dataset = dataset_hf.map(tokenize_func, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size)
    
    predictions_binaires = []
    predictions_labels = []
    
    # Mapping des balises vers le vocabulaire exact de votre évaluateur MultiLabelBinarizer
    tag_to_label = {
        "[DRUG]": "drug",
        "[ROUTE]": "route",
        "[UNIT]": "unit_dosage", 
        "[DOSE]": "dosage" # Regroupe sous_dosage et sur_dosage
    }
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            sorties = modele(input_ids=input_ids, attention_mask=attention_mask)
            preds_tokens = torch.argmax(sorties.logits, dim=-1)
            
            for i in range(preds_tokens.shape[0]):
                # Reconversion des identifiants en texte pour lire les balises
                tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
                preds = preds_tokens[i].cpu().tolist()
                mask = attention_mask[i].cpu().tolist()
                
                phrase_has_error = 0
                detected_categories = set() # Set pour éviter les doublons si plusieurs tokens d'un même champ sont en erreur
                current_context = "none"
                
                for token, pred, m in zip(tokens, preds, mask):
                    if m == 0: 
                        continue # On ignore les tokens de padding
                    
                    # 1. Mise à jour du contexte si on croise une balise
                    if token in tag_to_label:
                        current_context = tag_to_label[token]
                        continue 
                        
                    # 2. Si le token courant est une erreur, on l'associe au contexte actif
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



def executer_pipeline_entrainement(df_final):
    """Fonction principale orchestrant l'appel séquentiel de tous les modules."""
    device = configurer_environnement()
    
    special_tokens = ["[GEN]", "[AGE]", "[ADM]", "[DX]", "[BIO]", "[DRUG]", "[DOSE]", "[UNIT]", "[ROUTE]"]
    tokenizer = initialiser_tokenizer("microsoft/deberta-v3-small", special_tokens)
    
    max_length = calculer_longueur_maximale(df_final, tokenizer)
    dataset_tokenise = preparer_dataset(df_final, tokenizer, max_length)
    
    vocab_size = len(tokenizer)
    generator, discriminator = initialiser_modeles(vocab_size, device)
    
    chemin_modele = lancer_entrainement(
        tokenizer, dataset_tokenise, generator, discriminator, vocab_size # Remplacement de df_final par dataset_tokenise
    ) 
    config_export = {
        "chemin_modele": chemin_modele,
        "max_length": max_length,
        "vocab_size": vocab_size,
        "special_tokens": special_tokens
    }
    
    return config_export


def executer_pipeline_inference(df_test, config):
    colonnes_labels = ['error_types', 'nb_errors']
    
    df_sans_labels, df_labels_caches = preparer_donnees_test(df_test, colonnes_labels)
    
    # Récupération simultanée des probabilités binaires et du typage des erreurs
    preds_binaires, preds_labels = executer_predictions_contextuelles(df_sans_labels, config)
    
    df_resultat = df_test.copy() 
    df_resultat['label_pred'] = preds_binaires
    df_resultat['label_vrai'] = df_resultat['nb_errors'].apply(lambda x: 1 if x > 0 else 0)
    
    # Insertion des labels multi-classes extraits dynamiquement
    df_resultat['error_types_pred'] = preds_labels
    
    # Standardisation de la vérité terrain :
    # Le modèle ne sachant pas distinguer "sous" ou "sur" dosage (juste une anomalie de nombre),
    # on regroupe ces labels sous l'étiquette unique "dosage" pour que l'évaluation soit juste.
    if 'error_types' in df_resultat.columns:
        df_resultat['error_types'] = df_resultat['error_types'].astype(str)
        df_resultat['error_types'] = df_resultat['error_types'].str.replace('sous_dosage', 'dosage')
        df_resultat['error_types'] = df_resultat['error_types'].str.replace('sur_dosage', 'dosage')
        
        # Homogénéisation des noms générés ("unit" vs "unit_dosage")
        # Remplace strictement "unit" (si isolé ou avec séparateur) par "unit_dosage"
        import re
        df_resultat['error_types'] = df_resultat['error_types'].apply(
            lambda x: re.sub(r'\bunit\b', 'unit_dosage', x)
        )

    return df_resultat

    """Orchestre le retrait des labels, la prédiction, et la reconstruction du DataFrame."""
    colonnes_labels = ['error_types', 'nb_errors']
    
    df_sans_labels, df_labels_caches = preparer_donnees_test(df_test, colonnes_labels)
    
    predictions_binaires = executer_predictions(df_sans_labels, config)
    
    # Reconstruction du DataFrame final
    df_resultat = df_test.copy() # On repart de l'original pour garantir l'intégrité
    df_resultat['label_pred'] = predictions_binaires
    
    # Création de la colonne vraie cible binaire pour l'évaluation future
    df_resultat['label_vrai'] = df_resultat['nb_errors'].apply(lambda x: 1 if x > 0 else 0)
    
    # REMARQUE : Logique multi-label manquante (expliquée dans l'analyse critique)
    df_resultat['error_types_pred'] = df_resultat['label_pred'].apply(lambda x: "erreur_detectee" if x == 1 else "none")

    return df_resultat
