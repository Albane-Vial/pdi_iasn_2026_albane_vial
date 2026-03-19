import os
import torch
import itertools
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, DebertaV2Config, DebertaV2ForMaskedLM, DebertaV2ForTokenClassification, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader, Dataset as TorchDataset

def modele_old(df_final):
    # 1. Configuration matérielle stricte
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Calcul alloué sur : {device}")

    # 2. Initialisation du Tokenizer et du Vocabulaire
    model_checkpoint = "microsoft/deberta-v3-small"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)

    special_tokens = ["[GEN]", "[AGE]", "[ADM]", "[DX]", "[BIO]", "[DRUG]", "[DOSE]", "[UNIT]", "[ROUTE]"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # 3. Calcul mathématique de la longueur maximale (max_length)
    print("Analyse de la distribution des tokens en cours...")
    # On encode chaque phrase pour compter son nombre réel de tokens
    longueurs = df_final['phrase_clinique'].apply(lambda x: len(tokenizer.encode(x)))

    # On identifie le 99e percentile pour ignorer les valeurs aberrantes (outliers) extrêmes
    longueur_99_percentile = int(np.percentile(longueurs, 99))
    max_length_calculable = min(longueur_99_percentile + 5, 512) # Marge de sécurité de 5, plafond absolu à 512
    print(f"Paramètre max_length défini dynamiquement sur : {max_length_calculable}")

    # 4. Préparation du Dataset
    train_ds = Dataset.from_pandas(df_final[['phrase_clinique']])

    def tokenize_func(examples):
        return tokenizer(examples["phrase_clinique"], truncation=True, padding="max_length", max_length=max_length_calculable)

    tokenized_train = train_ds.map(tokenize_func, batched=True)

    # Nettoyage
    cols_to_remove = ["phrase_clinique"]
    tokenized_train_clean = tokenized_train.remove_columns([c for c in cols_to_remove if c in tokenized_train.column_names])
    tokenized_train_clean.set_format("torch")

    # 5. Instanciation et Redimensionnement des Modèles
    v_size = len(tokenizer)

    gen_config = DebertaV2Config.from_pretrained("microsoft/deberta-v3-xsmall")
    gen_config.vocab_size = v_size
    generator = DebertaV2ForMaskedLM(gen_config)
    generator.resize_token_embeddings(v_size)
    generator.to(device)

    disc_config = DebertaV2Config.from_pretrained("microsoft/deberta-v3-small", num_labels=2)
    disc_config.vocab_size = v_size
    discriminator = DebertaV2ForTokenClassification(disc_config)
    discriminator.resize_token_embeddings(v_size)
    discriminator.to(device)

    # 6. Surcharge de l'API Trainer
    class DebertaV3RTDTrainer(Trainer):
        def create_optimizer(self):
            if self.optimizer is None:
                disc_params = [p for p in self.model.parameters() if p.requires_grad]
                gen_params = [p for p in generator.parameters() if p.requires_grad]
                combined_params = itertools.chain(disc_params, gen_params)
                optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
                self.optimizer = optimizer_cls(combined_params, **optimizer_kwargs)
            return self.optimizer

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            gen_outputs = generator(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"]
            )
            
            with torch.no_grad():
                probs = torch.softmax(gen_outputs.logits, dim=-1)
                sampled_tokens = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(probs.shape[0], -1)
                sampled_tokens = torch.clamp(sampled_tokens, max=v_size - 1)
                
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

    # 7. Configuration de l'entraînement (Adaptée pour GPU Kaggle)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir="/teamspace/studios/this_studio/deberta_rtd_medical_checkpoints_old",
        per_device_train_batch_size=2,          # RÉDUCTION STRICHTE : de 4 à 2
        gradient_accumulation_steps=16,         # COMPENSATION STRICHTE : de 8 à 16 (2 * 16 = 32)
        fp16=False,                             
        num_train_epochs=5,
        learning_rate=5e-5,
        remove_unused_columns=False,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none"
    )

    trainer = DebertaV3RTDTrainer(
        model=discriminator,
        args=training_args,
        train_dataset=tokenized_train_clean,
        data_collator=data_collator,
    )

    print("Début de l'entraînement...")
    generator.train() # Force le générateur en mode entraînement
    trainer.train()
    output_dir = "/teamspace/studios/this_studio/"
    chemin_sauvegarde = os.path.join(output_dir, "modele_mimic_debertaV3_old")
    trainer.save_model(chemin_sauvegarde)
    tokenizer.save_pretrained(chemin_sauvegarde)
    return chemin_sauvegarde

def preparer_donnees_test(df_test, colonnes_a_masquer=['error_types', 'nb_errors']):
    """Isole les labels pour éviter la fuite de données (Data Leakage)."""
    df_labels_caches = df_test[colonnes_a_masquer].copy()
    df_sans_labels = df_test.drop(columns=colonnes_a_masquer)
    return df_sans_labels, df_labels_caches

class DatasetInference(TorchDataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        # Retourne un dictionnaire de tenseurs (input_ids, attention_mask) pour l'index donné
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

def executer_predictions(df_entree, config, batch_size=16):
    """Charge le modèle et exécute la prédiction sur CPU ou GPU."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 1. Chargement depuis le chemin sauvegardé
    tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint"])
    modele = DebertaV2ForTokenClassification.from_pretrained(config["chemin_modele"])

#    tokenizer = AutoTokenizer.from_pretrained(config["chemin_modele"])
#    modele = DebertaV2ForTokenClassification.from_pretrained(config["chemin_modele"])
    modele.to(device)
    modele.eval() # Désactive le Dropout et BatchNorm

    # 2. Tokenisation
    encodings = tokenizer(
        df_entree['phrase_clinique'].tolist(),
        truncation=True,
        padding="max_length",
        max_length=config["max_length"],
        return_tensors="pt"
    )
    
    dataloader = DataLoader(DatasetInference(encodings), batch_size=batch_size)
    predictions_binaires = []
    
    # 3. Boucle de prédiction
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            sorties = modele(input_ids=input_ids, attention_mask=attention_mask)
            logits = sorties.logits
            
            # Argmax pour obtenir la classe prédite par token (0: original, 1: remplacé)
            preds_tokens = torch.argmax(logits, dim=-1)
            
            # Masquage des tokens de padding pour ne pas fausser l'analyse
            masque_actif = attention_mask == 1
            
            for i in range(preds_tokens.shape[0]):
                # On extrait uniquement les prédictions des vrais tokens de la phrase
                tokens_valides = preds_tokens[i][masque_actif[i]]
                
                # Agrégation binaire : si au moins 1 token est classé comme "erreur" (1), la phrase est en erreur
                if 1 in tokens_valides:
                    predictions_binaires.append(1)
                else:
                    predictions_binaires.append(0)

    return predictions_binaires


def executer_pipeline_inference_old(df_test, config):
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
