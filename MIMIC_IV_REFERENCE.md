# MIMIC-IV v2.1 — Complete Reference

Source: Beth Israel Deaconess Medical Center (BIDMC), Boston, MA  
Covers: 2008–2019 (~300k patients, ~430k hospital admissions)  
Official docs: https://mimic.mit.edu/docs/iv/

> **Local path**: `/home/lepagnol/Projets/mimic_albanevial/mimic-iv-2.1/`  
> Files are **plain uncompressed CSVs** (not .csv.gz).

---

## Key identifiers (shared across all tables)

| Identifier   | Scope           | Description                                      |
|--------------|-----------------|--------------------------------------------------|
| `subject_id` | Patient          | Unique per patient, consistent across all tables |
| `hadm_id`    | Hospital stay    | Unique per hospital admission                    |
| `stay_id`    | ICU stay         | Unique per ICU stay (derived from transfers)     |

> **Time shift**: All dates are shifted into the future for de-identification.  
> Use `anchor_year` + `anchor_year_group` from `patients` to estimate the real year.

---

## Module overview

```
MIMIC-IV v2.1 (local download)
├── hosp/        Hospital-wide EHR (21 tables)
└── icu/         ICU clinical info system — MetaVision (8 tables)

⚠️ ed/ module (Emergency Department) NOT downloaded locally.

Separate datasets (same subject_id, not downloaded):
├── MIMIC-CXR    Chest X-rays
├── MIMIC-IV-Note Clinical notes (discharge, radiology)
└── MIMIC-IV-ECG ECG waveforms
```

---

## MODULE: hosp

All data from the hospital-wide EHR. Not just ICU patients — includes outpatient labs too.

### patients
> One row per patient. The starting point for every analysis.

| Column             | Type        | Description                                              |
|--------------------|-------------|----------------------------------------------------------|
| `subject_id`       | INT PK      | Unique patient identifier                                |
| `gender`           | VARCHAR(1)  | Genotypical sex: `M` or `F`                              |
| `anchor_age`       | INT         | Patient age in `anchor_year` (capped at 91 if > 89)      |
| `anchor_year`      | INT         | Shifted year for the patient (de-identified)             |
| `anchor_year_group`| VARCHAR     | True year range, e.g. `"2008 - 2010"`                    |
| `dod`              | TIMESTAMP   | Date of death (NULL if alive; censored 1 yr post-discharge)|

### admissions
> One row per hospital admission (`hadm_id`). Defines every hospital stay.

| Column                | Type        | Description                                             |
|-----------------------|-------------|---------------------------------------------------------|
| `subject_id`          | INT         | Patient                                                 |
| `hadm_id`             | INT PK      | Hospital admission ID                                   |
| `admittime`           | TIMESTAMP   | Admission datetime                                      |
| `dischtime`           | TIMESTAMP   | Discharge datetime                                      |
| `deathtime`           | TIMESTAMP   | In-hospital death time (NULL if survived)               |
| `admission_type`      | VARCHAR(40) | e.g. ELECTIVE, URGENT, EW EMER., DIRECT EMER.           |
| `admission_location`  | VARCHAR(60) | Where patient came from (e.g. EMERGENCY ROOM)           |
| `discharge_location`  | VARCHAR(60) | Where patient went (e.g. HOME, DIED, SNF)               |
| `insurance`           | VARCHAR     | Insurance type                                          |
| `language`            | VARCHAR(10) | Preferred language                                      |
| `marital_status`      | VARCHAR(30) | Marital status                                          |
| `race`                | VARCHAR(80) | Race/ethnicity                                          |
| `edregtime`           | TIMESTAMP   | ED registration time (NULL if not via ED)               |
| `edouttime`           | TIMESTAMP   | ED discharge time                                       |
| `hospital_expire_flag`| SMALLINT    | 1 = died in hospital, 0 = survived                      |

### transfers
> All unit transfers including ICU, ward, ED. Source of `icustays`.

| Column           | Type       | Description                              |
|------------------|------------|------------------------------------------|
| `subject_id`     | INT        |                                          |
| `hadm_id`        | INT        |                                          |
| `transfer_id`    | INT PK     |                                          |
| `eventtype`      | VARCHAR    | `admit`, `transfer`, `discharge`         |
| `careunit`       | VARCHAR    | Unit name (e.g. `Medical Intensive Care Unit`)|
| `intime`         | TIMESTAMP  |                                          |
| `outtime`        | TIMESTAMP  |                                          |

### labevents  ⚠️ LARGE (~118M rows)
> All lab results (blood, urine, etc.). Cross-reference with `d_labitems`.

| Column             | Type        | Description                                        |
|--------------------|-------------|----------------------------------------------------|
| `labevent_id`      | INT PK      |                                                    |
| `subject_id`       | INT         |                                                    |
| `hadm_id`          | INT         | May be NULL for outpatient labs                    |
| `specimen_id`      | INT         | Groups measurements from the same physical sample  |
| `itemid`           | INT         | Lab test type → join `d_labitems`                  |
| `charttime`        | TIMESTAMP   | When specimen was taken                            |
| `storetime`        | TIMESTAMP   | When result became available                       |
| `value`            | VARCHAR(200)| Raw result value (text)                            |
| `valuenum`         | DOUBLE      | Numeric value (NULL if non-numeric)                |
| `valueuom`         | VARCHAR(20) | Unit of measurement                                |
| `ref_range_lower`  | DOUBLE      | Normal lower bound                                 |
| `ref_range_upper`  | DOUBLE      | Normal upper bound                                 |
| `flag`             | VARCHAR(10) | `abnormal` if out of range                         |
| `priority`         | VARCHAR(7)  | `ROUTINE` or `STAT`                                |
| `comments`         | TEXT        | Free-text comments (de-identified as `___`)        |

### d_labitems
> Dictionary for `labevents.itemid`. Small table — load fully.

| Column       | Type       | Description                        |
|--------------|------------|------------------------------------|
| `itemid`     | INT PK     |                                    |
| `label`      | VARCHAR    | Test name (e.g. "Glucose")         |
| `fluid`      | VARCHAR    | Specimen type (e.g. "Blood")       |
| `category`   | VARCHAR    | Category (e.g. "Chemistry")        |

### diagnoses_icd
> Billed ICD-9/ICD-10 diagnoses per admission.

| Column       | Type       | Description                              |
|--------------|------------|------------------------------------------|
| `subject_id` | INT        |                                          |
| `hadm_id`    | INT        |                                          |
| `seq_num`    | INT        | Priority of diagnosis (1 = primary)      |
| `icd_code`   | VARCHAR    | ICD code                                 |
| `icd_version`| SMALLINT   | 9 or 10                                  |

### procedures_icd
> Billed ICD-9/ICD-10 procedures per admission.

| Column       | Type       | Description                              |
|--------------|------------|------------------------------------------|
| `subject_id` | INT        |                                          |
| `hadm_id`    | INT        |                                          |
| `seq_num`    | INT        | Priority of procedure                    |
| `chartdate`  | DATE       | Date procedure was performed             |
| `icd_code`   | VARCHAR    | ICD code                                 |
| `icd_version`| SMALLINT   | 9 or 10                                  |

### d_icd_diagnoses / d_icd_procedures
> Dictionary for ICD codes → human-readable labels.

| Column       | Description                             |
|--------------|-----------------------------------------|
| `icd_code`   | ICD code                                |
| `icd_version`| 9 or 10                                 |
| `long_title` | Full text description                   |

### emar  (Electronic Medication Administration Record)
> Scanned at point of care. Very detailed medication administration log.

| Column              | Type        | Description                                   |
|---------------------|-------------|-----------------------------------------------|
| `subject_id`        | INT         |                                               |
| `hadm_id`           | INT         |                                               |
| `emar_id`           | VARCHAR PK  |                                               |
| `emar_seq`          | INT         | Order of administration                       |
| `poe_id`            | VARCHAR     | Links to `poe` (provider order)               |
| `pharmacy_id`       | INT         | Links to `pharmacy`                           |
| `charttime`         | TIMESTAMP   | Time drug was administered                    |
| `medication`        | TEXT        | Drug name                                     |
| `event_txt`         | VARCHAR     | e.g. "Administered", "Not Given", "Flushed"   |
| `scheduletime`      | TIMESTAMP   | Scheduled administration time                 |
| `storetime`         | TIMESTAMP   |                                               |

### emar_detail
> Line-level detail for each emar event (dose amounts, infusion rates, barcode info).

| Column                            | Description                                     |
|-----------------------------------|-------------------------------------------------|
| `subject_id`                      |                                                 |
| `emar_id`                         | Links to `emar`                                 |
| `emar_seq`                        |                                                 |
| `parent_field_ordinal`            |                                                 |
| `administration_type`             |                                                 |
| `pharmacy_id`                     |                                                 |
| `barcode_type`                    |                                                 |
| `reason_for_no_barcode`           |                                                 |
| `complete_dose_not_given`         |                                                 |
| `dose_due` / `dose_due_unit`      | Expected dose                                   |
| `dose_given` / `dose_given_unit`  | Actual dose administered                        |
| `will_remainder_of_dose_be_given` |                                                 |
| `product_amount_given`            |                                                 |
| `product_unit`                    |                                                 |
| `product_code`                    |                                                 |
| `product_description`             |                                                 |
| `product_description_other`       |                                                 |
| `prior_infusion_rate`             |                                                 |
| `infusion_rate` / `infusion_rate_unit` |                                            |
| `infusion_rate_adjustment`        |                                                 |
| `infusion_rate_adjustment_amount` |                                                 |
| `route`                           | Administration route                            |
| `infusion_complete`               |                                                 |
| `completion_interval`             |                                                 |
| `new_iv_bag_hung`                 |                                                 |
| `continued_infusion_in_other_location` |                                            |
| `restart_interval`                |                                                 |
| `side` / `site`                   | Anatomical side/site                            |
| `non_formulary_visual_verification` |                                              |

### prescriptions
> Prescribed medications (order-level, less granular than emar).

| Column              | Description                                    |
|---------------------|------------------------------------------------|
| `subject_id`        |                                                |
| `hadm_id`           |                                                |
| `pharmacy_id`       | Links to `pharmacy` and `emar`                 |
| `poe_id`            | Links to provider order in `poe`               |
| `poe_seq`           | Sequence number within the order               |
| `starttime`         | Prescribed start time                          |
| `stoptime`          | Prescribed stop time                           |
| `drug_type`         | MAIN, BASE, or ADDITIVE                        |
| `drug`              | Drug name (free text)                          |
| `formulary_drug_cd` | Hospital formulary code                        |
| `gsn`               | Generic Sequence Number                        |
| `ndc`               | National Drug Code                             |
| `prod_strength`     | e.g. "12.5mg Tablet"                           |
| `form_rx`           | Container type (e.g. TABLET, VIAL)             |
| `dose_val_rx`       | Prescribed dose value                          |
| `dose_unit_rx`      | Dose unit                                      |
| `form_val_disp`     | Amount of drug in a single formulary dose      |
| `form_unit_disp`    | Unit for formulary dose amount                 |
| `doses_per_24_hrs`  | Frequency (1=daily, 2=BID, etc.)               |
| `route`             | Administration route (IV, PO, etc.)            |

### pharmacy
> Pharmacy-level medication orders. More detail than `prescriptions`, links to `emar`.

| Column              | Description                                      |
|---------------------|--------------------------------------------------|
| `subject_id`        |                                                  |
| `hadm_id`           |                                                  |
| `pharmacy_id`       | PK — links to `prescriptions` and `emar`         |
| `poe_id`            | Links to `poe`                                   |
| `starttime`         |                                                  |
| `stoptime`          |                                                  |
| `medication`        | Drug name                                        |
| `proc_type`         | e.g. "Unit Dose"                                 |
| `status`            | e.g. "Discontinued via patient discharge"        |
| `entertime`         | When order was entered                           |
| `verifiedtime`      | When order was verified by pharmacist            |
| `route`             |                                                  |
| `frequency`         | e.g. "Q6H:PRN", "Q8H"                           |
| `disp_sched`        | Dispensing schedule (hours)                      |
| `infusion_type`     |                                                  |
| `sliding_scale`     |                                                  |
| `lockout_interval`  |                                                  |
| `basal_rate`        |                                                  |
| `one_hr_max`        |                                                  |
| `doses_per_24_hrs`  |                                                  |
| `duration`          | Numeric duration                                 |
| `duration_interval` | Unit (e.g. "Hours")                              |
| `expiration_value`  |                                                  |
| `expiration_unit`   |                                                  |
| `expirationdate`    |                                                  |
| `dispensation`      | e.g. "Omnicell", "Floor Stock Item"              |
| `fill_quantity`     |                                                  |

### poe  (Provider Order Entry)
> All physician orders (~39M rows). Links prescriptions/procedures back to the original order.

| Column                  | Description                                  |
|-------------------------|----------------------------------------------|
| `poe_id`                | PK                                           |
| `poe_seq`               | Sequence number within patient               |
| `subject_id`            |                                              |
| `hadm_id`               |                                              |
| `ordertime`             | When order was placed                        |
| `order_type`            | e.g. "Lab", "Respiratory", "Medications"     |
| `order_subtype`         | e.g. "Oxygen Therapy"                        |
| `transaction_type`      | e.g. "New"                                   |
| `discontinue_of_poe_id` | Which order this discontinues                |
| `discontinued_by_poe_id`| Which later order discontinued this          |
| `order_status`          | e.g. "Inactive", "Active"                    |

### poe_detail
> Free-text field/value pairs for each order in `poe`.

| Column       | Description                         |
|--------------|-------------------------------------|
| `poe_id`     | Links to `poe`                      |
| `poe_seq`    |                                     |
| `subject_id` |                                     |
| `field_name` | Parameter name                      |
| `field_value`| Parameter value (free text)         |

### microbiologyevents  (~3.2M rows)
> Microbiology cultures (blood, urine, CSF…) and sensitivities.

| Column                  | Description                                                  |
|-------------------------|--------------------------------------------------------------|
| `microevent_id`         | PK                                                           |
| `subject_id`            |                                                              |
| `hadm_id`               | May be NULL (assigned via transfers table)                   |
| `micro_specimen_id`     | Groups measurements from the same physical specimen          |
| `chartdate`             | Date of specimen collection (always present)                 |
| `charttime`             | Datetime of specimen collection (NULL when time unknown)     |
| `spec_itemid`           | Specimen type code (internal)                                |
| `spec_type_desc`        | e.g. "BLOOD CULTURE"                                         |
| `test_seq`              | Delineates multiple samples (e.g. aerobic vs anaerobic)      |
| `storedate`             | Date result became available                                 |
| `storetime`             | Datetime result became available                             |
| `test_itemid`           | Test performed                                               |
| `test_name`             | e.g. "ANAEROBIC BOTTLE"                                      |
| `org_itemid`            | Organism identified (NULL = no growth)                       |
| `org_name`              | e.g. "STAPHYLOCOCCUS AUREUS" (NULL = negative culture)       |
| `isolate_num`           | Isolate number for antibiotic testing                        |
| `quantity`              | Quantity of organism (rarely populated)                      |
| `ab_itemid`             | Antibiotic tested                                            |
| `ab_name`               | e.g. "OXACILLIN"                                             |
| `dilution_text`         | Raw MIC text, e.g. "<=0.12"                                  |
| `dilution_comparison`   | Comparison operator, e.g. "<="                               |
| `dilution_value`        | Numeric MIC value                                            |
| `interpretation`        | S (Sensitive), R (Resistant), I (Intermediate), P (Pending)  |
| `comments`              | Free-text comments (de-identified as `___`)                  |

### services
> Hospital service(s) caring for the patient (e.g. Medicine, Surgery).

| Column        | Description                         |
|---------------|-------------------------------------|
| `subject_id`  |                                     |
| `hadm_id`     |                                     |
| `transfertime`| When patient transferred to service |
| `prev_service`| Previous service                    |
| `curr_service`| Current service                     |

### drgcodes
> Diagnosis-Related Group (DRG) codes billed per admission.

| Column         | Description                                    |
|----------------|------------------------------------------------|
| `subject_id`   |                                                |
| `hadm_id`      |                                                |
| `drg_type`     | `HCFA` or `APR`                                |
| `drg_code`     | DRG code                                       |
| `description`  | Human-readable description                     |
| `drg_severity` | APR severity of illness (1–4, NULL for HCFA)   |
| `drg_mortality`| APR mortality risk (1–4, NULL for HCFA)        |

### hcpcsevents
> HCPCS (Healthcare Common Procedure Coding System) events billed per admission.

| Column            | Description                      |
|-------------------|----------------------------------|
| `subject_id`      |                                  |
| `hadm_id`         |                                  |
| `chartdate`       |                                  |
| `hcpcs_cd`        | HCPCS code                       |
| `seq_num`         | Sequence number                  |
| `short_description`| Human-readable description      |

### d_hcpcs
> Dictionary for HCPCS codes.

| Column             | Description              |
|--------------------|--------------------------|
| `code`             | HCPCS code               |
| `category`         | Category                 |
| `long_description` | Full description         |
| `short_description`| Short description        |

### omr  (Online Medical Record)
> Miscellaneous outpatient/clinic measurements (e.g. BMI, blood pressure). (~6.4M rows)

| Column         | Description                             |
|----------------|-----------------------------------------|
| `subject_id`   |                                         |
| `chartdate`    |                                         |
| `seq_num`      | Order within date                       |
| `result_name`  | Measurement name (e.g. "BMI")           |
| `result_value` | Value as text                           |

---

## MODULE: icu

Data from MetaVision (iMDSoft) — the ICU-specific information system.  
Star schema: `icustays` + `d_items` ← linked to all `*events` tables.

### icustays  (~73k rows)
> One row per ICU stay. Derived from `transfers`.

| Column           | Type      | Description                                    |
|------------------|-----------|------------------------------------------------|
| `subject_id`     | INT       |                                                |
| `hadm_id`        | INT       |                                                |
| `stay_id`        | INT PK    | ICU stay identifier                            |
| `first_careunit` | VARCHAR   | First ICU type (e.g. `Medical Intensive Care Unit`)|
| `last_careunit`  | VARCHAR   | Last ICU type                                  |
| `intime`         | TIMESTAMP | ICU admission time                             |
| `outtime`        | TIMESTAMP | ICU discharge time                             |
| `los`            | DOUBLE    | Length of stay in **fractional days**          |

### d_items
> Dictionary of all ICU concepts (itemid). Cross-reference for all events tables.

| Column                              | Description                                    |
|-------------------------------------|------------------------------------------------|
| `itemid`                            | Unique concept identifier                      |
| `label`                             | Concept name (e.g. "Heart Rate")               |
| `abbreviation`                      | Short name                                     |
| `linksto`                           | Which table it links to (e.g. `chartevents`)   |
| `category`                          | e.g. "Routine Vital Signs", "Labs"             |
| `unitname`                          | Unit (e.g. "bpm")                              |
| `param_type`                        | Numeric, Text, Date/Time, etc.                 |
| `lownormalvalue` / `highnormalvalue`| Normal range                                   |

### chartevents  ⚠️ HUGE (~314M rows)
> The majority of all ICU data: vitals, ventilator settings, neuro assessments, labs.

| Column        | Type        | Description                                     |
|---------------|-------------|-------------------------------------------------|
| `subject_id`  | INT         |                                                 |
| `hadm_id`     | INT         |                                                 |
| `stay_id`     | INT         |                                                 |
| `charttime`   | TIMESTAMP   | Time of observation                             |
| `storetime`   | TIMESTAMP   | Time manually entered/validated                 |
| `itemid`      | INT         | → `d_items`                                     |
| `value`       | VARCHAR(200)| Text value                                      |
| `valuenum`    | DOUBLE      | Numeric value (NULL if non-numeric)             |
| `valueuom`    | VARCHAR(20) | Unit (e.g. "bpm", "mmHg")                       |
| `warning`     | SMALLINT    | 1 if care provider flagged warning              |

> ⚠️ Lab values are duplicated here from `labevents`. When there's a conflict, trust `labevents`.

### inputevents  (~9M rows)
> IV fluids and medications infused. Continuous infusions + intermittent boluses.

| Column                          | Description                                               |
|---------------------------------|-----------------------------------------------------------|
| `subject_id`                    |                                                           |
| `hadm_id`                       |                                                           |
| `stay_id`                       |                                                           |
| `starttime`                     | Infusion start                                            |
| `endtime`                       | Infusion end (bolus = starttime + 1 min)                  |
| `storetime`                     | When manually entered/validated                           |
| `itemid`                        | → `d_items`                                               |
| `amount`                        | Amount administered between start/end                     |
| `amountuom`                     | Amount unit (e.g. "mL")                                   |
| `rate`                          | Rate of infusion                                          |
| `rateuom`                       | Rate unit (e.g. "mL/hour", "mcg/kg/min")                  |
| `orderid`                       | Groups components of the same solution                    |
| `linkorderid`                   | Links same order across rate changes                      |
| `ordercategoryname`             | e.g. "Continuous Med", "Bolus Med"                        |
| `secondaryordercategoryname`    | Secondary category                                        |
| `ordercomponenttypedescription` | Role of substance (main order, additive, mixed solution)  |
| `ordercategorydescription`      | Higher-level order category                               |
| `patientweight`                 | Patient weight in kg                                      |
| `totalamount`                   | Total fluid in the bag                                    |
| `totalamountuom`                | Unit for total amount                                     |
| `isopenbag`                     | 1 = open bag (amount uncertain)                           |
| `continueinnextdept`            | 1 = order continued after transfer                        |
| `statusdescription`             | e.g. "Changed", "Stopped", "FinishedRunning", "Paused"    |
| `originalamount`                | Drug amount remaining in bag at starttime                 |
| `originalrate`                  | Originally planned rate (may differ from `rate`)          |

### outputevents  (~4.2M rows)
> Patient outputs: urine, drains, etc.

| Column        | Description                    |
|---------------|--------------------------------|
| `subject_id`  |                                |
| `hadm_id`     |                                |
| `stay_id`     |                                |
| `charttime`   | Time of output measurement     |
| `storetime`   | When manually entered/validated|
| `itemid`      | → `d_items`                    |
| `value`       | Amount output (numeric)        |
| `valueuom`    | Unit (usually "mL")            |

### procedureevents  (~696k rows)
> Documented procedures (ventilation, dialysis, etc.). Absence ≠ procedure not done.

| Column                    | Description                                          |
|---------------------------|------------------------------------------------------|
| `subject_id`              |                                                      |
| `hadm_id`                 |                                                      |
| `stay_id`                 |                                                      |
| `starttime`               |                                                      |
| `endtime`                 |                                                      |
| `storetime`               | When manually entered                                |
| `itemid`                  | → `d_items` (e.g. 225792 = Invasive Ventilation)     |
| `value`                   | Duration of procedure (e.g. 461 minutes)             |
| `valueuom`                | "min", "hour", "day", or "None" (instantaneous)      |
| `location`                | Anatomical location (e.g. "Left Upper Arm")          |
| `locationcategory`        | Location category (e.g. "Invasive Venous")           |
| `orderid`                 | Links to physician order                             |
| `linkorderid`             | Links repeated procedures under same original order  |
| `ordercategoryname`       | Type of procedure                                    |
| `ordercategorydescription`|                                                      |
| `patientweight`           | Patient weight in kg                                 |
| `isopenbag`               |                                                      |
| `continueinnextdept`      | 1 = continued after transfer                         |
| `statusdescription`       | "FinishedRunning", "Stopped", "Paused"               |
| `originalamount`          |                                                      |
| `originalrate`            |                                                      |

### datetimeevents  (~7.1M rows)
> Charted items that are a **date/time** (e.g. "Date of last dialysis").

| Column        | Description                                     |
|---------------|-------------------------------------------------|
| `subject_id`  |                                                 |
| `hadm_id`     |                                                 |
| `stay_id`     |                                                 |
| `charttime`   | Time of observation                             |
| `storetime`   |                                                 |
| `itemid`      | → `d_items`                                     |
| `value`       | Datetime string                                 |
| `valueuom`    |                                                 |
| `warning`     |                                                 |

### ingredientevents  (~11.6M rows)
> Nutritional/water content of IV infusions (links to `inputevents`).

| Column              | Description                                     |
|---------------------|-------------------------------------------------|
| `subject_id`        |                                                 |
| `hadm_id`           |                                                 |
| `stay_id`           |                                                 |
| `starttime`         |                                                 |
| `endtime`           |                                                 |
| `storetime`         |                                                 |
| `itemid`            | → `d_items` (ingredient type)                   |
| `amount`            | Amount of ingredient                            |
| `amountuom`         | Unit                                            |
| `rate`              | Infusion rate                                   |
| `rateuom`           | Rate unit                                       |
| `orderid`           | Links to `inputevents`                          |
| `linkorderid`       |                                                 |
| `statusdescription` | e.g. "FinishedRunning", "Stopped"               |
| `originalamount`    |                                                 |
| `originalrate`      |                                                 |

---

## Common join patterns

```sql
-- ICU patients with demographics
SELECT p.subject_id, p.gender, p.anchor_age, i.stay_id, i.los
FROM patients p
JOIN admissions a USING (subject_id)
JOIN icustays  i USING (hadm_id);

-- Vital signs (heart rate) for an ICU stay
SELECT ce.charttime, ce.valuenum AS heart_rate
FROM chartevents ce
JOIN d_items di USING (itemid)
WHERE ce.stay_id = 12345
  AND di.label = 'Heart Rate';

-- Lab results (glucose) for a hospital admission
SELECT le.charttime, le.valuenum AS glucose, le.valueuom
FROM labevents le
JOIN d_labitems dl USING (itemid)
WHERE le.hadm_id = 2000001
  AND dl.label = 'Glucose';

-- In-hospital mortality
SELECT a.hadm_id, a.hospital_expire_flag
FROM admissions a
WHERE a.hospital_expire_flag = 1;
```

---

## File sizes at a glance (uncompressed CSVs)

| File                         | Size   | Rows (approx) |
|------------------------------|--------|---------------|
| icu/chartevents.csv          | 27 GB  | 314 M         |
| hosp/labevents.csv           | 13 GB  | 118 M         |
| hosp/emar_detail.csv         | 5.1 GB | 54.5 M        |
| hosp/emar.csv                | 3.6 GB | 26.7 M        |
| hosp/poe.csv                 | 3.4 GB | 39.3 M        |
| hosp/pharmacy.csv            | 2.9 GB | 13.6 M        |
| hosp/prescriptions.csv       | 2.4 GB | 15.4 M        |
| icu/inputevents.csv          | 2.2 GB | 9.0 M         |
| icu/ingredientevents.csv     | 1.9 GB | 11.6 M        |
| icu/datetimeevents.csv       | 703 MB | 7.1 M         |
| hosp/microbiologyevents.csv  | 698 MB | 3.2 M         |
| icu/outputevents.csv         | 325 MB | 4.2 M         |
| hosp/omr.csv                 | 253 MB | 6.4 M         |
| hosp/poe_detail.csv          | 178 MB | 3.0 M         |
| hosp/transfers.csv           | 151 MB | 1.89 M        |
| hosp/diagnoses_icd.csv       | 129 MB | 4.75 M        |
| icu/procedureevents.csv      | 121 MB | 696 k         |
| hosp/procedures_icd.csv      | 26 MB  | 668 k         |
| hosp/drgcodes.csv            | 41 MB  | 604 k         |
| hosp/services.csv            | 20 MB  | 468 k         |
| hosp/admissions.csv          | 67 MB  | 431 k         |
| hosp/hcpcsevents.csv         | 9.3 MB | 151 k         |
| icu/icustays.csv             | 11 MB  | 73 k          |
| hosp/patients.csv            | 9.5 MB | 300 k         |
| hosp/d_icd_diagnoses.csv     | 8.5 MB | 110 k         |
| hosp/d_icd_procedures.csv    | 7.1 MB | 85 k          |
| hosp/d_hcpcs.csv             | 3.2 MB | 89 k          |
| icu/d_items.csv              | 360 KB | 4 k           |
| hosp/d_labitems.csv          | 64 KB  | 1.6 k         |

> **Start small**: `patients`, `admissions`, `icustays`, `d_items`, `d_labitems` are all tiny.  
> **Big files**: `chartevents` and `labevents` — use column selection and time filtering.

---

## Scale of the dataset (local)

- **~299,777** unique patients
- **~431,088** hospital admissions
- **~73,141** ICU stays
- **314 million** charted ICU observations
- **118 million** lab measurements
- **15.4 million** prescription orders
- **26.7 million** medication administration records (emar)

---

## Useful links

- Docs: https://mimic.mit.edu/docs/iv/
- Code & derived tables: https://github.com/MIT-LCP/mimic-code
- BigQuery sandbox: https://mimic.mit.edu/docs/gettingstarted/cloud/bigquery/
