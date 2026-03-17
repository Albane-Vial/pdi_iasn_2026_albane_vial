# MIMIC-IV v2.1 — Complete Reference

Source: Beth Israel Deaconess Medical Center (BIDMC), Boston, MA  
Covers: 2008–2019 (~300k patients, ~500k hospital admissions)  
Official docs: https://mimic.mit.edu/docs/iv/

---

## Key identifiers (shared across all tables)

| Identifier   | Scope           | Description                                      |
|--------------|-----------------|--------------------------------------------------|
| `subject_id` | Patient          | Unique per patient, consistent across all tables |
| `hadm_id`    | Hospital stay    | Unique per hospital admission (2000000–2999999)  |
| `stay_id`    | ICU stay         | Unique per ICU stay (derived from transfers)     |
| `stay_id`    | ED stay          | Same column name, separate namespace in ED module|

> **Time shift**: All dates are shifted into the future for de-identification.  
> Use `anchor_year` + `anchor_year_group` from `patients` to estimate the real year.

---

## Module overview

```
MIMIC-IV v2.1
├── hosp/        Hospital-wide EHR (22 tables)
├── icu/         ICU clinical info system — MetaVision (9 tables)
└── ed/          Emergency Department (6 tables)

Separate datasets (same subject_id):
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
| `admission_type`      | VARCHAR(40) | 9 types: ELECTIVE, URGENT, EW EMER., DIRECT EMER., etc. |
| `admit_provider_id`   | VARCHAR(10) | Anonymised admitting provider                           |
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
| `order_provider_id`| VARCHAR(10) |                                                    |
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
> Dictionary for `labevents.itemid`. Small table — download fully.

| Column       | Type       | Description                        |
|--------------|------------|------------------------------------|
| `itemid`     | INT PK     |                                    |
| `label`      | VARCHAR    | Test name (e.g. "Glucose")         |
| `fluid`      | VARCHAR    | Specimen type (e.g. "Blood")       |
| `category`   | VARCHAR    | Category (e.g. "Chemistry")        |
| `loinc_code` | VARCHAR    | LOINC code                         |

### diagnoses_icd
> Billed ICD-9/ICD-10 diagnoses per admission.

| Column       | Type       | Description                              |
|--------------|------------|------------------------------------------|
| `subject_id` | INT        |                                          |
| `hadm_id`    | INT        |                                          |
| `seq_num`    | INT        | Priority of diagnosis (1 = primary)      |
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
| `enter_provider_id` | VARCHAR     |                                               |
| `charttime`         | TIMESTAMP   | Time drug was administered                    |
| `medication`        | TEXT        | Drug name                                     |
| `event_txt`         | VARCHAR     | e.g. "Administered", "Not Given"              |
| `scheduletime`      | TIMESTAMP   | Scheduled administration time                 |
| `storetime`         | TIMESTAMP   |                                               |

### prescriptions
> Prescribed medications (order-level, less granular than emar).

| Column              | Description                                    |
|---------------------|------------------------------------------------|
| `subject_id`        |                                                |
| `hadm_id`           |                                                |
| `pharmacy_id`       | Links to `pharmacy` and `emar`                 |
| `poe_id`            | Links to provider order in `poe`               |
| `poe_seq`           | Sequence number within the order               |
| `order_provider_id` | Anonymised provider who initiated the order    |
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

### microbiologyevents
> Microbiology cultures (blood, urine, CSF…) and sensitivities.

| Column                  | Description                                                  |
|-------------------------|--------------------------------------------------------------|
| `microevent_id`         | PK                                                           |
| `subject_id`            |                                                              |
| `hadm_id`               | May be NULL (assigned via transfers table)                   |
| `micro_specimen_id`     | Groups measurements from the same physical specimen          |
| `order_provider_id`     | Anonymised provider who ordered the test                     |
| `chartdate`             | Date of specimen collection (always present)                 |
| `charttime`             | Datetime of specimen collection (NULL when time unknown)     |
| `spec_itemid`           | Specimen type code (internal, not `d_labitems`)              |
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
| `interpretation`        | S (Sensitive), R (Resistant), I (Intermediate), P (Pending) |
| `comments`              | Free-text comments (de-identified as `___`)                  |

### services
> Hospital service(s) caring for the patient (e.g. Medicine, Surgery).

| Column        | Description                    |
|---------------|--------------------------------|
| `subject_id`  |                                |
| `hadm_id`     |                                |
| `transfertime`| When patient transferred to service |
| `prev_service`| Previous service               |
| `curr_service`| Current service                |

### omr  (Online Medical Record)
> Miscellaneous outpatient/clinic measurements (e.g. BMI, blood pressure).

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

| Column           | Type    | Description                                    |
|------------------|---------|------------------------------------------------|
| `subject_id`     | INT     |                                                |
| `hadm_id`        | INT     |                                                |
| `stay_id`        | INT PK  | ICU stay identifier                            |
| `first_careunit` | VARCHAR | First ICU type (e.g. `Medical Intensive Care Unit`)|
| `last_careunit`  | VARCHAR | Last ICU type                                  |
| `intime`         | TIMESTAMP | ICU admission time                           |
| `outtime`        | TIMESTAMP | ICU discharge time                           |
| `los`            | DOUBLE  | Length of stay in **fractional days**          |

### d_items
> Dictionary of all ICU concepts (itemid). Cross-reference for all events tables.

| Column       | Description                                    |
|--------------|------------------------------------------------|
| `itemid`     | Unique concept identifier                      |
| `label`      | Concept name (e.g. "Heart Rate")               |
| `abbreviation` | Short name                                   |
| `linksto`    | Which table it links to (e.g. `chartevents`)   |
| `category`   | e.g. "Routine Vital Signs", "Labs"             |
| `unitname`   | Unit (e.g. "bpm")                              |
| `param_type` | Numeric, Text, Date/Time, etc.                 |
| `lownormalvalue` / `highnormalvalue` | Normal range                |

### chartevents  ⚠️ HUGE (~313M rows)
> The majority of all ICU data: vitals, ventilator settings, neuro assessments, labs.

| Column        | Type        | Description                                     |
|---------------|-------------|-------------------------------------------------|
| `subject_id`  | INT         |                                                 |
| `hadm_id`     | INT         |                                                 |
| `stay_id`     | INT         |                                                 |
| `caregiver_id`| INT         | Who documented the observation                  |
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
| `caregiver_id`                  | Who documented the event                                  |
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
| `caregiver_id`| Who documented the event       |
| `charttime`   | Time of output measurement     |
| `storetime`   | When manually entered/validated|
| `itemid`      | → `d_items`                    |
| `value`       | Amount output (numeric)        |
| `valueuom`    | Unit (usually "mL")            |

### procedureevents  (~696k rows)
> Documented procedures (ventilation, dialysis, etc.). Absence ≠ procedure not done.

| Column                   | Description                                          |
|--------------------------|------------------------------------------------------|
| `subject_id`             |                                                      |
| `hadm_id`                |                                                      |
| `stay_id`                |                                                      |
| `caregiver_id`           | Who documented the event                             |
| `starttime`              |                                                      |
| `endtime`                |                                                      |
| `storetime`              | When manually entered                                |
| `itemid`                 | → `d_items` (e.g. 225792 = Invasive Ventilation)     |
| `value`                  | Duration of procedure (e.g. 461 minutes)             |
| `valueuom`               | "min", "hour", "day", or "None" (instantaneous)      |
| `location`               | Anatomical location (e.g. "Left Upper Arm")          |
| `locationcategory`       | Location category (e.g. "Invasive Venous")           |
| `orderid`                | Links to physician order                             |
| `linkorderid`            | Links repeated procedures under same original order  |
| `ordercategoryname`      | Type of procedure                                    |
| `ordercategorydescription`|                                                     |
| `patientweight`          | Patient weight in kg                                 |
| `isopenbag`              |                                                      |
| `continueinnextdept`     | 1 = continued after transfer                         |
| `statusdescription`      | "FinishedRunning", "Stopped", "Paused"               |
| `originalamount`         | Present but no clear meaning                         |
| `originalrate`           | Present but no clear meaning (always 0 or 1)         |

### datetimeevents
> Charted items that are a **date/time** (e.g. "Date of last dialysis").

Same structure as `chartevents` but `value` holds a datetime string.

### ingredientevents
> Nutritional/water content of IV infusions (links to `inputevents`).

| Column      | Description                     |
|-------------|---------------------------------|
| `stay_id`   |                                 |
| `itemid`    | → `d_items` (ingredient type)   |
| `starttime` |                                 |
| `endtime`   |                                 |
| `amount`    | Amount of ingredient            |
| `amountuom` | Unit                            |
| `patientweight` |                             |

---

## MODULE: ed  (Emergency Department)

Separate from hosp/icu. Linked via `subject_id` and `hadm_id`.

### edstays
> One row per ED visit. The ED equivalent of `icustays`.

| Column          | Description                                    |
|-----------------|------------------------------------------------|
| `subject_id`    |                                                |
| `hadm_id`       | NULL if patient discharged without admission   |
| `stay_id`       | ED stay identifier (different namespace from ICU)|
| `intime`        | ED arrival                                     |
| `outtime`       | ED departure                                   |
| `gender`        |                                                |
| `race`          |                                                |
| `arrival_transport` | e.g. "WALK IN", "AMBULANCE"               |
| `disposition`   | e.g. "ADMITTED", "HOME", "TRANSFER"            |

### triage
> Triage assessment at ED arrival (one row per ED stay).

| Column           | Description                          |
|------------------|--------------------------------------|
| `subject_id`     |                                      |
| `stay_id`        |                                      |
| `temperature`    | °F                                   |
| `heartrate`      | bpm                                  |
| `resprate`       | breaths/min                          |
| `o2sat`          | SpO₂ %                               |
| `sbp` / `dbp`    | Systolic / diastolic BP mmHg         |
| `pain`           | Pain score 0–10                      |
| `acuity`         | ESI triage level 1–5 (1=most urgent) |
| `chiefcomplaint` | Free-text chief complaint            |

### vitalsign
> Vital signs throughout the ED stay (multiple rows per stay).

| Column       | Description                   |
|--------------|-------------------------------|
| `subject_id` |                               |
| `stay_id`    |                               |
| `charttime`  |                               |
| `temperature`|                               |
| `heartrate`  |                               |
| `resprate`   |                               |
| `o2sat`      |                               |
| `sbp` / `dbp`|                               |
| `rhythm`     | Cardiac rhythm                |
| `pain`       | Pain score                    |

### diagnosis
> ICD diagnoses assigned in the ED.

| Column       | Description                         |
|--------------|-------------------------------------|
| `subject_id` |                                     |
| `stay_id`    |                                     |
| `seq_num`    | Diagnosis priority                  |
| `icd_code`   |                                     |
| `icd_version`| 9 or 10                             |
| `icd_title`  | Description (included inline here)  |

### medrecon
> Medication reconciliation at ED admission.

| Column         | Description                       |
|----------------|-----------------------------------|
| `subject_id`   |                                   |
| `stay_id`      |                                   |
| `charttime`    |                                   |
| `name`         | Medication name                   |
| `gsn`          | Generic Sequence Number           |
| `ndc`          | National Drug Code                |
| `etc_rn`       | ETC drug category                 |
| `etccode`      | ETC code                          |
| `etcdescription` | ETC category description        |

### pyxis
> Medications dispensed from automated dispensing cabinets in the ED.

| Column       | Description                         |
|--------------|-------------------------------------|
| `subject_id` |                                     |
| `stay_id`    |                                     |
| `charttime`  |                                     |
| `med_rn`     | Row number within stay              |
| `name`       | Medication name                     |
| `gsn_rn`     | GSN row number                      |
| `gsn`        | Generic Sequence Number             |

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

## File sizes at a glance (approximate)

| File                       | Size   | Rows          |
|----------------------------|--------|---------------|
| icu/chartevents.csv.gz     | ~2 GB  | 313 M         |
| hosp/labevents.csv.gz      | ~1.5 GB| 118 M         |
| hosp/emar.csv.gz           | ~300 MB| —             |
| hosp/emar_detail.csv.gz    | ~600 MB| —             |
| hosp/prescriptions.csv.gz  | ~200 MB| —             |
| icu/inputevents.csv.gz     | ~200 MB| —             |
| hosp/microbiologyevents.csv| ~50 MB | —             |
| icu/icustays.csv           | ~4 MB  | 73k           |
| hosp/admissions.csv        | ~15 MB | ~500k         |
| hosp/patients.csv          | ~5 MB  | ~300k         |
| *d_* tables                | <5 MB  | thousands     |

> **Start small**: `patients`, `admissions`, `icustays`, `d_items`, `d_labitems` are all tiny.  
> **Big files**: `chartevents` and `labevents` — use column selection and time filtering.

---

## Scale of the dataset

- **~300,000** unique patients  
- **~500,000** hospital admissions  
- **~73,000** ICU stays  
- **~425,000** ED stays  
- **313 million** charted ICU observations  
- **118 million** lab measurements  

---

## Useful links

- Docs: https://mimic.mit.edu/docs/iv/
- Code & derived tables: https://github.com/MIT-LCP/mimic-code
- BigQuery sandbox: https://mimic.mit.edu/docs/gettingstarted/cloud/bigquery/
- Kaggle dataset: https://www.kaggle.com/datasets/mangeshwagle/mimic-iv-2-1
