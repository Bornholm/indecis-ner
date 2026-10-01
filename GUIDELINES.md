# Annotation guidelines: personal data in workplace documents

These rules label the synthetic documents of `synth/` and, later, the documents written by language models. They extend the guidelines of the [Text Anonymization Benchmark](https://github.com/NorskRegnesentral/text-anonymization-benchmark/blob/master/guidelines.md) (TAB) to French workplace documents: emails, payslips, employment contracts, prescriptions, sick notes, appraisal reports.

A passage gets up to three labels, one per question of the model.

## 1. Type (`entities`)

The eight types of TAB, so that TAB and these documents train the same heads.

| Type | What | Examples |
| --- | --- | --- |
| `PERSON` | names of people, with civility and title; initials, usernames | `Mme Carmen Roussel`, `Dr Weiss`, `J.-P. M.` |
| `CODE` | numbers and codes that identify something | social security number (NIR), employee number, IBAN, phone, email, case number |
| `LOC` | places: addresses, towns, sites, buildings | `12 rue des Lilas 21000 Dijon`, `site de Chenôve` |
| `ORG` | organisations: employers, schools, hospitals, unions, departments with a name | `Clinique du Parc`, `CFDT`, `service de pneumologie` |
| `DEM` | attributes of a person: job, rank, education, age, nationality, family situation, physical description, diagnosis | `infirmière de nuit`, `42 ans`, `marié, deux enfants`, `diabète de type 2` |
| `DATETIME` | dates, times and durations | `12/03/1981`, `depuis le 4 janvier 2021`, `trois semaines` |
| `QUANTITY` | meaningful quantities | salary, `taux d'incapacité de 20 %`, `1 750,00 €` |
| `MISC` | any other information that describes a person | a vehicle, a hobby, an event they took part in |

Spans are minimal: the name without the article, the whole address in one span when it is written in one line, the street and the town in two spans when they are on two lines.

## 2. Masking (`mask`)

Who is protected: **every natural person** the document mentions, as the GDPR protects all of them (TAB protected one applicant per ruling). Organisations are not protected for themselves; they are masked when they point to a person.

| Decision | Rule | Examples |
| --- | --- | --- |
| `DIRECT` | identifies a person on its own | full name, NIR, employee number, personal email or phone, IBAN, full home address |
| `QUASI` | identifies a person combined with other information | first or last name alone (a form's separate Nom and Prénom fields are both `DIRECT`), birth date, job, department, site, salary, hiring date, town of residence, employer, diagnosis, age |
| none | says nothing about a person | legal amounts (SMIC, social security ceiling), public bodies acting as such (`URSSAF`, `CPAM`), contract templates, dates of the document's issue when they concern nobody |

When in doubt between `QUASI` and none, choose `QUASI`: a missed quasi-identifier costs more than an extra mask.

## 3. Special categories (`sensitive`)

Data of article 9 GDPR, and article 10 for offences, labeled whatever their masking decision, because they call for a stronger treatment.

| Label | What | Examples |
| --- | --- | --- |
| `HEALTH` | physical or mental health, disability, care, medication, sick leave reasons | `lombalgie chronique`, `Doliprane 1000 mg`, `RQTH`, `arrêt pour burn-out` |
| `UNION` | trade union membership or mandate | `délégué syndical CGT`, `élue CSE` |
| `BELIEF` | religious or philosophical beliefs | `pratiquant`, `congé pour l'Aïd` |
| `POLITICS` | political opinions or membership | `adhérent d'un parti`, `conseiller municipal` |
| `ORIGIN` | racial or ethnic origin | `d'origine marocaine` (a nationality alone is `DEM` only) |
| `SEXUALITY` | sex life or sexual orientation | `son compagnon`, when it reveals orientation |
| `OFFENCE` | criminal convictions and offences | `condamné pour vol`, `casier judiciaire` |

## 4. What is not labeled

Headings, legal boilerplate, generic job descriptions that describe no one (`le poste comprend l'accueil des patients`), amounts of the employer's charges that do not depend on the employee, and placeholder values of forms (`[NOM]`).
