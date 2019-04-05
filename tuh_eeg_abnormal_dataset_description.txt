File: _AAREADME.txt
Database: TUH Abnormal EEG Corpus
Version: v2.0.0

----
Change Log:

(20180411) updated the documentation and prepared for release

(20180118) re-mapped files to be consistent with TUH EEG v1.1.0. removed
	   22 files that were found to be duplicates.

(20170912) bug fix release. 18 edf files containing negative values for age 
	   have been updated. 20 edf files containing ages that do not match 
	   their report have been updated. see the list at the end of 
           this file for the files that were changed from v1.1.1.

(20170816) bug fix release. The corpus now contains only averaged reference
	   EEG recordings.

(20170815) added documentation about the electrode configuration.

(20170708) includes the EDF files. corrupted headers were fixed.
	   overlap between the evaluation and training partitions
	   was eliminated. more statistics about the data is provided.

(20170314) a bug fix release. the corpus now contains only one file 
	   per session. also, we provide a suggested partitioning 
	   of the data into evaluation and training data.

(20170314) the initial release.
----

When you use this specific corpus in your research or technology
development, we ask that you reference the corpus using this
publication:

 Lopez, S. (2017). Automated Identification of Abnormal EEGs. 
 Temple University.

This publication can be retrieved from:

 https://www.isip.piconepress.com/publications/ms_theses/2017/abnormal/thesis/

Our preferred reference for the TUH EEG Corpus, from which this
seizure corpus was derived, is:

 Obeid, I., & Picone, J. (2016). The Temple University Hospital EEG
 Data Corpus. Frontiers in Neuroscience, Section Neural Technology,
 10, 196. http://dx.doi.org/10.3389/fnins.2016.00196.

This file contains information about the demographics and relevant
statistics for the TUH EEG Abnormal Corpus, which contains EEG records
that are classified as clinically normal or abnormal.

FILENAME STRUCTURE:

 A typical filename in this corpus is:

  edf/train/normal/01_tcp_ar/101/00010194/s001_2013_01_09/00010194_s001_t001.edf

 The first segment, "edf/", is a directory name for the directory containing
 the data, which consists of edf files (*.edf) and EEG reports (*.txt).

 The second segment denotes either the evaluation data ("/eval") or
 the training data ("/train").

 The third segment ("normal") denotes whether the EEG is "normal" or
 "abnormal".

 The fourth segment ("/01_tcp_ar") denotes the type of channel configuration
 for the EEG. "/01_tcp_ar" refers to an AR reference configuration.
 In this corpus there is only one type of configuration used.

 The fifth segment ("101") is a three-digit identifier meant to keep
 the number of subdirectories in a directory manageable. This follows
 the TUH EEG v1.1.0 convention.

 The sixth segment ("/00010194") denotes an anonymized patient ID. The
 IDs are consistent across all of our databases involving Temple
 Hospital EEG data.

 The seventh segment ("/s001_2013_01_09") denotes the session number
 ("s001"), and the date the EEG was archived at the hospital
 ("01/09/2013"). The archive date is not necessarily the date the EEG
 was recorded (which is available in the EDF header), but close to
 it. EEGs are usually archived within a few days of being recorded.

 The eighth, or last, segment is the filename
 ("00010194_s001_t001.edf"). This includes the patient number, the
 session number and a token number ("t001").  EEGs are split into a
 series of files starting with *t000.edf, *t001.edf, ...  These
 represent pruned EEGs, so the original EEG is split into these
 segments, and uninteresting parts of the original recording were
 deleted (common in clinical practice).

 There are two types of files in this release: *.edf represents the signal
 data, and *.txt represents the EEG report.

PATIENT, SESSION AND FILE STATISTICS:

 The patient statistics are summarized in the table below:

 Patients:

  |----------------------------------------------|
  | Description |  Normal  | Abnormal |  Total   |
  |-------------+----------+----------+----------|
  | Evaluation  |      148 |      105 |      253 |
  |-------------+----------+----------+----------|
  | Train       |    1,237 |      893 |    2,130 |
  |-------------+----------+----------+----------|
  | Total       |    1,385 |      998 |    2,383 |
  |----------------------------------------------|
 
 It is important to note that (1) there is no overlap between patients
 in the evaluation and training sets, (2) patients only appear once in
 the evaluation set as either normal or abnormal (but not both), and
 (3) some patients appear more than once in the training set.

 Therefore, there are 253 unique patients in the evaluation set, but
 only 2,076 unique patients in the training set. Hence, there are 54
 patients that appear as both normal and abnormal in the training
 set. This was a conscious design decision as we wanted some examples
 of patients who demonstrated both morphologies.
 
 Patients can have multiple sessions. Below is a table describing the
 distribution of sessions:

 Sessions:

  |----------------------------------------------|
  | Description |  Normal  | Abnormal |  Total   |
  |-------------+----------+----------+----------|
  | Evaluation  |      150 |      126 |      276 |
  |-------------+----------+----------+----------|
  | Train       |    1,371 |    1,346 |    2,717 |
  |-------------+----------+----------+----------|
  | Total       |    1,521 |    1,472 |    2,993 |
  |----------------------------------------------|

 More than one session from a patient appears in this database. We
 selected files/sessions based on their relevance to the
 normal/abnormal detection problem - whether they display some
 challenging or interesting behavior. However, unlike v1.0.0, the
 evaluation set and training set are 100% disjoint - no patient
 appears in both partitions.

 Most of the patients in the evaluation set appear once (average
 number of sessions per patient is 1.09), while patients in the
 training set have an average of 1.28 sessions.

 Some basic statistics on the number of files and the number of hours
 of data are given below:

 Size (No. of Files / Hours of Data):

  |----------------------------------------------------------------------|
  | Description |      Normal      |     Abnormal     |      Total       |
  |-------------+------------------+------------------+------------------|
  | Evaluation  |   150 (   55.46) |   126 (   47.48) |   276 (  102.94) |
  |-------------+------------------+------------------+------------------|
  | Train       | 1,371 (  512.01) | 1,346 (  526.05) | 2,717 (1,038.06) |
  |-------------+------------------+------------------+------------------|
  | Total       | 1,521 (  567.47) | 1,472 (  573.53) | 2,993 (1,142.00) |
  |----------------------------------------------------------------------|

 Only one file from each session was included in this corpus. It is
 important to point out that each EEG session is comprised of several
 EDF files (the records are pruned before they are stored in the
 database).  A single file was selected from a session - typically the
 longest file in the session. We did not include multiple files from
 the same session. So the number of files and number of sessions are
 identical.

 Each file selected from a session was chosen by considering the
 length of the file (all the files in this corpus are longer than 15
 minutes) and/or the presence of relevant activity.

INTER-RATER AGREEMENT:

 A summary of the distribution of normal/abnormal EEGs is shown below:

 Evaluation:

  |-----------------------------------------------------------|
  | Description |    Files     |   Sessions   |    Patients   | 
  |-------------+--------------+--------------+---------------|
  | Abnormal    |   126 ( 46%) |   126 ( 46%) |    105 ( 42%) |
  |-------------+--------------+--------------+---------------|
  | Normal      |   150 ( 54%) |   150 ( 54%) |    148 ( 58%) |
  |-------------+--------------+--------------+---------------|
  | Total       |   276 (100%) |   276 (100%) |    253 (100%) |
  |-----------------------------------------------------------|

 Train:

  |-----------------------------------------------------------|
  | Description |    Files     |   Sessions   |    Patients   | 
  |-------------+--------------+--------------+---------------|
  | Abnormal    | 1,346 ( 50%) | 1,346 ( 50%) |    893 ( 42%) |
  |-------------+--------------+--------------+---------------|
  | Normal      | 1,371 ( 50%) | 1,371 ( 50%) |  1,237 ( 58%) |
  |-------------+--------------+--------------+---------------|
  | Total       | 2,717 (100%) | 2,717 (100%) |  2,130 (100%) |
  |-----------------------------------------------------------|

 In our v1.1.1 release, we manually reviewed the data to determine the
 extent to which our assessments were in agreement with the associated
 EEG reports. The outcome of this analysis was as follows:

 Evaluation:

  |---------------------------------------------------|
  | Description         |    Files     |   Patients   |
  |---------------------+--------------+--------------|
  | Positive Agreement* |   276 (100%) |   254 (100%) |
  |---------------------+--------------+--------------|
  | Negative Agreement* |     0 (  0%) |     0 (  0%) |
  |---------------------------------------------------|

 Train:

  |---------------------------------------------------|
  | Description         |    Files     |   Patients   |
  |---------------------+--------------+--------------|
  | Positive Agreement* | 2,700 ( 99%) | 2,110 ( 97%) |
  |---------------------+--------------+--------------|
  | Negative Agreement* |    27 (  1%) |    21 (  1%) |
  |---------------------------------------------------|

  Our annotators made their decisions based on evidence in the signal
  for the specific segment chosen. The EEG report contains a finding
  based on the patient history and overall EEG session.

DEMOGRAPHICS:

 This section contains general information about the patients' age and
 gender. It is important to point out that the information is reported
 by patient. Since the data spans over several years, some patients
 might be represented more than once (with different ages) in the age
 section.

 Gender Statistics (reported by patient):

  Evaluation:

   |--------------------------------------------|
   | Description  |    Files     |   Patients   |
   |--------------+--------------+--------------+
   | (F) Abnormal |    63 ( 23%) |    51 ( 20%) |
   |--------------+--------------+--------------+
   | (M) Abnormal |    63 ( 23%) |    54 ( 21%) |
   |--------------+--------------+--------------+
   | (F) Normal   |    85 ( 31%) |    84 ( 34%) |
   |--------------+--------------+--------------+
   | (M) Normal   |    65 ( 23%) |    64 ( 25%) |
   |--------------+--------------+--------------+
   | Total        |   276 (100%) |   253 (100%) |
   |--------------------------------------------|

  Train:

   |--------------------------------------------|
   | Description  |    Files     |   Patients   |
   |--------------+--------------+--------------+
   | (F) Abnormal |   679 ( 25%) |   454 ( 21%) |
   |--------------+--------------+--------------+
   | (M) Abnormal |   667 ( 25%) |   439 ( 21%) |
   |--------------+--------------+--------------+
   | (F) Normal   |   768 ( 28%) |   691 ( 32%) |
   |--------------+--------------+--------------+
   | (M) Normal   |   603 ( 22%) |   546 ( 26%) |
   |--------------+--------------+--------------+
   | Total        | 2,717 (100%) | 2,130 (100%) |
   |--------------------------------------------|

 Age Distribution:

  Below is a distribution of patient age based on the first session
  for each patient:

   |----------------------------------------------------------| 
   |              |                   Count                   |
   |              |---------------------+---------------------|
   |              |      Evaluation     |        Train        | 
   |     Age      |----------+----------+----------+----------|
   | Distribution | Abnormal |  Normal  | Abnormal |  Normal  |
   |--------------+----------+----------+----------+----------|
   |         0-10 |        0 |        0 |        5 |        3 |
   |        10-20 |        2 |        4 |       15 |       39 |
   |        20-30 |        6 |       27 |       85 |      239 |
   |        30-40 |       10 |       37 |       80 |      225 |
   |        40-50 |       20 |       27 |      151 |      368 |
   |        50-60 |       21 |       23 |      201 |      237 |
   |        60-70 |       13 |       17 |      171 |      139 |
   |        70-80 |       18 |        7 |      116 |       49 |
   |        80-90 |       14 |        5 |       63 |       34 |
   |       90-100 |        1 |        1 |        6 |        4 |
   |--------------+----------+----------+----------+----------|
   |        TOTAL |      105 |      148 |      893 |    1,237 |
   |----------------------------------------------------------| 

  Since sessions can be separated in time by a significant amount of
  time (often years), below is a distribution of age by session:

   |----------------------------------------------------------| 
   |              |                   Count                   |
   |              |---------------------+---------------------|
   |              |      Evaluation     |        Train        | 
   |     Age      |----------+----------+----------+----------|
   | Distribution | Abnormal |  Normal  | Abnormal |  Normal  |
   |--------------+----------+----------+----------+----------|
   |         0-10 |        0 |        0 |        5 |        3 |
   |        10-20 |        2 |        4 |       19 |       43 |
   |        20-30 |        7 |       27 |      129 |      263 |
   |        30-40 |       11 |       38 |      110 |      252 |
   |        40-50 |       25 |       27 |      225 |      310 |
   |        50-60 |       28 |       23 |      310 |      260 |
   |        60-70 |       14 |       18 |      286 |      146 |
   |        70-80 |       23 |        7 |      163 |       54 |
   |        80-90 |       15 |        5 |       93 |       36 |
   |       90-100 |        1 |        1 |        6 |        4 |
   |--------------+----------+----------+----------+----------|
   |        TOTAL |      126 |      150 |    1,346 |    1,371 |
   |----------------------------------------------------------| 

---
If you have any additional comments or questions about this data, please direct
them to help@nedcdata.org.

Best regards, 

Eva von Weltin
NEDC Data Resources Development Manager
