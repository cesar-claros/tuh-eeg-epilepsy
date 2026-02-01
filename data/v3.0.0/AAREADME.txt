Files: AAREADME.txt
Database: TUH EEG Epilepsy Corpus
Version: v3.0.0

-------------------------------------------------------------------------------
Change Log:

 v3.0.0 (20260107): Signal data was annotated; metadata was included.
 v2.0.1 (20240207): Headers were modified. No change to the signal data.

-------------------------------------------------------------------------------
This file contains some basic statistics about the TUH EEG Epilepsy
Corpus, a corpus developed to motivate the development of new methods
for automatic analysis of EEG files using machine learning. This
corpus is a subset of the TUH EEG Corpus and contains sessions from
patients with epilepsy. To balance the corpus, some sessions are
provided from patients that do not have epilepsy.

Subjects were sorted into epilepsy and no epilepsy categories by searching
the associated EEG reports for indications as to an epilepsy/no epilepsy 
diagnosis based on clinical history, medications at the time of recording, 
and EEG features associated with epilepsy such as spike and sharp waves.
A board-certified neurologist, Daniel Goldenholz, and his research team
reviewed and verified the decisions about each patient.

When you use this specific corpus in your research or technology development, 
we ask that you reference the corpus using this publication:

 Veloso, L., McHugh, J. R., von Weltin, E., Obeid, I., & Picone,
 J. (2017). Big Data Resources for EEGs: Enabling Deep Learning
 Research. In I. Obeid & J. Picone (Eds.), Proceedings of the IEEE
 Signal Processing in Medicine and Biology Symposium
 (p. 1). Philadelphia, Pennsylvania, USA: IEEE.

This publication can be retrieved from:

 https://www.isip.piconepress.com/publications/conference_presentations/2017/ieee_spmb/data/

Our preferred reference for the TUH EEG Corpus, from which this
seizure corpus was derived, is:

 Obeid, I., & Picone, J. (2016). The Temple University Hospital EEG
 Data Corpus. Frontiers in Neuroscience, Section Neural Technology,
 10, 196.

v3.0.0 of the TUH EEG Epilesy Corpus was based on v2.0.1 of the
TUH EEG Corpus and v2.0.5 of the TUH EEG Corpus. Please see the documentation
for TUH EEG v2.0.1 to understand how the data is structured.

Our annotation guidelines are documented here:

 Melles, A.-M., Paderewski, M., Oymann, R., Shah, V., Obeid, I., &
 Picone, J. (2025). The Natus Medical Incorporated Ambulatory EEG
 Corpus: Annotation Guidelines (p. 16). Temple
 University.

which can be retrieved from here:

 https://isip.piconepress.com/publications/reports/2025/nmae/annotations/

BASIC STATISTICS:

  |--------------------------------------------------------|
  | Description | (00) Epilepsy | (01) No Epilepsy | Total |
  |-------------+---------------+------------------+-------|
  | Patients    |           100 |              100 |   200 |
  |-------------+---------------+------------------+-------|
  | Sessions    |           530 |              168 |   698 |
  |-------------+---------------+------------------+-------|
  | Files       |         2,257 |              564 | 2,821 |
  |--------------------------------------------------------|

The total size of the corpus is 36 Gbytes.

There are several new features of this version of the corpus. First,
the files have been limited to 30 mins. in durations. Second, the data
has been annotated for seizures following the conventions used in
TUSZ. Third, the EEG reports have been analyzed and summarized in a
spreadsheet in /DOCS.

The directory /DOCS contains a few new things. First, there are the
montages that are used to visualize and annotate the data:

 01_tcp_ar_montage.txt
 02_tcp_le_montage.txt
 03_tcp_ar_a_montage.txt
 04_tcp_le_a_montage.txt

Next, there is the metadata spreadsheet that contains information about
each session and subject, such as a diagnosis and medication history:

 metadata_v00r.xlsx

Entries are provided per session. It is not uncommon that there are differences
in some metadata between sessions. EEG reports are inherently noisy. We
report information found in the report for each session, whether or not
thst is consistent with the other sessions.

Finally, there are two lists:

 sessions_common_with_tusz.list
 sessions_unique_to_tuep.list

that sort sessions based on whether they appear in TUSZ.

There are three types of files in this release:

 *.edf:    the EEG sampled data in European Data Format (edf)
 *.csv:    event-based annotations using all available seizure type classes
 *.csv_bi: term-based annotations using only two labels (bckg and seiz)

These are described in more detail in the TUSZ Corpus.

Finally, here are some basic descriptive statistics about the data.
The commands used to generate these numbers are shown below.
For the commands below, the starting point was here:

 nedc_130_[1]: pwd
 /data/isip/data/tuh_eeg_epilepsy/v3.0.0

( 1) Number of files:

 nedc_130_[1]: find 00_* -name "*.edf" | wc -l
 2257
 nedc_130_[1]: find 01_* -name "*.edf" | wc -l
 564
 nedc_130_[1]: find 0?_* -name "*.edf" | wc -l
 2821

( 2) Number of sessions:

 nedc_130_[1]: find 00_* -mindepth 2 -maxdepth 2 | wc -l
 530
 nedc_130_[1]: find 01_* -mindepth 2 -maxdepth 2 | wc -l
 168
 nedc_130_[1]: find 0?_* -mindepth 2 -maxdepth 2 | wc -l
 698

( 3) Number of subjects:

 nedc_130_[1]: find 00_* -mindepth 1 -maxdepth 1 | wc -l
 100
 nedc_130_[1]: find 01_* -mindepth 1 -maxdepth 1 | wc -l
 100
 nedc_130_[1]: find 0?_* -mindepth 1 -maxdepth 1 | wc -l
 200

( 4) Number of files with seizures:

 nedc_130_[1]: find 00_* -name "*.csv" -exec grep -H "sz," {} \; | cut -d"/" -f5 | cut -d":" -f1 | sort -u | wc -l
 128
 nedc_130_[1]: find 01_* -name "*.csv" -exec grep -H "sz," {} \; | cut -d"/" -f5 | cut -d":" -f1 | sort -u | wc -l
 1
 nedc_130_[1]: find 0?_* -name "*.csv" -exec grep -H "sz," {} \; | cut -d"/" -f5 | cut -d":" -f1 | sort -u | wc -l
 129

( 5) Number of sessions with seizures:

 nedc_130_[1]: find 00_* -name "*.csv" -exec grep -H "sz," {} \; | cut -d"/" -f2,3 | sort -u | wc -l
 45
 nedc_130_[1]: find 01_* -name "*.csv" -exec grep -H "sz," {} \; | cut -d"/" -f2,3 | sort -u | wc -l
 0
 nedc_130_[1]: find 0?_* -name "*.csv" -exec grep -H "sz," {} \; | cut -d"/" -f2,3 | sort -u | wc -l
 45

( 6) Number of patients with seizures:

 nedc_130_[1]: find 00_* -name "*.csv" -exec grep -H "sz," {} \; | cut -d"/" -f2 | sort -u | wc -l
 14
 nedc_130_[1]: find 01_* -name "*.csv" -exec grep -H "sz," {} \; | cut -d"/" -f2 | sort -u | wc -l
 0
 nedc_130_[1]: find 0?_* -name "*.csv" -exec grep -H "sz," {} \; | cut -d"/" -f2 | sort -u | wc -l
 14

( 7) Total number of seizure events (measured using *.csv_bi):

 nedc_130_[1]: find 00_* -name "*.csv_bi" -exec grep -H seiz {} \; | wc -l
 351
 nedc_130_[1]: find 01_* -name "*.csv_bi" -exec grep -H seiz {} \; | wc -l
 0
 nedc_130_[1]: find 0?_* -name "*.csv_bi" -exec grep -H seiz {} \; | wc -l
 351

( 8) Total duration (in secs):

 nedc_130_[1]: find 00_* -name "*.csv" -exec grep duration {} \; | awk '{ sum+=$4} END {print sum}'
 1909278
 nedc_130_[1]: find 01_* -name "*.csv" -exec grep duration {} \; | awk '{ sum+=$4} END {print sum}'
 365369
 nedc_130_[1]: find 0?_* -name "*.csv" -exec grep duration {} \; | awk '{ sum+=$4} END {print sum}'
 2274647

( 9) Total size of the corpus (00_* + 01_*):

 nedc_130_[1]: cd  /data/isip/data/tuh_eeg_seizure/
 nedc_130_[1]: du -sBM v3.0.0
 36448M	v3.0.0

(10) Total duration of seizure events (in secs):

 nedc_130_[1]: find 00_* -name "*.csv_bi" -exec grep -H "seiz," {} \; | cut -d"," -f2,3 | sed -e "s/,/ /g" | awk '{ sum +=($2-$1)} END {print sum}'
 20728.2
 nedc_130_[1]: find 01_* -name "*.csv_bi" -exec grep -H "seiz," {} \; | cut -d"," -f2,3 | sed -e "s/,/ /g" | awk '{ sum +=($2-$1)} END {print sum}'

 nedc_130_[1]: find 0?_* -name "*.csv_bi" -exec grep -H "seiz," {} \; | cut -d"," -f2,3 | sed -e "s/,/ /g" | awk '{ sum +=($2-$1)} END {print sum}'
 20728.2

-----------------------------

If you have any additional comments or questions about the data,
please direct them to help@nedcdata.org.

Best regards,

Joe Picone
