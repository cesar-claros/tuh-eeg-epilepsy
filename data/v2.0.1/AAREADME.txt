File: AAREADME.txt
Database: TUH EEG Event Corpus
Version: 2.0.1
-------------------------------------------------------------------------------

This file contains some basic statistics about the TUH EEG Event
Corpus. This is a subset of the TUH EEG Corpus and contains sessions
that are known to contain events including periodic lateralized
epileptiform discharge, generalized periodic epileptiform discharge,
spike and/or sharp wave discharges, artifact, and eye movement. This
version includes fixed edf files that previously had invalid headers
which were causing problems.

When you use this specific corpus in your research or technology development, 
we ask that you reference the corpus using this publication:

 Harati, A., Golmohammadi, M., Lopez, S., Obeid, I., & Picone, J. (2015). 
 Improved EEG Event Classification Using Differential Energy. Proceedings of 
 the IEEE Signal Processing in Medicine and Biology Symposium (pp. 1-4). 
 Philadelphia, Pennsylvania, USA. 

This publication can be retrieved from: 

 https://www.isip.piconepress.com/publications/conference_proceedings/2015/ieee_spmb/denergy/

Our preferred reference for the TUH EEG Corpus, from which this event corpus 
was derived, is:

 Obeid, I., & Picone, J. (2016). The Temple University Hospital EEG
 Data Corpus. Frontiers in Neuroscience, Section Neural Technology,
 10, 196. http://doi.org/http://dx.doi.org/10.3389/fnins.2016.00196

There are two main directories in this release: train and eval. The training 
directory contains data that you are allowed to use for the development of 
your technology. The evaluation set is disjoint from the training set and 
should only be used for testing. 

The pathname of a typical eval EEG file can be explained as follows:

Filename: ./edf/eval/032/bckg_032_a_.edf

Components:

edf: directory containing the edf files.

eval: part of the eval set (vs. train).

032: a random index to differentiate each evaluation set session.

bckg_032_a_.edf: the actual EEG file.
		bckg: this file contains background annotations.
		032: a reference to the eval index	
		a_.edf: EEG files are split into a series of files starting 
		with a_.edf, a_1.ef, ... These represent pruned EEGs, so the 
		original EEG is split into these segments, and uninteresting 
		parts of the original recording were deleted.

The pathname of typical train EEG file can be explained as follows:
    
Filename: ./edf/train/00002275/00002275_00000001.edf

Components:

edf: directory containing the edf files.

train: part of the train set.

00002275: an index that crossreferences this patient to v0.6.1 of the TUH EEG
	  Corpus. 

00002275_00000001.edf: The actual edf file. 
		      00002275: a reference to the train index. 
		      00000001: indicating that this is the first file in
		      associated with this patient. 

There are six types of files in this release:

 *.edf: the EEG sampled data in European Data Format (edf) 

 *.htk: feature extraction based on the approach explained in "Improved EEG 
        Event Classification Using Differential Energy".

 *.lab: annotation file with a label given for every 10 microseconds. named 
        according to channel number. 

 *.rec: annotation file with labels given in seconds. 

lab files use 4 letter codes: 

 spsw: spike and slow wave
 gped: generalized periodic epileptiform discharge
 pled: periodic lateralized epileptiform dischage
 eyem: eye movement
 artf: artifact
 bckg: background

In the format:

 117100000 117200000 eyem

The fields are: start and stop time in 10s of microseconds and label

rec files use numeric codes:

 1: spsw
 2: gped
 3: pled
 4: eyem
 5: artf
 6: bckg

In the format:

   13,90.4,91.4,6

The fields are: channel number, start time in seconds, stop time in seconds, 
and label. 

labels 1-5 (spsw, gped, pled, eyem, artf) are clear examples of the 
annotated classes. background (bckg) is the annotation used when the 
event is clearly not any of the other five classes, so the bckg label can be
seen as a catch-all class.

Clinical EEGs use a variety of channel configurations. In the larger TUH EEG 
Corpus, there are over 40 different channel configurations. In this subset, 
there are two type of EEGs: averaged reference (AR) and linked ears reference 
(LE). Fortunately, all files in this subset contain the standard channels you 
would expect from a 10/20 configuration, and all files can be converted to a 
TCP montage (which is what we use internally for our processing).

To learn more about this, please consult the following publication:

 Lopez, S., Gross, A., Yang, S., Golmohammadi, M., Obeid,I., &
 Picone, J. (2016). An Analysis of Two Common Reference Points for
 EEGs. In IEEE Signal Processing in Medicine and Biology Symposium
 (pp. 1–4). Philadelphia, Pennsylvania, USA. Available at:
 https://www.isip.piconepress.com/publications/conference_proceedings/2016/ieee_spmb/montages/.

The channel number in .rec and .lab files refers to the channels
defined using a standard ACNS TCP montage. This is our preferred way
of viewing seizure data. The montage is defined as follows:

 montage =  0, FP1-F7: EEG FP1-REF --  EEG F7-REF
 montage =  1, F7-T3:  EEG F7-REF  --  EEG T3-REF
 montage =  2, T3-T5:  EEG T3-REF  --  EEG T5-REF
 montage =  3, T5-O1:  EEG T5-REF  --  EEG O1-REF
 montage =  4, FP2-F8: EEG FP2-REF --  EEG F8-REF
 montage =  5, F8-T4 : EEG F8-REF  --  EEG T4-REF
 montage =  6, T4-T6:  EEG T4-REF  --  EEG T6-REF
 montage =  7, T6-O2:  EEG T6-REF  --  EEG O2-REF
 montage =  8, A1-T3:  EEG A1-REF  --  EEG T3-REF
 montage =  9, T3-C3:  EEG T3-REF  --  EEG C3-REF
 montage = 10, C3-CZ:  EEG C3-REF  --  EEG CZ-REF
 montage = 11, CZ-C4:  EEG CZ-REF  --  EEG C4-REF
 montage = 12, C4-T4:  EEG C4-REF  --  EEG T4-REF
 montage = 13, T4-A2:  EEG T4-REF  --  EEG A2-REF
 montage = 14, FP1-F3: EEG FP1-REF --  EEG F3-REF
 montage = 15, F3-C3:  EEG F3-REF  --  EEG C3-REF
 montage = 16, C3-P3:  EEG C3-REF  --  EEG P3-REF
 montage = 17, P3-O1:  EEG P3-REF  --  EEG O1-REF
 montage = 18, FP2-F4: EEG FP2-REF --  EEG F4-REF
 montage = 19, F4-C4:  EEG F4-REF  --  EEG C4-REF
 montage = 20, C4-P4:  EEG C4-REF  --  EEG P4-REF
 montage = 21, P4-O2:  EEG P4-REF  --  EEG O2-REF

For example, channel 1 is a difference between electrodes F7 and T3,
and represents an arithmetic difference of the channels
(F7-REF)-(T3-REF), which are channnels contained in the EDF file.

Finally, here are some basic descriptive statistics about the data:

EVALUATION SET:

files: 159
       containing spsw: 9
       containing gped: 28
       containing pled: 33
       containing artf: 46
       containing eyem: 35 
       containing bckg: 89

TRAINING SET:

files: 359
       containing spsw: 27
       containing gped: 51
       containing pled: 48
       containing artf: 164
       containing eyem: 46 
       containing bckg: 211

---
If you have any additional comments or questions about this data, please direct
them to help@nedcdata.org.

Best regards, 

Sean Ferrell
NEDC Data Resources Development Manager
