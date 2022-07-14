[![.NET](https://github.com/zamgi/lingvo--PosTagger-NER-ru-dnn/actions/workflows/dotnet.yml/badge.svg)](https://github.com/zamgi/lingvo--PosTagger-NER-ru-dnn/actions/workflows/dotnet.yml)

# PosTagger
Part of speech tagging of words in Russian language using deep neural network in C# for .NET

A tensors-based deep neural network used for PoS-tagging (sequence-labeling task) text in Russian based on word endings.
Supports both CPU and GPU computing.

#
Metrics for includes models:

 Custom markup corpus (sents = 41 989):
```
Common-F-Score = '89.41'

Adjective          : F-score = '90.11' Precision = '88.65' Recall = '91.62'
AdjectivePronoun   : F-score = '87.77' Precision = '88.18' Recall = '87.37'
Adverb             : F-score = '85.78' Precision = '86.04' Recall = '85.51'
AdverbialParticiple: F-score = '91.01' Precision = '92.47' Recall = '89.58'
AdverbialPronoun   : F-score = '83.15' Precision = '85.71' Recall = '80.74'
AuxiliaryVerb      : F-score = '93.38' Precision = '95.48' Recall = '91.36'
Conjunction        : F-score = '90.20' Precision = '88.89' Recall = '91.55'
Infinitive         : F-score = '97.38' Precision = '96.97' Recall = '97.80'
Interjection       : F-score = '80.00' Precision = '93.33' Recall = '70.00'
Noun               : F-score = '97.13' Precision = '97.45' Recall = '96.81'
Numeral            : F-score = '93.60' Precision = '93.78' Recall = '93.41'
Other              : F-score = '77.41' Precision = '80.76' Recall = '74.32'
Participle         : F-score = '68.52' Precision = '71.58' Recall = '65.71'
Particle           : F-score = '80.78' Precision = '83.27' Recall = '78.44'
PossessivePronoun  : F-score = '92.47' Precision = '90.39' Recall = '94.65'
Predicate          : F-score = '92.57' Precision = '91.33' Recall = '93.84'
Preposition        : F-score = '98.58' Precision = '98.07' Recall = '99.09'
Pronoun            : F-score = '91.82' Precision = '91.58' Recall = '92.05'
Punctuation        : F-score = '99.87' Precision = '99.83' Recall = '99.91'
Verb               : F-score = '96.76' Precision = '96.42' Recall = '97.10'

The number of part of speech categories = '20'
```
 "nerus_lenta.conllu" corpus (sents = 8 066 461):
```
Common-F-Score = '95.11'

ADJ  : F-score = '97.79' Precision = '97.09' Recall = '98.51'
ADP  : F-score = '99.90' Precision = '99.84' Recall = '99.96'
ADV  : F-score = '98.03' Precision = '98.75' Recall = '97.33'
AUX  : F-score = '99.35' Precision = '99.30' Recall = '99.40'
CCONJ: F-score = '99.64' Precision = '99.47' Recall = '99.82'
DET  : F-score = '97.24' Precision = '96.83' Recall = '97.64'
INTJ : F-score = '58.33' Precision = '77.78' Recall = '46.67'
NOUN : F-score = '98.19' Precision = '96.99' Recall = '99.42'
NUM  : F-score = '98.66' Precision = '99.04' Recall = '98.28'
PART : F-score = '98.21' Precision = '98.69' Recall = '97.74'
PRON : F-score = '98.75' Precision = '99.22' Recall = '98.29'
PROPN: F-score = '93.65' Precision = '98.27' Recall = '89.45'
PUNCT: F-score = '99.95' Precision = '99.95' Recall = '99.95'
SCONJ: F-score = '99.29' Precision = '99.22' Recall = '99.36'
SYM  : F-score = '86.54' Precision = '89.11' Recall = '84.11'
VERB : F-score = '98.47' Precision = '98.76' Recall = '98.19'
X    : F-score = '94.86' Precision = '94.52' Recall = '95.20'

The number of categories = '17'
```
#

Included PosTagger UI sample:
![alt tag](https://github.com/zamgi/lingvo--PosTagger-ru-dnn/blob/master/pos_tagger_ru.png)


# NER
Named-entity recognition in Russian language using deep neural network in C# for .NET

#
Metrics for includes models:

 "nerus_lenta.conllu" corpus (sents = 500 000):
```
Common-F-Score = '94.30'

B-LOC: F-score = '97.37' Precision = '97.88' Recall = '96.87'
B-ORG: F-score = '92.90' Precision = '93.34' Recall = '92.47'
B-PER: F-score = '96.21' Precision = '97.37' Recall = '95.08'
I-LOC: F-score = '91.90' Precision = '94.68' Recall = '89.28'
I-ORG: F-score = '90.43' Precision = '89.45' Recall = '91.43'
I-PER: F-score = '96.98' Precision = '97.54' Recall = '96.42'

The number of categories = '6'
```
 "nerus_lenta.conllu" corpus (sents = 1 000 000):
```
Common-F-Score = '96.78'

B-LOC: F-score = '98.46' Precision = '98.54' Recall = '98.39'
B-ORG: F-score = '95.22' Precision = '96.10' Recall = '94.35'
B-PER: F-score = '98.71' Precision = '99.02' Recall = '98.40'
I-LOC: F-score = '94.67' Precision = '95.63' Recall = '93.73'
I-ORG: F-score = '94.43' Precision = '94.92' Recall = '93.95'
I-PER: F-score = '98.94' Precision = '98.84' Recall = '99.04'

The number of categories = '6'
```
#

Included NER UI sample:
![alt tag](https://github.com/zamgi/lingvo--PosTagger-ru-dnn/blob/master/ner_ru.png)
