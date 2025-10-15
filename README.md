# <Your Project Name> 

[![License](https://img.shields.io/github/license/enesyugan/PIER-CodeSwitching-Evaluation)](https://github.com/enesyugan/PIER-CodeSwitching-Evaluation/blob/master/LICENSE)
[![Issues](https://img.shields.io/github/issues/enesyugan/PIER-CodeSwitching-Evaluation)](https://github.com/enesyugan/PIER-CodeSwitching-Evaluation/issues)
[![Stars](https://img.shields.io/github/stars/enesyugan/PIER-CodeSwitching-Evaluation)](https://github.com/enesyugan/PIER-CodeSwitching-Evaluation/stargazers)

## Overview
PIER (Point-of-Interest Error Rate) is a variant of Word-Error-Rate tailored for code-switching ASR: rather than scoring all words, PIER first tags a set of “points of interest” (e.g. the embedded-language tokens), computes the usual alignment between reference and hypothesis, then counts only the edit operations whose reference positions lie in that set, normalizing by the number of points of interest to yield an error rate focused purely on the code-switched segments.

> This repository is a fork of the [jiwer](https://github.com/jitsi/jiwer) library with added functionalities and modifications for PIER task. It enables calculating Point of Inerest Error Rate (PIER). The usage is the same as in WER/CER by jiwer.

## Features

- **Feature 1**: Calculating PIER.
- **Feature 2**: You can provide tagged refernce such as ["This is a \<tag reference\> sentence."]
- **Feature 3**: You can specifiy second language next to english and we automatically determine english (latin) as points-of-interest (words of embedded language).

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [License](#license)
4. [Citations](#citations)
4. [Contact](#contact)

---

## Installation

Clone this repository and install the dependencies:

```bash
git clone https://github.com/enesyugan/PIER-CodeSwitching-Evaluation.git
cd PIER-CodeSwitching-Evaluation
pip install -r requirements.txt

```

## Usage

Currently we only support mixing with English and X (X being any other language) for automatic tagging.
Otherwise you can provide tagged words it will calculate PIER for those words as well.

The most simple use-case is computing the Point-of-Interest Error Rate between two strings.

For languages that share the same latin script.
```python
import sys
sys.path.append(<path of repo code>/jiwer)
from measures import pier

# (Yea, that thing with the bots i don't believe it.)
reference = "Ja, das mit den <tag Bots> glaube ich nicht."
hypothesis = "Ja, das mit den Pots glaub ich nicht."

error = pier(reference, hypothesis)

```
This example was taken from ["DECM: Evaluating Bilingual ASR Performance on a Code-switching/mixing Benchmark"](https://aclanthology.org/2024.lrec-main.400.pdf).


For languages with differen writing scripts such as Arabic or Mandarin, Japanese taggs are not needed.
For Han/Kanji, Hiragana, Katakana spaces should be inserted between characters.

<!--The matrix language will be determined on corpus level and the PIER performance is calculated on the embedded langauge.-->
The matrix language will be set to the non-latin script and the PIER performance is calculated on the embedded langauge.

```python
import regex
import inflect
import re
import sys
sys.path.append(<path of the this code>/jiwer)
from measures import pier

def tokenize_for_mer(text):
  # Match: single Han char | number | simple English word (+ optional apostrophe chunk)
  reg_range = r"[\u4e00-\ufaff]|[0-9]+|[a-zA-Z]+\'*[a-z]*"
  matches = re.findall(reg_range, text, re.UNICODE)
  p = inflect.engine()
  res = []
  for item in matches:
    try:
      # Convert pure numerics that are not Han
      temp = p.number_to_words(item) if (item.isnumeric() and len(regex.findall(r'\p{Han}+', item)) == 0) else item
    except:
      temp = item
    res.append(temp)
  return res

reference = "我是从 camp 那边拿来的自从 mark 那时拿来了之后"
hypothesis = "是從cam那邊拿來的是從marc拿來的之後"

# Tokenize & normalize for PIER
reference = " ".join(tokenize_for_mer(reference)) 
hypothesis = " ".join(tokenize_for_mer(hypothesis)) 

error = pier(reference, hypothesis, scd_language="cmn")
```
This example was taken from ["SEAME:a mandarin-english code-switching speech corpus in south-east asia."](https://www.isca-archive.org/interspeech_2010/lyu10_interspeech.pdf).

## Citations

If you use this project in your work, please cite it as follows:

```bibtex
@inproceedings{ugan2025pier,
  title={Pier: A novel metric for evaluating what matters in code-switching},
  author={Ugan, Enes Yavuz and Pham, Ngoc-Quan and B{\"a}rmann, Leonard and Waibel, Alex},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```

## Contact

If you have any questions or issues, feel free to [open an issue](https://github.com/enesyugan/PIER-CodeSwitching-Evaluation/issues) or reach out to me at [enes.ugan@kit.edu].


## License

The jiwer package is released under the `Apache License, Version 2.0`.

For further information, see [`LICENCE`](./LICENSE).

