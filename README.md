# <Your Project Name> 

[![License](https://img.shields.io/github/license/enesyugan/PIER-CodeSwitching-Evaluation)](https://github.com/<your-github-username>/<your-repo-name>/blob/master/LICENSE)
[![Issues](https://img.shields.io/github/issues/enesyugan/PIER-CodeSwitching-Evaluation)](https://github.com/<your-github-username>/<your-repo-name>/issues)
[![Stars](https://img.shields.io/github/stars/enesyugan/PIER-CodeSwitching-Evaluation)](https://github.com/<your-github-username>/<your-repo-name>/stargazers)

## Overview

A brief description of your project. Explain what it does, why it matters, and any key details that make it unique.

_For example:_
> This repository is a fork of the [jiwer](https://github.com/jitsi/jiwer) library with added functionalities and modifications for X task. The project is designed to calculate word error rates (WER) and other metrics for evaluating automatic speech recognition (ASR) systems.

## Features

- **Feature 1**: Calculating PIER.
- **Feature 2**: You can proide tagged refernce such as ["This is a <tag reference> sentence."]
- **Feature 3**: You can specifiy second language next to english and we automatically determine matrix language and points-of-interest (words of embedded language).

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
cd jiwer
pip install -r requirements.txt

```

## Usage

Currently we only support mixing with English and X (X being any other language).

The most simple use-case is computing the word error rate between two strings:


For Languages pairs that are both 
```python
from jiwer import wer

#(Did you download the file?)
reference = "Hast du das <tag File> <tag gedownloadet>?"
hypothesis = "Hast du das File geload et"

error = per(reference, hypothesis)

```

For languages with differen writing scripts such as Arabic or Mandarin, Japanese taggs are not needed.
For Han/Kanji, Hiragana, Katakana spaces are inserted between characters.

The matrix language will be determined on corpus level and the PIER performance is calculated on the embedded langauge.

```python
from jiwer import wer

#(Did you download the file?)
reference = "我是从 camp 那边拿来的自从 mark 那时拿来了之后"
hypothesis = "是從cam那邊拿來的是從marc拿來的之後"


error = per(reference, hypothesis)

```
## Citations

If you use this project in your work, please cite it as follows:

```bibtex
@article{your-citation-key,
  author    = {Your Name},
  title     = {Title of Your Work},
  journal   = {Journal Name},
  volume    = {Volume Number},
  number    = {Issue Number},
  pages     = {Page Numbers},
  year      = {Year},
  publisher = {Publisher},
  doi       = {DOI}
}
```

## Contact

If you have any questions or issues, feel free to [open an issue](https://github.com/enesyugan/PIER-CodeSwitching-Evaluation/issues) or reach out to me at [enes.ugan@kit.edu].


## License

The jiwer package is released under the `Apache License, Version 2.0`.

For further information, see [`LICENCE`](./LICENSE).

