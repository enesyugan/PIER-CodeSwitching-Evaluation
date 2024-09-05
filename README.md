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
3. [Examples](#examples)
4. [Contributing](#contributing)
5. [License](#license)
6. [Citations](#citations)

---

## Installation

Clone this repository and install the dependencies:

```bash
git clone https://github.com/<your-github-username>/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt



## Usage

The most simple use-case is computing the word error rate between two strings:

```python
from jiwer import wer

reference = "hello world"
hypothesis = "hello duck"

error = wer(reference, hypothesis)
```

## Licence

The jiwer package is released under the `Apache License, Version 2.0`.

For further information, see [`LICENCE`](./LICENSE).

