#
# JiWER - Jitsi Word Error Rate
#
# Copyright @ 2018 - present 8x8, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
The core algorithm(s) for processing a one or more reference and hypothesis sentences
so that measures can be computed and an alignment can be visualized.
"""

from dataclasses import dataclass

from typing import Any, List, Union
from itertools import chain

import rapidfuzz

from rapidfuzz.distance import Opcodes

from jiwer import transforms as tr
from jiwer.transformations import wer_default, cer_default
import re
import inflect

__all__ = [
    "AlignmentChunk",
    "WordOutput",
    "CharacterOutput",
    "process_words",
    "process_characters",
]


@dataclass
class AlignmentChunk:
    """
    Define an alignment between two subsequence of the reference and hypothesis.

    Attributes:
        type: one of `equal`, `substitute`, `insert`, or `delete`
        ref_start_idx: the start index of the reference subsequence
        ref_end_idx: the end index of the reference subsequence
        hyp_start_idx: the start index of the hypothesis subsequence
        hyp_end_idx: the end index of the hypothesis subsequence
    """

    type: str

    ref_start_idx: int
    ref_end_idx: int

    hyp_start_idx: int
    hyp_end_idx: int

    def __post_init__(self):
        if self.type not in ["replace", "insert", "delete", "equal", "substitute"]:
            raise ValueError("")

        # rapidfuzz uses replace instead of substitute... For consistency, we change it
        if self.type == "replace":
            self.type = "substitute"

        if self.ref_start_idx > self.ref_end_idx:
            raise ValueError(
                f"ref_start_idx={self.ref_start_idx} "
                f"is larger "
                f"than ref_end_idx={self.ref_end_idx}"
            )
        if self.hyp_start_idx > self.hyp_end_idx:
            raise ValueError(
                f"hyp_start_idx={self.hyp_start_idx} "
                f"is larger "
                f"than hyp_end_idx={self.hyp_end_idx}"
            )


@dataclass
class WordOutput:
    """
    The output of calculating the word-level levenshtein distance between one or more
    reference and hypothesis sentence(s).

    Attributes:
        references: The reference sentences
        hypotheses: The hypothesis sentences
        alignments: The alignment between reference and hypothesis sentences
        wer: The word error rate
        mer: The match error rate
        wil: The word information lost measure
        wip: The word information preserved measure
        hits: The number of correct words between reference and hypothesis sentences
        substitutions: The number of substitutions required to transform hypothesis
                       sentences to reference sentences
        insertions: The number of insertions required to transform hypothesis
                       sentences to reference sentences
        deletions: The number of deletions required to transform hypothesis
                       sentences to reference sentences

    """

    # processed input data
    references: List[List[str]]
    hypotheses: List[List[str]]

    # alignment
    alignments: List[List[AlignmentChunk]]

    # measures
    wer: float
    mer: float
    wil: float
    wip: float

    # stats
    hits: int
    substitutions: int
    insertions: int
    deletions: int


def process_words(
    reference: Union[str, List[str]],
    hypothesis: Union[str, List[str]],
    reference_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
) -> WordOutput:
    """
    Compute the word-level levenshtein distance and alignment between one or more
    reference and hypothesis sentences. Based on the result, multiple measures
    can be computed, such as the word error rate.

    Args:
        reference: The reference sentence(s)
        hypothesis: The hypothesis sentence(s)
        reference_transform: The transformation(s) to apply to the reference string(s)
        hypothesis_transform: The transformation(s) to apply to the hypothesis string(s)

    Returns:
        (WordOutput): The processed reference and hypothesis sentences
    """
    # validate input type
    if isinstance(reference, str):
        reference = [reference]
    if isinstance(hypothesis, str):
        hypothesis = [hypothesis]
    if any(len(t) == 0 for t in reference):
        raise ValueError("one or more references are empty strings")

    # pre-process reference and hypothesis by applying transforms
    ref_transformed = _apply_transform(
        reference, reference_transform, is_reference=True
    )
    hyp_transformed = _apply_transform(
        hypothesis, hypothesis_transform, is_reference=False
    )

    if len(ref_transformed) != len(hyp_transformed):
        raise ValueError(
            "After applying the transforms on the reference and hypothesis sentences, "
            f"their lengths must match. "
            f"Instead got {len(ref_transformed)} reference and "
            f"{len(hyp_transformed)} hypothesis sentences."
        )

    # Change each word into a unique character in order to compute
    # word-level levenshtein distance
    ref_as_chars, hyp_as_chars = _word2char(ref_transformed, hyp_transformed)

    # keep track of total hits, substitutions, deletions and insertions
    # across all input sentences
    num_hits, num_substitutions, num_deletions, num_insertions = 0, 0, 0, 0

    # also keep track of the total number of words in the reference and hypothesis
    num_rf_words, num_hp_words = 0, 0

    # anf finally, keep track of the alignment between each reference and hypothesis
    alignments = []

    for reference_sentence, hypothesis_sentence in zip(ref_as_chars, hyp_as_chars):
        # Get the required edit operations to transform reference into hypothesis
        edit_ops = rapidfuzz.distance.Levenshtein.editops(
            reference_sentence, hypothesis_sentence
        )

        # count the number of edits of each type
        substitutions = sum(1 if op.tag == "replace" else 0 for op in edit_ops)
        deletions = sum(1 if op.tag == "delete" else 0 for op in edit_ops)
        insertions = sum(1 if op.tag == "insert" else 0 for op in edit_ops)
        hits = len(reference_sentence) - (substitutions + deletions)

        # update state
        num_hits += hits
        num_substitutions += substitutions
        num_deletions += deletions
        num_insertions += insertions
        num_rf_words += len(reference_sentence)
        num_hp_words += len(hypothesis_sentence)
        alignments.append(
            [
                AlignmentChunk(
                    type=op.tag,
                    ref_start_idx=op.src_start,
                    ref_end_idx=op.src_end,
                    hyp_start_idx=op.dest_start,
                    hyp_end_idx=op.dest_end,
                )
                for op in Opcodes.from_editops(edit_ops)
            ]
        )

    # Compute all measures
    S, D, I, H = num_substitutions, num_deletions, num_insertions, num_hits

    wer = float(S + D + I) / float(H + S + D)
    mer = float(S + D + I) / float(H + S + D + I)
    wip = (
        (float(H) / num_rf_words) * (float(H) / num_hp_words)
        if num_hp_words >= 1
        else 0
    )
    wil = 1 - wip

    # return all output
    return WordOutput(
        references=ref_transformed,
        hypotheses=hyp_transformed,
        alignments=alignments,
        wer=wer,
        mer=mer,
        wil=wil,
        wip=wip,
        hits=num_hits,
        substitutions=num_substitutions,
        insertions=num_insertions,
        deletions=num_deletions,
    )


########################################################################################
# Implementation of character error rate


@dataclass
class CharacterOutput:
    """
    The output of calculating the character-level levenshtein distance between one or
    more reference and hypothesis sentence(s).

    Attributes:
        references: The reference sentences
        hypotheses: The hypothesis sentences
        alignments: The alignment between reference and hypothesis sentences
        cer: The character error rate
        hits: The number of correct characters between reference and hypothesis
              sentences
        substitutions: The number of substitutions required to transform hypothesis
                       sentences to reference sentences
        insertions: The number of insertions required to transform hypothesis
                       sentences to reference sentences
        deletions: The number of deletions required to transform hypothesis
                       sentences to reference sentences
    """

    # processed input data
    references: List[List[str]]
    hypotheses: List[List[str]]

    # alignment
    alignments: List[List[AlignmentChunk]]

    # measures
    cer: float

    # stats
    hits: int
    substitutions: int
    insertions: int
    deletions: int


def process_characters(
    reference: Union[str, List[str]],
    hypothesis: Union[str, List[str]],
    reference_transform: Union[tr.Compose, tr.AbstractTransform] = cer_default,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = cer_default,
) -> CharacterOutput:
    """
    Compute the character-level levenshtein distance and alignment between one or more
    reference and hypothesis sentences. Based on the result, the character error rate
    can be computed.

    Note that the by default this method includes space (` `) as a
    character over which the error rate is computed. If this is not desired, the
    reference and hypothesis transform need to be modified.

    Args:
        reference: The reference sentence(s)
        hypothesis: The hypothesis sentence(s)
        reference_transform: The transformation(s) to apply to the reference string(s)
        hypothesis_transform: The transformation(s) to apply to the hypothesis string(s)

    Returns:
        (CharacterOutput): The processed reference and hypothesis sentences.

    """
    # make sure the transforms end with tr.ReduceToListOfListOfChars(),

    # it's the same as word processing, just every word is of length 1
    result = process_words(
        reference, hypothesis, reference_transform, hypothesis_transform
    )

    return CharacterOutput(
        references=result.references,
        hypotheses=result.hypotheses,
        alignments=result.alignments,
        cer=result.wer,
        hits=result.hits,
        substitutions=result.substitutions,
        insertions=result.insertions,
        deletions=result.deletions,
    )


def process_pier(
    reference: Union[str, List[str]],
    hypothesis: Union[str, List[str]],
    reference_transform: Union[tr.Compose, tr.AbstractTransform] = cer_default,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = cer_default,
	scd_language: str=None,
	split_hyphen: bool=False,
):
    """
    Compute word-level levenstein disatnace and alignment between one or more reference and hypothesis sentences.
    Based on tagged words relevent alignements are extrected and PER is calculated.
    """

    # validate input type
    if isinstance(reference, str):
        reference = [reference]
    if isinstance(hypothesis, str):
        hypothesis = [hypothesis]
    if any(len(t) == 0 for t in reference):
        raise ValueError("one or more references are empty strings")

    poi_indices, other_indices = [], []
    num_poi_words, num_other_words = [], []
    reference_notag = []

    matrix_lang = determine_matrix_language(reference, split_hyphen, scd_language) if scd_language!= None else None
    print(f"MATRIX LANG: {matrix_lang}")
    for ref in reference:
        poi_ind, o_ind = extract_indices(ref, split_hyphen, scd_language,  matrix_lang, fixedtags=True)
        #print(f"{ref} l: {len(ref.split())}\n{poi_ind}\n{o_ind}")
        poi_indices.append(poi_ind)
        other_indices.append(o_ind)
        num_poi_words.append(len(poi_indices))
        num_other_words.append(len(other_indices))
        ref_notag = re.sub(r'<tag (.*?)>', r'\1', ref)
        if split_hyphen:
            ref_notag = ref_notag.replace("-", " ")
        #reference_notag.append(re.sub(r'<tag (.*?)>', r'\1', ref))
        reference_notag.append(ref_notag)

    # pre-process reference and hypothesis by applying transforms
    ref_transformed = _apply_transform(
        reference_notag, reference_transform, is_reference=True
    )
    hyp_transformed = _apply_transform(
        hypothesis, hypothesis_transform, is_reference=False
    )
    
    if len(ref_transformed) != len(hyp_transformed):
        raise ValueError(
            "After applying the transforms on the reference and hypothesis sentences, "
            f"their lengths must match. "
            f"Instead got {len(ref_transformed)} reference and "
            f"{len(hyp_transformed)} hypothesis sentences."
        )

    # Change each word into a unique character in order to compute
    # word-level levenshtein distance
    ref_as_chars, hyp_as_chars = _word2char(ref_transformed, hyp_transformed)
    
    I, S, D = 0, 0, 0
    oI, oS, oD = 0, 0, 0
    poiWords, otherWords = 0, 0
    counter = 0
    H, oH = 0, 0

    for reference_sentence, hypothesis_sentence, poi_idxs, other_idxs in zip(ref_as_chars, hyp_as_chars, poi_indices, other_indices):
        poiWords += len(poi_idxs)
        otherWords += len(other_idxs)
        total_len = len(reference_notag[counter].split())

        edit_ops = rapidfuzz.distance.Levenshtein.editops(
                reference_sentence, hypothesis_sentence
                )
        #print(edit_ops)
        if len(poi_idxs) > 0:
            insertions, deletions, substitutions, hits = get_idsh(edit_ops, poi_idxs, total_len)

            S += substitutions
            I += insertions
            D += deletions
            H += hits

        if len(other_idxs) > 0:
            insertions, deletions, substitutions, hits = get_idsh(edit_ops, other_idxs, total_len)
            oS += substitutions
            oI += insertions
            oD += deletions
            oH += hits
        counter += 1

    PER = ((I+D+S)/(H+S+D))*100 if (H+S+D) > 0 else 0.
    oPER = ((oI+oD+oS)/(oH+oS+oD))*100 if (oH+oS+oD) > 0 else 0.
    
    res = {
        "poi": {
            "PER": PER,
            "insertions": I,
            "deletions": D,
            "substitutions":S,
            "hits": H,
            "poiWords": (H+S+D),
            },
        "rest": {
            "PER": oPER,
            "insertions": oI,
            "deletions": oD,
            "substitutions": oS,
            "hits": oH,
            "otherWords": (oH+oS+oD),
            }
        }

    return res


################################################################################
# Implementation of helper methods


def _apply_transform(
    sentence: Union[str, List[str]],
    transform: Union[tr.Compose, tr.AbstractTransform],
    is_reference: bool,
):
    # Apply transforms. The transforms should collapse input to a
    # list with lists of words
    transformed_sentence = transform(sentence)

    # Validate the output is a list containing lists of strings
    if is_reference:
        if not _is_list_of_list_of_strings(
            transformed_sentence, require_non_empty_lists=True
        ):
            raise ValueError(
                "After applying the transformation, each reference should be a "
                "non-empty list of strings, with each string being a single word."
            )
    else:
        if not _is_list_of_list_of_strings(
            transformed_sentence, require_non_empty_lists=False
        ):
            raise ValueError(
                "After applying the transformation, each hypothesis should be a "
                "list of strings, with each string being a single word."
            )

    return transformed_sentence


def _is_list_of_list_of_strings(x: Any, require_non_empty_lists: bool):
    if not isinstance(x, list):
        return False

    for e in x:
        if not isinstance(e, list):
            return False

        if require_non_empty_lists and len(e) == 0:
            return False

        if not all([isinstance(s, str) for s in e]):
            return False

    return True


def _word2char(reference: List[List[str]], hypothesis: List[List[str]]):
    # tokenize each word into an integer
    vocabulary = set(chain(*reference, *hypothesis))

    if "" in vocabulary:
        raise ValueError(
            "Empty strings cannot be a word. "
            "Please ensure that the given transform removes empty strings."
        )

    word2char = dict(zip(vocabulary, range(len(vocabulary))))

    reference_chars = [
        "".join([chr(word2char[w]) for w in sentence]) for sentence in reference
    ]
    hypothesis_chars = [
        "".join([chr(word2char[w]) for w in sentence]) for sentence in hypothesis
    ]

    return reference_chars, hypothesis_chars

def tokenize_for_mer(text):
    """
    split Hiragana, Katakana, Kanji/Han characters similar to Mixed-error-rate
    """
    #reg_range = r"[\u4e00-\ufaff]|[0-9]+|[a-zA-Z]+\'*[a-z]*"
    reg_range = r"[\u4E00-\u9FFF]|[\u3040-\u309F]|[\u30A0-\u30FF]|[\uFF00-\uFFEF]|[0-9]+|[a-zA-Z]+\'*[a-z]*"
    matches = re.findall(reg_range, text, re.UNICODE)
    p = inflect.engine()
    res = []
    for item in matches:
        try:
            temp = p.number_to_words(item) if (item.isnumeric() and len(regex.findall(r'\p{Han}+', item)) == 0) else item
        except:
            temp = item
        res.append(temp)
    return res


def tag_words(words, switch=False):
    latin_containing_pattern = r'\b\w*[a-zA-Z]+\w*\b'
    latin_pattern_with_numbers = r'\b[a-zA-Z0-9]+(?:\'[a-zA-Z0-9]+)?\b'

    num_words = len(words)
    eng_words = 0
    mixed_words = 0
    rest = 0

    for i, word in enumerate(words):
        if re.match(latin_pattern_with_numbers, word):
            eng_words += 1
            if not re.search(r'<tag\s.*?>.*?</eng>', word) and not switch:
                words[i] = f'<tag {word}>'

        elif re.match(latin_containing_pattern, word):
            #eng_words += 1
            mixed_words += 1
            if not re.search(r'<tag\s.*?>.*?</eng>', word):
                words[i] = f'<tag {word}>'
        else:
            rest += 1
            if switch and not re.search(r'<tag\s.*?>.*?</eng>', word):
                words[i] = f'<tag {word}>'
                

    res = {
        "words": words,
        "eng_words": eng_words,
        "mixed_words": mixed_words,
        "rest": rest,
        }
    return res



def tag_poi_words(text, scd_language, matrix_lang=None, fixedtags=False):
    if "<tag" in text.split(): raise ValueError(f"Your REF file contains tagged words '<tag', you also set scd_language to {scd_language}. Choose one")
    if scd_language == "cmn" or scd_language == "jap":
        text = " ".join(tokenize_for_mer(text))
    words = text.split()
    orig_words = words.copy()

    if matrix_lang == "eng":
        res = tag_words(words, switch=True)
    else:
        res = tag_words(words)

    if res["eng_words"] +res["mixed_words"] == len(words):
        return text

    if not fixedtags:
        if (res["eng_words"]+res["mixed_words"]+res["rest"]) != len(words): 
            print("WTF"); print(ASD)
        elif res["eng_words"] > res["rest"]:
            res = tag_words(orig_words, switch=True)

    tagged_text = ' '.join(res["words"])
    return tagged_text
    

def determine_matrix_language(reference, split_hyphen, scd_language):
    corpus_poi_indices=list()
    corpus_other_indices = list()
    for ref in reference:
        text = tag_poi_words(ref, scd_language, fixedtags=True)
        corrected_text = re.sub(r'<tag\s+([^>]+)\s*>', r'<tag \1>', text)
        pattern = re.compile(r'<tag (.*?)>')
        poi_indices = []
        tags = 0
        if split_hyphen:
        	all_words = text.replace("-", " ").split()
        else:
        	all_words = text.split()
        
        for match in pattern.finditer(text):
            poi_text = match.group(1)
            if split_hyphen:
                start_index = len(text[:match.start()].replace("-", " ").split()) -tags
                words_in_poi = poi_text.replace("-", " ").split()
                end_index = start_index + len(words_in_poi)
            else:
                start_index = len(text[:match.start()].split()) - tags
                end_index = start_index + len(poi_text.split())
            poi_indices.extend(range(start_index, end_index))
            tags += 1
        
        all_indices = list(range(len(all_words)-tags))
        other_indices = [index for index in all_indices if index not in poi_indices]
        corpus_poi_indices.extend(all_indices)
        corpus_other_indices.extend(other_indices)

    if len(corpus_poi_indices) >= len(corpus_other_indices):
        return scd_language
    else:
        return "eng"
    


def add_space_before_punctuation(text):
    # Regular expression to match punctuation that should have a space before it
    pattern = re.compile(r'(?<!\s)([,.!?;:，。！？；：、،؟])')
    # Substitute the matched punctuation with a space before it
    return pattern.sub(r' \1', text)

def extract_indices(text, split_hyphen, scd_language, matrix_lang, fixedtags):
    if scd_language != None: 
        text = tag_poi_words(text, scd_language, matrix_lang="cmn", fixedtags=True)
    else:
        text = add_space_before_punctuation(text)
    #print(text)
    # Correct the incorrect annotation pattern by removing the space before '>'
    corrected_text = re.sub(r'<tag\s+([^>]+)\s*>', r'<tag \1>', text)
    pattern = re.compile(r'<tag (.*?)>')
    poi_indices = []
    tags = 0
    if split_hyphen:
    	all_words = text.replace("-", " ").split()
    else:
    	all_words = text.split()
    
    for match in pattern.finditer(text):
        poi_text = match.group(1)
        if split_hyphen:
            start_index = len(text[:match.start()].replace("-", " ").split()) -tags
            words_in_poi = poi_text.replace("-", " ").split()
            end_index = start_index + len(words_in_poi)
        else:
            start_index = len(text[:match.start()].split()) - tags
            end_index = start_index + len(poi_text.split())
        poi_indices.extend(range(start_index, end_index))
        tags += 1
    
    all_indices = list(range(len(all_words)-tags))
    other_indices = [index for index in all_indices if index not in poi_indices]
    #print(f"p: {poi_indices} o: {other_indices}")
   # if len(poi_indices) > len(other_indices):  
   #     print(text)
       # poi = poi_indices.copy()
       # poi_indices = other_indices
        #other_indices =poi

#    else:
 #      print("CMN")
    return poi_indices, other_indices

def get_idsh(edit_ops, indices, total_len):
    substitutions = sum(1 if op.tag == "replace" and op.src_pos in indices else 0 for op in edit_ops)
    deletions = sum(1 if op.tag == "delete" and op.src_pos in indices else 0 for op in edit_ops)
    insertions = sum(1 if op.tag == "insert" and op.src_pos in indices  else 0 for op in edit_ops)

    s,d,i =0,0,0
    if indices[-1] == total_len-1:
        s = sum(1 if op.tag == "replace" and op.src_pos >= total_len else 0 for op in edit_ops)
        d = sum(1 if op.tag == "delete" and op.src_pos >= total_len else 0 for op in edit_ops)
        i = sum(1 if op.tag == "insert" and op.src_pos >= total_len else 0 for op in edit_ops)
        #if d > 0 or i > 0 or s>0:
        #    print("$$$$$$")
         #   print(indices, total_len-1)
          ##  print(edit_ops)
           # print(Opcodes.from_editops(edit_ops))
           # print(i, d, s)
        # print(DAS)
    #print(insertions, deletions, substitutions)
    substitutions += s
    deletions += d
    insertions += i
    hits = len(indices) - (substitutions + deletions)
    return insertions, deletions, substitutions, hits


