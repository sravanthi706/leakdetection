### MSRP paraphrase benchmark (test split, 1,725 pairs)

| Representation | ROC-AUC | Best-threshold accuracy |
|---|---|---|
| tfidf_char_3_5 | 0.755 | 0.722 |
| tfidf_word_1_2 | 0.715 | 0.707 |
| wordllama | 0.724 | 0.702 |
| majority class | – | 0.665 |

### Leak-message detection vs the committed JEE 2023 papers

| Leak form | n | Lexical AUC | Semantic AUC | Answer-key AUC | **Combined AUC** | Combined recall @1% FPR | Right question named |
|---|---|---|---|---|---|---|---|
| answer_key | 60 | 0.485 | 0.482 | 1.000 | **1.000** | 1.000 | – |
| excerpt | 65 | 0.973 | 0.748 | 0.500 | **0.968** | 0.769 | 0.97 |
| hinglish (hand-written) | 40 | 0.839 | 0.606 | 0.500 | **0.853** | 0.300 | 1.00 |
| ocr | 65 | 1.000 | 0.991 | 0.500 | **1.000** | 1.000 | 0.98 |
| paraphrase (hand-written) | 40 | 0.944 | 0.850 | 0.500 | **0.976** | 0.700 | 1.00 |
| typed | 65 | 0.991 | 0.867 | 0.500 | **0.990** | 0.969 | 0.97 |
| verbatim | 65 | 1.000 | 1.000 | 0.500 | **1.000** | 1.000 | 1.00 |
| ALL leak forms | 400 | 0.895 | 0.804 | 0.575 | **0.976** | 0.858 | 0.99 |

False-positive rate (combined score > 0.5) by type of legitimate message:

- past paper: verbatim: 0.004
- past paper: excerpt: 0.000
- past paper: ocr: 0.000
- past paper: typed: 0.000
- past paper: paraphrase: 0.004
- past paper: answer_key: 0.000
- chat / ads / scams: 0.000
