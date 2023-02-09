## Code

The repository contains the work for subtask 1 of Shared Task 3 of CASE@EMNLP 2022. 

The task is to identify causal structure in sentences from news corpus.

In the work, we have considered several large language models and also considered the annotation information present in the dataset to come up with a modified loss for causality detection.

The code is reproducible and the results can be reproduced by running run_script.py.

To obtain the results, run run_script.py using:

```
python3 run_script.py --model_name <name of model> --loss_name <name of loss>
```
## Paper
*Codebase of the paper [NoisyAnnot@ Causal News Corpus 2022: Causality Detection using Multiple Annotation Decision](https://aclanthology.org/2022.case-1.11/)* by  Nguyen Quynh Anh and Mitra Arka at [EMNLP 2022](https://2022.emnlp.org).

To cite the paper, use the BibTex below:

```
@inproceedings{nguyen-mitra-2022-noisyannot,
    title = "{N}oisy{A}nnot@ Causal News Corpus 2022: Causality Detection using Multiple Annotation Decisions",
    author = "Nguyen, Quynh Anh  and
      Mitra, Arka",
    booktitle = "Proceedings of the 5th Workshop on Challenges and Applications of Automated Extraction of Socio-political Events from Text (CASE)",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.case-1.11",
    pages = "79--84",
    abstract = "The paper describes the work that has been submitted to the 5th workshop on Challenges and Applications of Automated Extraction of socio-political events from text (CASE 2022). The work is associated with Subtask 1 of Shared Task 3 that aims to detect causality in protest news corpus. The authors used different large language models with customized cross-entropy loss functions that exploit annotation information. The experiments showed that bert-based-uncased with refined cross-entropy outperformed the others, achieving a F1 score of 0.8501 on the Causal News Corpus dataset.",
}
```
### Contact Us
* Arka Mitra, amitra[at]ethz.ch
* Quynh Anh Nguyen, quynguyen[at]ethz.ch
