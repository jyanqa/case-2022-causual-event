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
*Codebase of the paper [NoisyAnnot@ Causal News Corpus 2022: Causality Detection using Multiple Annotation Decision](https://arxiv.org/abs/2210.14852)* by  Nguyen Quynh Anh and Mitra Arka at [EMNLP 2022](https://2022.emnlp.org).

To cite the paper, use the BibTex below:

```
@misc{https://doi.org/10.48550/arxiv.2210.14852,
  doi = {10.48550/ARXIV.2210.14852},
  
  url = {https://arxiv.org/abs/2210.14852},
  
  author = {Nguyen, Quynh Anh and Mitra, Arka},
  
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Causality Detection using Multiple Annotation Decision},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}

}
```
### Contact Us
* Arka Mitra, amitra[at]ethz.ch
* Quynh Anh Nguyen, quynguyen[at]ethz.ch
