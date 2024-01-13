# frl-with-unreliable-label


This repository conatins the implementation of the paper "Fair Representation Learning with Unreliable Labels"


## Abstract

In learning with fairness, for every instance, its label can be systematically flipped to another class due to the practitioner’s prejudice, namely, label bias. The existing well-studied fair representation learning methods focus on removing the dependency between the sensitive factors and the input data, but do not address how the representations retain useful information when the labels are unreliable. In fact, we find that the learned representations become random or degenerated when the instance is contaminated by label bias. To alleviate this issue, we investigate the problem of learning fair representations that are independent of the sensitive factors while retaining the task-relevant information given only access to unreliable labels. Our model disentangles the dependency between fair representations and sensitive factors in the latent space. To remove the reliance between the labels and sensitive factors, we incorporate an additional penalty based on mutual information. The learned purged fair representations can then be used in any downstream processing. We demonstrate the superiority of our method over previous works through multiple experiments on both syn thetic and real-world datasets.



## Reference 
```
@InProceedings{pmlr-v206-zhang23g,
  title = 	 {Fair Representation Learning with Unreliable Labels},
  author =       {Zhang, Yixuan and Zhou, Feng and Li, Zhidong and Wang, Yang and Chen, Fang},
  booktitle = 	 {Proceedings of The 26th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {4655--4667},
  year = 	 {2023},
  editor = 	 {Ruiz, Francisco and Dy, Jennifer and van de Meent, Jan-Willem},
  volume = 	 {206},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {25--27 Apr},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v206/zhang23g/zhang23g.pdf},
  url = 	 {https://proceedings.mlr.press/v206/zhang23g.html},
}
```
