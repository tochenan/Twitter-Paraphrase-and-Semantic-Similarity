# Grasp Hometask Mingran

## Preprocessing
Before proceeding with the training process, I first examined and preprocessed the training dataset. I removed the data points where labels were not present and identified that there were twice as many false labels as true labels, which requires me to balance the class weights during training.

## Exercise1
Large language models like BERT and GPT have millions of parameters and possess an intrinsic representation of language structure. Thus, I decided to utilize transfer learning and use a pre-trained large language model like BERT to solve this task. This approach in the past has been illustrated to achieve state-of-the-art performance in many popular NLP benchmarks

I first tokenize my training dataset, split it into training and validation dataset and store it in a format understandable by PyTorch. Then, I use the training dataset to fine-tune BERT. During this fine-tuning process, I also conduct a hyperparameter search to optimize hyperparameters such as learning rate, batch size, and the number of training epochs. After obtaining the best-performing model, I use it to generate predictions for the test dataset.

model link: https://drive.google.com/drive/folders/1W9iJPkYgovqZT76XJtRm5Z6QYgm5EqOQ?usp=sharing
code: scripts/BERT_Finetune.ipynb (use in google colab)


## Exercise 2
### (A) & Exercise 3
As time is scarce for this task, instead of fine-tuning a large model and trying out some advanced architectures, I used a linear SVM with regularization on extracted language-relevant features from the sentence pairs. The extracted language-relevant features are:

1. Cosine Similarity using TF-IDF: TF-IDF represents the importance of words by adjusting their count in the sentence by the inverse of their frequency in the corpus. This gives more weight to rarer words in the sentence. Computing similarity using TF-IDF reflects word-level similarities between the two sentences.

2. Jaccard Similarity: Jaccard similarity measures sentence similarity by calculating the size of the intersection divided by the size of the union.

3. Sentence Length Difference: The absolute difference in the number of words between the two sentences.

4. Number of Common Words.

5. POS Tags Similarity: This evaluates the grammatical structure similarity of the sentence pairs. Paraphrased sentences are more likely to have similar grammatical structures.

model link: model/best_svc_model_C_0.01.pkl
code: scripts/feature_engineering_svm.py

## Evaluation statistics

Model |     Code|         F |      P |        R | P Correlation | MaxF1 | P MaxF1 | R MaxF1
---------|-----------|--------- |---------|---------|-------------------|-----------|--------------|------------
BERT  | 03BCL | 0.602  | 0.763| 0.497 |             0.575 |   0.688 |       0.702 |   0.674
SVC    | 04        | 0.646  | 0.810 | 0.537|             0.569 |   0.653 |       0.807 |   0.549

Overall, both models show comparable prediction performance in terms of F1 score, precision, and recall, with the SVC slightly outperforming the fine-tuned BERT model on these metrics. However, with an optimal threshold, the BERT model performs better than the SVC, with slightly higher precision.

For both models, recall is lower than precision, which indicates that the models have a high number of false negatives. To improve the precision-recall balance, a lower decision threshold can be implemented to define what is positive. This issue is likely due to the fact that there are fewer examples of the positive class in the training dataset, which may not fully represent the feature statistics of the true positive class, leading to lower recall.


### (B)
Given enough time and resources (and perhaps more MLE resources), there are several improvements that can be made to the models in Exercise 1:

1. Use domain-specific stop words and custom tokenization, such as using trend names as stop words.

2. Assess the distribution of semantic representations in the training dataset and include more data to complement the distribution if the training dataset is biased. This could improve the generalizability of the model.

3. Use more continuous measures of judgment to preserve nuanced differences between sentence pairs. Although some mid labels have been removed from the dataset, a (3,2) and a (4,1) have clear differences in the confidence of the decision and should be modulated by the model as well. This can be achieved by normalizing the judgment vote. For example, if the votes are (3, 2), the similarity score would be 3/(3+2) . The model can be modified to perform a regression task and output a continuous similarity score

4.  Use GPUs or TPUs for faster training and implement distributed training strategies to handle larger datasets.

5. Explore and compare BERT model fine-tuned using different regularizer and training methods. 

**i. Adopt contrastive learning in the fine-tuning process **

e.g. use a contrastive loss function.  This can be incorporated using a siamese network that  generate embeddings for pairs of inputs (in this case the sentence pair). By using a contrastive loss function, the network can learn to produce similar embeddings for similar inputs and dissimilar embeddings for dissimilar inputs which would enhance the contrast of positive labelled class and negative labeled class.

**ii. Utilize better regularization method **

Fine-tuning large models on limited datasets can lead to overfitting, causing the model to learn more about noise rather than the underlying patterns of the data. One regularization method that can be implemented is described in this paper: https://paperswithcode.com/paper/smart-robust-and-efficient-fine-tuning-for. which has performed exceptionally well on  Paraphrase Identification on Quora Question Pairs.

In this paper, they use smoothness-induced regularization to ensure that the model parameters do not change too drastically during fine-tuning, enabling a smoother learning process. At the same time, they employ the Bregman proximal point optimization, which restricts the optimization step to a region where the model can perform well, thus effectively retaining pre-trained knowledge and avoiding forgetting.