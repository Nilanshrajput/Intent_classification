# Intent_classification

## 1.Dataset

The data contains more than 2000 user queries that have been generated for each intent with crowdsourcing methods.

The Dataset is seggregated into train.csv, valid.csv and test.csv and available in the Dataset folder.

The dataset is categorized into seven intents such as:

1.  **AddToPlaylist**

2. **BookRestraunt**
 
3. **GetWeather**
 
4. **PlayMusic**
 
5. **RateBook**
 
6. **SearchCreativeWork**
 
7. **SearchScreeningEvent**

### The model uses **Mlflow** for experiment tracking, and **Pytorch Lightning** and **hugging-face Transformers** Library

#### Link to the Github-repo: https://github.com/snipsco/nlu-benchmark/tree/master/2017-06-custom-intent-engines

#### Link to the paper: https://arxiv.org/abs/1805.10190


 
## Model:

BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

**Link to the BERT paper:** https://arxiv.org/abs/1810.04805


A blog explaining about transformers and evolution of BERT: https://jalammar.github.io/illustrated-bert/ 
