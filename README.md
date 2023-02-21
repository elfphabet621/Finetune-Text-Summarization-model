# Finetune-Text-Summarization-model

A bilingual model for summarizing English and Vietnamese document

Text summarization is one of the most challenging NLP tasks. It requires the model to understand long passages and generate text that captures the main ideas in a document.

To think about it, text summarization is sort of similar to machine translation: we translate from a long passage to a shorter version that captures the salient features of the input.

Many popular and powerful pre-trained models from the *HuggingFace* that can be fine-tuned using our own data

I chose T5 - an architecture pre-trained in a text-to-text framework. Then I fine-tuned on the task of summarization on a small sample of data. Doing this allows me to debug and work fast toward on the project, also due to the constrain of time and hardware, I train on a few epochs.

I chose T5 - an architecture pre-trained in a text-to-text framework. Then I fine-tuned on the task of summarization on a small sample of data. Doing this allows me to debug and work fast toward on the project, also due to the constrain of time and hardware, I train on a few epochs.

## 1. Preparing a Bilingual Corpus 
[wiki_lingua](https://huggingface.co/datasets/wiki_lingua) This corpus consists of articles and corresponding summaries from WikiHow. The number of article-summary pairs for English
are 141,457 and 19,600 for Vietnamese

Only consider <code>document</code> and <code>summary</code> columns

Take a portion of data and split them into 2 sets: train + test
```python
vietnamese_train_ds, vietnamese_test_ds = load_dataset("wiki_lingua", "vietnamese", split=['train[:1000]', 'train[2000:2500]'])
english_train_ds, english_test_ds = load_dataset("wiki_lingua", "english", split=['train[:1000]', 'train[2000:2500]'])
```

Concat the English and Spanish into a single Dataset to create an actual bilingual dataset
```python
def concat_two_datasets(dataset1, dataset2):
  dataset = concatenate_datasets([dataset1, dataset2])
  dataset = dataset.shuffle(seed=42)
  return dataset
```

Since the max sequence length of our model is 512, I filter out very long and very short article
```python
dataset = dataset.filter(lambda x: get_len(x['document']) > 20 and get_len(x['document']) < 512)
```

## 2.Preprocessing Data
Tokenize and Encode our data pairs.

Using mt5 tokenizer (SentencePiece tokenizer)as my checkpoint. 
```python
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

Set the upper limit and truncate anything that passes the max length.
```python
def preprocess_function(examples):
    model_inputs = tokenizer(examples["document"],
        max_length=max_input_length,
        truncation=True)
    
    labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```

## 3. Metric for text summarization
Commonly used metric: ROUGE score

$ROUGE_{recall} = \frac{\textrm{num of overlapping word}}{\textrm{total words in reference}}$

$ROUGE_{precision} = \frac{\textrm{num of overlapping word}}{\textrm{total words in summary}}$

$ROUGE_{F1 score} = 2\frac{\textrm{precision}.\textrm{recall}}{\textrm{precision}+\textrm{recall}}$

- The rouge1 is the overlap of unigrams
- The rouge2 measures the overlap between bigrams 
- rougeL and rougeLsum measure the longest matching sequences of words by looking for the longest common substrings
- The rougeLsum is computed over a whole summary, while rougeL is computed as the average over individual sentences.

## 4. Baseline
Using the sentence tokenizer from <code>nltk</code> to extract the first three sentences in an article is often considered a strong baseline

```python
def three_sentence_summary(text):
    return "\n".join(sent_tokenize(text)[:3])

def evaluate_baseline(dataset, metric):
    summaries = [three_sentence_summary(text) for text in dataset["document"]]
    return metric.compute(predictions=summaries, references=dataset["summary"])
 ```
 
## 5. Fine-tuning with Accelerate
- Iterating over epoch, train the model using mini-batch generated from train_dataloader
- At the end of each epoch, compute the ROUGE score (generating the tokens -> decoding them (and the reference summaries) into text -> compute score)

## Conclusion
Although the result was not so good, remember that this was trained only a tiny amount of data and only 3 epochs. What I have achieved is
a workflow for fine-tuning a text summarization model and understanding what it's like to deal with a multilingual corpus.
