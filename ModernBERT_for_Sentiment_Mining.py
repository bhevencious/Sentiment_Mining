# -*- coding: utf-8 -*-
import torch
from transformers.utils import is_flash_attn_2_available
print("\n", "Torch:", torch.__version__)
print("CUDA (compiled):", torch.version.cuda)
print("Flash-Attention 2 available:", is_flash_attn_2_available())

import numpy as np
fname = "Amazon-Reviews-2023"
tformer_name = "answerdotai/ModernBERT-base"
#tformer_name = "chandar-lab/NeoBERT"
#tformer_name = "bert-base-uncased"  #BERT (Encoder) Transformer model trained on text where casing distinction has been removed. "hello"=="Hello"=="HELLO". 
category_list = ["raw_review_All_Beauty", "raw_review_Appliances", "raw_review_Amazon_Fashion", "raw_review_Musical_Instruments", "raw_review_Video_Games", "raw_review_CDs_and_Vinyl", "raw_review_Software", "raw_review_Baby_Products", "raw_review_Toys_and_Games", "raw_review_Automotive"]

train_log_file = open("./results/"+fname+"_train_metrics_log.txt", "a")
print("-------------- " +fname+ " dataset --------------", file=train_log_file)
print(" * , TRAIN_RUNTIME, TRAIN_SAMPLES_PER_SECOND, TRAIN_STEPS_PER_SECOND", file=train_log_file)
train_log_file.close()

test_log_file = open("./results/"+fname+"_test_metrics_log.txt", "a")
print("-------------- " +fname+ " dataset --------------", file=test_log_file)
print(" * , BAL_ACCURACY, PRECISION, RECALL, F1-SCORE, ROC_AUC, MCC", file=test_log_file)
test_log_file.close()

label_stats_file = open("./results/"+fname+"_class_label_stats.txt", "a")
print("-------------- " +fname+ " dataset --------------", file=label_stats_file)
print(" * , 0(VERY_NEG), 1(NEGATIVE), 2(NEUTRAL), 3(POSITIVE), 4(VERY_POS)", file=label_stats_file)
label_stats_file.close()

for category in category_list:
    seed = 42
    train_log_file = open("./results/"+fname+"_train_metrics_log.txt", "a")
    test_log_file = open("./results/"+fname+"_test_metrics_log.txt", "a")
    label_stats_file = open("./results/"+fname+"_class_label_stats.txt", "a")
    
    #load_dataset(): returns a dictionary = DatasetDict({train: Dataset,  test: Dataset})
    from datasets import load_dataset, concatenate_datasets
    dataset = load_dataset(path="McAuley-Lab/"+fname, name=category, split='full', trust_remote_code=True) #split='train|valid|test|full'
    print("\n", dataset) #return cols/headers/feats & rows/samples present in dataset
    print("\n", "This is the 1st row in this dataset:\n", dataset[0])
    
    #During training, the Transformer automatically infers 'features' and 'targets' from keys in the dataset. The inference is usually done by: LLM.forward() method.
    #The keys(input_ids,token_type_ids,attention_mask) denote the 'features', and they are generated after TOKENIZATION on the sample['text'] field/col/key wrt. dataset.
    #Also, key(labels) denotes the 'targets/classes' for classification tasks, and is computed herein based on the sample['rating'] score.
    
    # ADD/CREATE 'Fine-Grained Sentiment' target_label to each sample in dataset
    def add_target_label(sample):
        sample['rating'] = float(sample['rating'])
        if (sample['rating'] >= 1.0) and (sample['rating'] < 1.5):
            sample['labels'] = 0 #'Very -ve' review
        elif (sample['rating'] >= 1.5) and (sample['rating'] < 2.5):
            sample['labels'] = 1 #'-ve' review
        elif (sample['rating'] >= 2.5) and (sample['rating'] < 3.5):
            sample['labels'] = 2 #'Neutral' review
        elif (sample['rating'] >= 3.5) and (sample['rating'] < 4.5):
            sample['labels'] = 3 #'+ve' review
        elif (sample['rating'] >= 4.5) and (sample['rating'] <= 5.0):
            sample['labels'] = 4 #'Very +ve' review
        else:
            sample['labels'] = 2 #'Neutral' review
        return sample
    #map(): apply a 'preprocessing' func. to each example/sample in a dataset, independently or in batches.
    dataset = dataset.map(add_target_label, batched=False)
    print("\n", dataset) #return headers/cols present in dataset
    
    # 'Fine-Grained Sentiment' Classes/Labels statistics
    class_very_neg = dataset['labels'].count(0)
    class_neg = dataset['labels'].count(1)
    class_neutral = dataset['labels'].count(2)
    class_pos = dataset['labels'].count(3)
    class_very_pos = dataset['labels'].count(4)
    class_labels = {"very_neg(0)":class_very_neg, "neg(1)":class_neg, "neutral(2)":class_neutral, "pos(3)":class_pos, "very_pos(4)":class_very_pos,}
    print(category+":,", class_very_neg,",", class_neg,",", class_neutral,",", class_pos,",", class_very_pos, file=label_stats_file)
    print("\n", "CLASS/LABEL (Fine-Grained Sentiments) Report:\n", class_labels)
    
    # Data Resampling based on the "Minority Class/Label"
    # minClass = min([class_very_neg, class_neg, class_neutral, class_pos, class_very_pos])
    # c0 = dataset.filter(lambda sample: sample["labels"] in {0}).shuffle(seed=seed).select(range(minClass))
    # c1 = dataset.filter(lambda sample: sample["labels"] in {1}).shuffle(seed=seed).select(range(minClass))
    # c2 = dataset.filter(lambda sample: sample["labels"] in {2}).shuffle(seed=seed).select(range(minClass))
    # c3 = dataset.filter(lambda sample: sample["labels"] in {3}).shuffle(seed=seed).select(range(minClass))
    # c4 = dataset.filter(lambda sample: sample["labels"] in {4}).shuffle(seed=seed).select(range(minClass))
    # resmpl_data = concatenate_datasets([c0, c1, c2, c3, c4])
    # class_labels = {"very_neg(0)":resmpl_data['labels'].count(0), "neg(1)":resmpl_data['labels'].count(1), "neutral(2)":resmpl_data['labels'].count(2), "pos(3)":resmpl_data['labels'].count(3), "very_pos(4)":resmpl_data['labels'].count(4),}
    # print(category+":,", resmpl_data['labels'].count(0),",", resmpl_data['labels'].count(1),",", resmpl_data['labels'].count(2),",", resmpl_data['labels'].count(3),",", resmpl_data['labels'].count(4), file=label_stats_file)
    # print("\n", "'RESAMPLED' CLASS/LABEL (Fine-Grained Sentiments) Report:\n", class_labels)
    
    # Load the 'tokenizer' wrt. a pretrained LLM model
    from transformers import AutoTokenizer, DataCollatorWithPadding
    llm_tokenizer = AutoTokenizer.from_pretrained(tformer_name, use_fast=True)    
    def preprocess_func(sample):  #Tokenization: Truncation to 512 tokens.
        return llm_tokenizer(sample['text'], truncation=True, max_length=512).to("cuda")
    #map(): apply a 'preprocessing' func. to each example/sample in a dataset, independently or in batches.
    tokenized_dataset = dataset.map(preprocess_func, batched=True) #tokenized_dataset = resmpl_data.map(preprocess_func, batched=True) 
    # Dynamic Padding at Training Time: To ensure all inputs are of the same length
    collator = DataCollatorWithPadding(llm_tokenizer, pad_to_multiple_of=8)
    
    # ChecK/Scan for NaN values in dataset
    #for i, sample in enumerate(tokenized_dataset):  #enumerate(data): returns indexes for 'i' wrt. dataset 
    #    for key in ["input_ids", "token_type_ids", "attention_mask", "labels"]:
    #        if (key in sample) and (np.isnan(np.array(sample[key])).any()):
    #            print("NaN detected at sample: ", i, " in column/field '", key, "'")
    #            break
    
    # Split dataset into TRAIN and TEST sets
    if "test" not in tokenized_dataset:  #load_dataset() returns a dictionary: DatasetDict({train:Dataset, test:Dataset})
        split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=seed)
        train_dataset = split_dataset["train"]
        test_dataset = split_dataset["test"]
    else:
        train_dataset = tokenized_dataset["train"]
        test_dataset = tokenized_dataset["test"]
    print("\n", "Train-Set sample[0]:\n", train_dataset[0])
    print("\n", "Test-Set sample[0]:\n", test_dataset[0])
    
    # Load actual 'pretrained LLM model' for sequence classification
    from transformers import AutoModelForSequenceClassification
    llm_model = AutoModelForSequenceClassification.from_pretrained(tformer_name,
                                                                   num_labels=5,  #LLM model will have 5 output labels: 0, 1, 2, 3, 4 AND so, 'num_labels=5'
                                                                   #attn_implementation="flash_attention_2",  #init. FLASH-ATTENTION on GPU
                                                                   torch_dtype=torch.bfloat16).to("cuda")
    
    # Count of 'Trainable' and 'Non-Trainable' parameters
    trainable_params = sum(p.numel() for p in llm_model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in llm_model.parameters() if not p.requires_grad)    
    print("\n", "Trainable Parameters:", trainable_params)
    print("Non-trainable Parameters:", non_trainable_params)
    print("Total Parameters:", trainable_params + non_trainable_params)

    # Instantiate & Initialize a custom 'optimizer'
    #from lomo_optim import AdaLomo
    #ada_lomo = AdaLomo(llm_model, lr=2e-5, weight_decay=0.01)  #lr=1e-3 AND weight_decay=0.01(enables L2 regularization)
    
    # Compute total number of Training Steps or Iterations per dataset
    batch_size = 32  #48
    num_epochs = 2  #4
    num_train_steps = (len(train_dataset) / batch_size) * num_epochs
    #If using "gradient accumulation":  num_training_steps = (len(train_dataset) / (batch_size * gradient_accumulation_steps)) * num_epochs
    
    # Instantiate & Initialize Trainer() method: 'HuggingFace' Trainer() simplifies training-loop by handling gradient updates, evaluation, and logging. We only need to pass the model, training arguments, dataset, and tokenizer.
    from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
    trainer = Trainer(
        tokenizer=llm_tokenizer,
        data_collator=collator,
        model=llm_model,
        args=TrainingArguments(
            output_dir="./results",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,  #8
            per_device_eval_batch_size=batch_size,  #8
            max_grad_norm=1.0,  #Disabled as recommendation for 'AdaLomo|Lomo' optimizer. For other optimizers, ACTIVATE it: 'max_grad_norm=1.0'.
            eval_strategy="epoch",  #no|steps|epoch  "no(NO eval. in training)|steps(Eval. per step/iteration)|epoch(Eval. at end of each epoch)"
            logging_strategy="epoch",  #no|steps|epoch  NB: step==iteration
            save_strategy="epoch",  #no|steps|epoch|best  NB: step==iteration; best==save done when a new 'best_metric' is achieved.
            bf16=True,  #'googleBrain-floatPoint16': False = disable mixed-precision training (only floatPoint32 used - fp16 & bf16 CAN'T be used together as 'True').
            fp16=False,  #'floatPoint16': False = disable mixed-precision training (only floatPoint32 used).
            metric_for_best_model="eval_loss",  #Palliate_Overfitting: enable Early-Stopping via or "eval_loss|eval_f1|eval_accuracy"
            greater_is_better=False,  #Palliate_Overfitting:: works with 'metric_for_best_model' above - will be 'True' if monitoring F1|Accuracy.
            load_best_model_at_end=True,  #Palliate_Overfitting:: works with 'metric_for_best_model' and 'greater_is_better' above.
            label_smoothing_factor=0.02,  #Palliate_Overfitting:: works like [1, 0, 1] --> [0.98, 0.02, 0.98]. Notice subtraction & addition of "label_smoothing_factor" value.
            group_by_length=True,  #Bucket samples by length & batch similar-length samples together.
            optim="adamw_torch_fused",
            adam_epsilon=1e-6,  #Small, positive constant added to "Adam optimizer" to prevent division-by-zero & ensure numerical stability
            learning_rate=2.5e-5,  #Palliate_Overfitting:: use of smaller Learning-Rate
            weight_decay=0.015,  #Palliate_Overfitting:: discourages large weights via enabling L2-regularization         
            # Warmup-Stable-Decay (WSD)
            lr_scheduler_type="warmup_stable_decay",  #Palliate_Overfitting:: stabilizes early training & avoids overfitting from aggressive updates. "linear|cosine|constant|etc."
            warmup_steps=int(0.06 * num_train_steps),
            lr_scheduler_kwargs={
                "warmup_type": "linear",  #Warm-up type (linear, cosine, etc.)
                "num_decay_steps": int(0.14 * num_train_steps),  #Steps for the decay phase
                "decay_type": "cosine",  #Decay shape
                "min_lr_ratio": 0.0,  #Final learning rate ratio, if needed
                #"num_stable_steps": int(0.80 * num_train_steps),  #Number of steps to hold learning rate. Commented to avoid conflict with auto-set "num_training_steps".
            }
        ),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        #optimizers=(ada_lomo, None),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],  #Palliate_Overfitting:: enable Early-Stopping via or "eval_loss|eval_f1|eval_accuracy"        
    )
    
    # TRAINING the LLM model. The train() method will handle the training process, logging the results at each epoch.
    train_res = trainer.train()
    temp = train_res.metrics
    train_rtime = temp['train_runtime']
    train_samps_per_sec = temp['train_samples_per_second']
    train_itr_per_sec = temp['train_steps_per_second']
    
    train_metrics = {"train_runtime":train_rtime, "train_samples_per_second":train_samps_per_sec, "train_steps_per_second":train_itr_per_sec,}
    print(category+":,", train_rtime,",", train_samps_per_sec,",", train_itr_per_sec, file=train_log_file)
    print("\n", "TRAINING Report:\n", train_metrics)
    
    # EVALUATE the LLM model. We evaluate the model on the "test" dataset to check how well it generalizes to new, unseen data.
    from scipy.special import softmax
    from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
    from sklearn.preprocessing import LabelBinarizer
    test_res = trainer.predict(test_dataset)  #"trainer.predict()" auto-calls "model.eval()" to turn off 'Dropout' & 'Batch-Normalization' updates
    gTruth = test_res.label_ids
    logits = test_res.predictions
    gTruth_1hot = LabelBinarizer().fit_transform(gTruth)  #LabelBinarizer().classes_ (Returns the 'classes/labels' present in the dataset)
    preds = np.argmax(logits, axis=1)  #Returns the MAX value per col/feat(axis=1) wrt. each row/sample.
    probs = softmax(logits, axis=1)  #Returns the SOFTMAX value per col/feat(axis=1) wrt. each row/sample. I.E. sum of each row = 1.
    print("gTruth:\n", gTruth)
    print("logits:\n", logits)
    print("gTruth_1hot:\n", gTruth_1hot)
    print("preds:\n", preds)
    print("probs:\n", probs)
        
    #'macro': compute metrics for each class/label, and find their unweighted mean. Doesn't take label-imbalance into account.
    #'weighted': compute metrics for each class/label, and find their average weighted by 'support'(count of true instances per class/label).
    #'bal_acc|precision|recall|f1-score|auc|mcc' here ALL take into account class/label-imbalance
    bal_acc = round(float(balanced_accuracy_score(gTruth, preds)), 3)
    precsn = round(float(precision_score(gTruth, preds, average="weighted")), 3)
    recll = round(float(recall_score(gTruth, preds, average="weighted")), 3)
    f1 = round(float(f1_score(gTruth, preds, average='weighted')), 3)
    auc = round(float(roc_auc_score(gTruth_1hot, probs, multi_class="ovr", average="weighted")), 3)
    mcc = round(float(matthews_corrcoef(gTruth, preds)), 3)
    
    test_metrics = {"bal_accuracy":bal_acc, "precision":precsn, "recall":recll, "f1-score":f1, "auc":auc, "mcc":mcc,}
    print(category+":,", bal_acc,",", precsn,",", recll,",", f1,",", auc,",", mcc, file=test_log_file)
    print("\n", "TEST Report:\n", test_metrics)
    
    # COMPUTE Learing Curve wrt. 'cost/loss' function: Extract training and evaluation losses
    import matplotlib.pyplot as plt
    logs = trainer.state.log_history
    train_loss = [log["loss"] for log in logs if "loss" in log and "eval_loss" not in log]
    train_epoch = [log["epoch"] for log in logs if "loss" in log and "eval_loss" not in log]
    test_loss = [log["eval_loss"] for log in logs if "eval_loss" in log]
    test_epoch = [log["epoch"] for log in logs if "eval_loss" in log]
    
    # VISUALIZATIONS
    # Learning Curve: Plot performance of architecture(LLM) wrt. its training & testing
    plt.figure(figsize=(15,6))  # Using "plt.figure()" to tweak the resultant graph plot
    
    plt.subplot(1, 2, 1)  # subplot(rows, cols, active_ROW-COL_in_subplotSpace)
    plt.grid()
    plt.plot(train_epoch, train_loss, "k-", marker='o', label='Training(loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Cost/Loss() function')
    plt.legend(loc="best")
    plt.title(category)
    plt.subplot(1, 2, 2)  # subplot(rows, cols, active_ROW-COL_in_subplotSpace)
    plt.grid()
    plt.plot(test_epoch, test_loss, "b--", marker='o', label='Validation(loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Cost/Loss() function')
    plt.legend(loc="best")
    plt.title(category)
    
    plt.savefig("./results/" + fname+"-"+category + "_learning_curve.png")
    plt.show()
    
    # Confusion Matrix: Plot wrt. architecture(LLM) predictions during testing
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(gTruth, preds)
    classes = ["Very -ve", "-ve", "Neutral", "+ve", "Very +ve"]    
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Grays", cbar=True, linewidths=0.7, linecolor="black", square=True)
    ax.set_xlabel("Predicted (Polarity) Labels");
    ax.set_ylabel("Groundtruth (Polarity) Labels");
    
    plt.tight_layout();
    plt.savefig("./results/" + fname+"-"+category + "_confusion_matrix.png")
    plt.show()    
    # VISUALIZATIONS

    # Close log files
    train_log_file.close()
    test_log_file.close()
    label_stats_file.close()
    
    # Free-up GPU memory for next dataset
    import gc
    del trainer  #delinks Trainer() object
    #del llm_tokenizer  #delinks AutoTokenizer() object
    del llm_model  #delinks AutoModelForSequenceClassification() object
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  #Return cached blocks to the driver
        torch.cuda.ipc_collect()  #Free any IPC handles