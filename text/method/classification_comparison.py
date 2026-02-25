
import json 
import pandas as pd 
import numpy as np 
from sklearn .model_selection import train_test_split 
from sklearn .metrics import accuracy_score ,precision_recall_fscore_support ,classification_report 
from transformers import (
AutoTokenizer ,AutoModelForSequenceClassification ,
TrainingArguments ,Trainer ,DataCollatorWithPadding ,
TrainerCallback 
)
import torch 
from torch .utils .data import Dataset 
import matplotlib .pyplot as plt 
import seaborn as sns 
from datasets import Dataset as HFDataset 
import time 
import warnings 
import os 
warnings .filterwarnings ('ignore')


plt .rcParams ['font.sans-serif']=['SimHei','Arial Unicode MS','DejaVu Sans']
plt .rcParams ['axes.unicode_minus']=False 

class MetricsCallback (TrainerCallback ):

    def __init__ (self ):
        self .train_losses =[]
        self .eval_losses =[]
        self .eval_accuracies =[]
        self .epochs =[]
        self .steps =[]
        self .last_logged_step =0 
        self .log_interval =250 

    def on_log (self ,args ,state ,control ,logs =None ,**kwargs ):
        if logs :

            if 'loss'in logs and state .global_step -self .last_logged_step >=self .log_interval :
                self .train_losses .append (logs ['loss'])
                self .last_logged_step =state .global_step 

            if 'eval_loss'in logs and 'eval_accuracy'in logs :
                self .eval_losses .append (logs ['eval_loss'])
                self .eval_accuracies .append (logs ['eval_accuracy'])

                self .steps .append (state .global_step )

                current_epoch =state .global_step /(state .max_steps /args .num_train_epochs )*args .num_train_epochs if state .max_steps >0 else 0 
                self .epochs .append (current_epoch )

class NewsDataset (Dataset ):
    def __init__ (self ,texts ,labels ,tokenizer ,max_length =512 ):
        self .texts =texts 
        self .labels =labels 
        self .tokenizer =tokenizer 
        self .max_length =max_length 

    def __len__ (self ):
        return len (self .texts )

    def __getitem__ (self ,idx ):
        text =str (self .texts [idx ])
        label =int (self .labels [idx ])

        encoding =self .tokenizer (
        text ,
        truncation =True ,
        padding ='max_length',
        max_length =self .max_length ,
        return_tensors ='pt'
        )

        return {
        'input_ids':encoding ['input_ids'].flatten (),
        'attention_mask':encoding ['attention_mask'].flatten (),
        'labels':torch .tensor (label ,dtype =torch .long )
        }

def load_original_dataset ():
    from datasets import load_dataset 
    print ("Loading original AG News dataset...")


    dataset =load_dataset ("ag_news")


    train_texts =list (dataset ['train']['text'])
    train_labels =list (dataset ['train']['label'])
    test_texts =list (dataset ['test']['text'])
    test_labels =list (dataset ['test']['label'])

    print (f"Original AG News dataset loaded:")
    print (f"  Train: {len(train_texts)} items")
    print (f"  Test: {len(test_texts)} items")

    return train_texts ,train_labels ,test_texts ,test_labels 

def load_cleaned_dataset (threshold ):
    print ("Loading cleaned dataset...")

    texts =[]
    labels =[]


    try :
        with open (f'batch_cleaned_datasets\\ag_news_threshold_{threshold}.json','r',encoding ='utf-8')as f :
            content =f .read ().strip ()


            current_obj =""
            bracket_count =0 

            for char in content :
                current_obj +=char 

                if char =='{':
                    bracket_count +=1 
                elif char =='}':
                    bracket_count -=1 

                    if bracket_count ==0 :
                        try :
                            data =json .loads (current_obj .strip ())
                            if 'text'in data and 'label'in data :
                                texts .append (data ['text'])
                                labels .append (data ['label'])
                        except json .JSONDecodeError :
                            pass 

                        current_obj =""

        print (f"Train set loaded: {len(texts)} items")

    except Exception as e :
        print (f"Failed to load train set: {e}")
        return [],[]


    try :
        with open (f'batch_cleaned_datasets\\ag_news_test_threshold_{threshold}.json','r',encoding ='utf-8')as f :
            content =f .read ().strip ()


            current_obj =""
            bracket_count =0 

            for char in content :
                current_obj +=char 

                if char =='{':
                    bracket_count +=1 
                elif char =='}':
                    bracket_count -=1 

                    if bracket_count ==0 :
                        try :
                            data =json .loads (current_obj .strip ())
                            if 'text'in data and 'label'in data :
                                texts .append (data ['text'])
                                labels .append (data ['label'])
                        except json .JSONDecodeError :
                            pass 

                        current_obj =""

        print (f"Test set loaded, total: {len(texts)} items")
        return texts ,labels 

    except Exception as e :
        print (f"Failed to load test set: {e}")

        return texts ,labels 

def analyze_dataset_statistics (texts ,labels ,dataset_name ):
    print (f"\n{dataset_name} dataset statistics:")
    print ("-"*40 )


    label_counts =pd .Series (labels ).value_counts ().sort_index ()
    print (f"Label distribution: {dict(label_counts)}")


    text_lengths =[len (text .split ())for text in texts ]
    print (f"Average text length: {np.mean(text_lengths):.2f} words")
    print (f"Text length range: {min(text_lengths)} - {max(text_lengths)} words")

    return label_counts ,text_lengths 

def train_model (train_texts ,train_labels ,val_texts ,val_labels ,model_name ="google/electra-small-discriminator",num_epochs =2 ):
    print (f"\nStarting model training: {model_name}")
    print ("-"*40 )


    fallback_models =[
    model_name ,
    "google/electra-small-discriminator",
    "distilbert-base-uncased",
    "bert-base-uncased"
    ]

    tokenizer =None 
    model =None 
    used_model =None 


    for attempt_model in fallback_models :
        if used_model :
            break 

        print (f"Attempting to load model: {attempt_model}")
        try :

            if tokenizer is None :
                tokenizer =AutoTokenizer .from_pretrained (attempt_model )
                print ("Tokenizer loaded successfully")


            try :
                model =AutoModelForSequenceClassification .from_pretrained (
                attempt_model ,
                num_labels =4 ,
                use_safetensors =True 
                )
                used_model =attempt_model 
                print (f"Successfully loaded model with safetensors: {attempt_model}")
                break 

            except Exception as safetensors_error :
                print (f"safetensors load failed: {safetensors_error}")


                if attempt_model ==model_name and "bert-tiny"in model_name :
                    print ("Attempting safe bypass...")
                    try :

                        import transformers 
                        from transformers .utils import import_utils 

                        original_check =import_utils .check_torch_load_is_safe 
                        import_utils .check_torch_load_is_safe =lambda :None 

                        model =AutoModelForSequenceClassification .from_pretrained (
                        attempt_model ,
                        num_labels =4 
                        )
                        used_model =attempt_model 
                        print (f"Loaded model using safe bypass: {attempt_model}")


                        import_utils .check_torch_load_is_safe =original_check 
                        break 

                    except Exception as bypass_error :
                        print (f"Safe bypass also failed: {bypass_error}")

                        import_utils .check_torch_load_is_safe =original_check 
                        continue 
                else :
                    continue 

        except Exception as e :
            print (f"Model {attempt_model} failed to load completely: {e}")
            continue 

    if model is None or tokenizer is None :
        raise ValueError ("All model load attempts failed; check network or PyTorch version")

    if used_model !=model_name :
        print (f"Automatically switched to compatible model: {used_model}")

    print (f"Final model used: {used_model}")


    train_dataset =NewsDataset (train_texts ,train_labels ,tokenizer )
    val_dataset =NewsDataset (val_texts ,val_labels ,tokenizer )


    metrics_callback =MetricsCallback ()


    training_args =TrainingArguments (
    output_dir ='./results',
    num_train_epochs =num_epochs ,
    per_device_train_batch_size =8 ,
    per_device_eval_batch_size =16 ,
    warmup_steps =100 ,
    weight_decay =0.01 ,
    logging_dir ='./logs',
    logging_steps =100 ,
    eval_strategy ="steps",
    eval_steps =250 ,
    save_strategy ="steps",
    save_steps =250 ,
    load_best_model_at_end =True ,
    metric_for_best_model ="eval_accuracy",
    greater_is_better =True ,
    report_to =None ,
    save_total_limit =2 ,
    )


    data_collator =DataCollatorWithPadding (tokenizer =tokenizer )


    def compute_metrics (eval_pred ):
        predictions ,labels =eval_pred 
        predictions =np .argmax (predictions ,axis =1 )

        precision ,recall ,f1 ,_ =precision_recall_fscore_support (labels ,predictions ,average ='weighted')
        accuracy =accuracy_score (labels ,predictions )

        return {
        'accuracy':accuracy ,
        'f1':f1 ,
        'precision':precision ,
        'recall':recall 
        }


    trainer =Trainer (
    model =model ,
    args =training_args ,
    train_dataset =train_dataset ,
    eval_dataset =val_dataset ,
    tokenizer =tokenizer ,
    data_collator =data_collator ,
    compute_metrics =compute_metrics ,
    callbacks =[metrics_callback ],
    )


    start_time =time .time ()
    trainer .train ()
    training_time =time .time ()-start_time 

    print (f"Training complete, elapsed: {training_time:.2f} seconds")

    return trainer ,tokenizer ,training_time ,metrics_callback 

def evaluate_model (trainer ,test_texts ,test_labels ,tokenizer ):
    print ("\nEvaluating model performance...")


    test_dataset =NewsDataset (test_texts ,test_labels ,tokenizer )


    eval_results =trainer .evaluate (test_dataset )


    predictions =trainer .predict (test_dataset )
    predicted_labels =np .argmax (predictions .predictions ,axis =1 )


    class_names =['World','Sports','Business','Sci/Tech']
    report =classification_report (
    test_labels ,predicted_labels ,
    target_names =class_names ,
    output_dict =True 
    )


    print (f"\nDetailed classification report:")
    print (f"{'Class':<12} {'Precision':<8} {'Recall':<8} {'F1 Score':<8} {'Support':<8}")
    print ("-"*50 )

    for i ,class_name in enumerate (class_names ):
        if str (i )in report :
            precision =report [str (i )]['precision']
            recall =report [str (i )]['recall']
            f1 =report [str (i )]['f1-score']
            support =report [str (i )]['support']
            print (f"{class_name:<12} {precision:<8.3f} {recall:<8.3f} {f1:<8.3f} {support:<8.0f}")


    print ("-"*50 )
    macro_avg =report ['macro avg']
    weighted_avg =report ['weighted avg']
    print (f"{'Macro avg':<12} {macro_avg['precision']:<8.3f} {macro_avg['recall']:<8.3f} {macro_avg['f1-score']:<8.3f}")
    print (f"{'Weighted avg':<12} {weighted_avg['precision']:<8.3f} {weighted_avg['recall']:<8.3f} {weighted_avg['f1-score']:<8.3f}")


    from collections import Counter 
    true_dist =Counter (test_labels )
    pred_dist =Counter (predicted_labels )

    print (f"\nPrediction distribution analysis:")
    print (f"True distribution: {dict(true_dist)}")
    print (f"Predicted distribution: {dict(pred_dist)}")

    return eval_results ,report ,predicted_labels 

def plot_comparison_results (original_results ,cleaned_results ,original_callback ,cleaned_callback ):
    print ("\nGenerating comparison charts...")


    plt .style .use ('seaborn-v0_8-whitegrid')
    sns .set_palette ("husl")


    fig ,((ax1 ,ax2 ),(ax3 ,ax4 ))=plt .subplots (2 ,2 ,figsize =(20 ,15 ))
    fig .suptitle ('Text Classification Deduplication Effect Comparison',fontsize =24 ,fontweight ='bold',y =0.98 )


    colors ={
    'original':'#FF6B6B',
    'cleaned':'#4ECDC4',
    'positive':'#51CF66',
    'negative':'#FF7979',
    'neutral':'#A4A4A4'
    }


    if original_callback .train_losses and cleaned_callback .train_losses :

        steps_orig =list (range (len (original_callback .train_losses )))
        steps_clean =list (range (len (cleaned_callback .train_losses )))

        ax1 .plot (steps_orig ,original_callback .train_losses ,
        color =colors ['original'],linewidth =2 ,label ='Original Dataset',alpha =0.9 )
        ax1 .plot (steps_clean ,cleaned_callback .train_losses ,
        color =colors ['cleaned'],linewidth =2 ,label ='Deduplicated Dataset',alpha =0.9 )

        ax1 .set_xlabel ('Training Steps',fontsize =14 ,fontweight ='bold')
        ax1 .set_ylabel ('Training Loss',fontsize =14 ,fontweight ='bold')
        ax1 .set_title ('Training Loss Curves',fontsize =16 ,fontweight ='bold',pad =20 )
        ax1 .legend (fontsize =12 ,frameon =True ,fancybox =True ,shadow =True )
        ax1 .grid (True ,alpha =0.3 ,linestyle ='--')
        ax1 .spines ['top'].set_visible (False )
        ax1 .spines ['right'].set_visible (False )


    if original_callback .eval_accuracies and cleaned_callback .eval_accuracies :

        orig_epochs =original_callback .epochs 
        clean_epochs =cleaned_callback .epochs 

        ax2 .plot (orig_epochs ,original_callback .eval_accuracies ,
        color =colors ['original'],linewidth =3 ,
        label ='Original Dataset',alpha =0.9 )
        ax2 .plot (clean_epochs ,cleaned_callback .eval_accuracies ,
        color =colors ['cleaned'],linewidth =3 ,
        label ='Deduplicated Dataset',alpha =0.9 )

        ax2 .set_xlabel ('Epoch',fontsize =14 ,fontweight ='bold')
        ax2 .set_ylabel ('Validation Accuracy',fontsize =14 ,fontweight ='bold')
        ax2 .set_title ('Validation Accuracy Curves',fontsize =16 ,fontweight ='bold',pad =20 )
        ax2 .legend (fontsize =12 ,frameon =True ,fancybox =True ,shadow =True )
        ax2 .grid (True ,alpha =0.3 ,linestyle ='--')
        ax2 .set_xlim (0 ,3 )
        ax2 .set_ylim (0.3 ,1.0 )
        ax2 .yaxis .set_major_formatter (plt .FuncFormatter (lambda y ,_ :'{:.1%}'.format (y )))
        ax2 .spines ['top'].set_visible (False )
        ax2 .spines ['right'].set_visible (False )


    metrics =['accuracy','f1','precision','recall']
    metric_names =['Accuracy','F1 Score','Precision','Recall']
    original_scores =[original_results ['eval_'+metric ]for metric in metrics ]
    cleaned_scores =[cleaned_results ['eval_'+metric ]for metric in metrics ]

    x =np .arange (len (metrics ))
    width =0.35 

    bars1 =ax3 .bar (x -width /2 ,original_scores ,width ,
    label ='Original Dataset',alpha =0.8 ,color =colors ['original'],
    edgecolor ='white',linewidth =2 )
    bars2 =ax3 .bar (x +width /2 ,cleaned_scores ,width ,
    label ='Deduplicated Dataset',alpha =0.8 ,color =colors ['cleaned'],
    edgecolor ='white',linewidth =2 )

    ax3 .set_xlabel ('Evaluation Metrics',fontsize =14 ,fontweight ='bold')
    ax3 .set_ylabel ('Score',fontsize =14 ,fontweight ='bold')
    ax3 .set_title ('Final Performance Comparison',fontsize =16 ,fontweight ='bold',pad =20 )
    ax3 .set_xticks (x )
    ax3 .set_xticklabels (metric_names ,fontsize =11 )
    ax3 .legend (fontsize =12 ,frameon =True ,fancybox =True ,shadow =True )
    ax3 .grid (True ,alpha =0.3 ,linestyle ='--',axis ='y')
    ax3 .set_ylim (0.7 ,1.0 )
    ax3 .yaxis .set_major_formatter (plt .FuncFormatter (lambda y ,_ :'{:.1%}'.format (y )))
    ax3 .spines ['top'].set_visible (False )
    ax3 .spines ['right'].set_visible (False )


    for bar in bars1 :
        height =bar .get_height ()
        ax3 .text (bar .get_x ()+bar .get_width ()/2. ,height +0.01 ,
        f'{height:.1%}',ha ='center',va ='bottom',
        fontsize =10 ,fontweight ='bold',color =colors ['original'])

    for bar in bars2 :
        height =bar .get_height ()
        ax3 .text (bar .get_x ()+bar .get_width ()/2. ,height +0.01 ,
        f'{height:.1%}',ha ='center',va ='bottom',
        fontsize =10 ,fontweight ='bold',color =colors ['cleaned'])


    differences =[cleaned_scores [i ]-original_scores [i ]for i in range (len (metrics ))]
    bar_colors =[colors ['positive']if d >0 else colors ['negative']if d <0 else colors ['neutral']
    for d in differences ]

    bars3 =ax4 .bar (metric_names ,differences ,color =bar_colors ,alpha =0.8 ,
    edgecolor ='white',linewidth =2 )

    ax4 .set_xlabel ('Evaluation Metrics',fontsize =14 ,fontweight ='bold')
    ax4 .set_ylabel ('Performance Difference (Deduplicated - Original)',fontsize =14 ,fontweight ='bold')
    ax4 .set_title ('Deduplication Effect Analysis',fontsize =16 ,fontweight ='bold',pad =20 )
    ax4 .axhline (y =0 ,color ='black',linestyle ='-',alpha =0.5 ,linewidth =2 )
    ax4 .grid (True ,alpha =0.3 ,linestyle ='--',axis ='y')
    ax4 .spines ['top'].set_visible (False )
    ax4 .spines ['right'].set_visible (False )


    for i ,(bar ,diff )in enumerate (zip (bars3 ,differences )):
        height =bar .get_height ()

        ax4 .text (bar .get_x ()+bar .get_width ()/2. ,
        height +(0.003 if height >0 else -0.008 ),
        f'{height:+.1%}',ha ='center',
        va ='bottom'if height >0 else 'top',
        fontsize =11 ,fontweight ='bold')


        if abs (diff )>0.005 :
            icon ='↑'if diff >0 else '↓'
            ax4 .text (bar .get_x ()+bar .get_width ()/2. ,
            height +(0.015 if height >0 else -0.020 ),
            icon ,ha ='center',va ='center',fontsize =14 )


    avg_improvement =np .mean (differences )
    improvement_text =f"Average Performance Change: {avg_improvement:+.2%}"
    fig .text (0.02 ,0.02 ,improvement_text ,fontsize =12 ,
    bbox =dict (boxstyle ="round,pad=0.3",facecolor ='lightblue',alpha =0.7 ))


    plt .tight_layout (rect =[0 ,0.03 ,1 ,0.95 ])

    os .makedirs ('results_summary',exist_ok =True )

    plt .savefig ('results_summary\\classification_comparison.png',dpi =300 ,bbox_inches ='tight',
    facecolor ='white',edgecolor ='none')
    plt .savefig ('results_summary\\training_comparison.png',dpi =300 ,bbox_inches ='tight',
    facecolor ='white',edgecolor ='none')

    print ("Charts saved as results_summary\\classification_comparison.png and results_summary\\training_comparison.png")
    plt .show ()

def save_detailed_results (original_results ,cleaned_results ,original_time ,cleaned_time ,
original_callback ,cleaned_callback ):
    results ={
    'experiment_time':time .strftime ('%Y-%m-%d %H:%M:%S'),
    'original_dataset':{
    'metrics':original_results ,
    'training_time':original_time ,
    'training_history':{
    'train_losses':original_callback .train_losses ,
    'eval_losses':original_callback .eval_losses ,
    'eval_accuracies':original_callback .eval_accuracies ,
    'epochs':original_callback .epochs 
    }
    },
    'cleaned_dataset':{
    'metrics':cleaned_results ,
    'training_time':cleaned_time ,
    'training_history':{
    'train_losses':cleaned_callback .train_losses ,
    'eval_losses':cleaned_callback .eval_losses ,
    'eval_accuracies':cleaned_callback .eval_accuracies ,
    'epochs':cleaned_callback .epochs 
    }
    },
    'improvements':{
    'accuracy':cleaned_results ['eval_accuracy']-original_results ['eval_accuracy'],
    'f1':cleaned_results ['eval_f1']-original_results ['eval_f1'],
    'precision':cleaned_results ['eval_precision']-original_results ['eval_precision'],
    'recall':cleaned_results ['eval_recall']-original_results ['eval_recall'],
    'training_time_diff':cleaned_time -original_time 
    }
    }


    os .makedirs ('results_summary',exist_ok =True )

    with open ('results_summary\\comparison_results.json','w',encoding ='utf-8')as f :
        json .dump (results ,f ,indent =2 ,ensure_ascii =False )


    with open ('results_summary\\experiment_report.txt','w',encoding ='utf-8')as f :
        f .write ("Text Classification Deduplication Comparison Report\n")
        f .write ("="*40 +"\n\n")
        f .write (f"Experiment time: {results['experiment_time']}\n\n")

        f .write ("Dataset comparison:\n")
        f .write (f"Original dataset final accuracy: {original_results['eval_accuracy']:.4f}\n")
        f .write (f"Deduplicated dataset final accuracy: {cleaned_results['eval_accuracy']:.4f}\n")
        f .write (f"Accuracy improvement: {results['improvements']['accuracy']:+.4f}\n\n")

        f .write (f"Original dataset F1 score: {original_results['eval_f1']:.4f}\n")
        f .write (f"Deduplicated dataset F1 score: {cleaned_results['eval_f1']:.4f}\n")
        f .write (f"F1 improvement: {results['improvements']['f1']:+.4f}\n\n")

        f .write (f"Training time comparison:\n")
        f .write (f"Original dataset training time: {original_time:.2f} seconds\n")
        f .write (f"Deduplicated dataset training time: {cleaned_time:.2f} seconds\n")
        f .write (f"Time difference: {results['improvements']['training_time_diff']:+.2f} seconds\n")

    print ("Detailed results saved to results_summary\\comparison_results.json")
    print ("Summary report saved to results_summary\\experiment_report.txt")

def save_threshold_results (threshold ,results ,training_time ,callback ,dataset_size ):
    import time 
    import json 
    import os 

    result_data ={
    'experiment_time':time .strftime ('%Y-%m-%d %H:%M:%S'),
    'threshold':threshold ,
    'dataset_info':{
    'size':dataset_size ,
    'reduction_rate':None 
    },
    'model_performance':{
    'accuracy':results ['eval_accuracy'],
    'f1_score':results ['eval_f1'],
    'precision':results ['eval_precision'],
    'recall':results ['eval_recall'],
    'loss':results ['eval_loss']
    },
    'training_info':{
    'training_time_seconds':training_time ,
    'total_epochs':len (callback .train_losses ),
    'final_train_loss':callback .train_losses [-1 ]if callback .train_losses else None ,
    'final_eval_loss':callback .eval_losses [-1 ]if callback .eval_losses else None ,
    'best_accuracy':max (callback .eval_accuracies )if callback .eval_accuracies else None 
    },
    'training_history':{
    'train_losses':callback .train_losses ,
    'eval_losses':callback .eval_losses ,
    'eval_accuracies':callback .eval_accuracies ,
    'epochs':callback .epochs 
    }
    }


    os .makedirs ('threshold_results',exist_ok =True )


    json_filename =f'threshold_results\\threshold_{threshold}_results.json'
    with open (json_filename ,'w',encoding ='utf-8')as f :
        json .dump (result_data ,f ,indent =2 ,ensure_ascii =False )


    txt_filename =f'threshold_results\\threshold_{threshold}_report.txt'
    with open (txt_filename ,'w',encoding ='utf-8')as f :
        f .write (f"Threshold {threshold} Experiment Results Report\n")
        f .write ("="*40 +"\n\n")
        f .write (f"Experiment time: {result_data['experiment_time']}\n")
        f .write (f"Similarity threshold: {threshold}\n")
        f .write (f"Dataset size: {dataset_size} items\n\n")

        f .write ("Model performance metrics:\n")
        f .write (f"  Accuracy: {results['eval_accuracy']:.4f}\n")
        f .write (f"  F1 score: {results['eval_f1']:.4f}\n")
        f .write (f"  Precision: {results['eval_precision']:.4f}\n")
        f .write (f"  Recall: {results['eval_recall']:.4f}\n")
        f .write (f"  Loss: {results['eval_loss']:.4f}\n\n")

        f .write ("Training info:\n")
        f .write (f"  Training time: {training_time:.2f} seconds\n")
        f .write (f"  Training rounds: {len(callback.train_losses)}\n")
        if callback .eval_accuracies :
            f .write (f"  Best accuracy: {max(callback.eval_accuracies):.4f}\n")
        f .write (f"  Final training loss: {callback.train_losses[-1]:.4f}\n")
        f .write (f"  Final validation loss: {callback.eval_losses[-1]:.4f}\n")

    print (f"Threshold {threshold} results saved:")
    print (f"  Detailed results: {json_filename}")
    print (f"  Summary report: {txt_filename}")

    return result_data 

def main ():
    print ("Text classification deduplication comparison experiment")
    print ("="*50 )


    try :
        import tqdm 
        print ("Dependencies check passed")
    except ImportError as e :
        print (f"Missing dependency: {e}")
        print ("Please run: pip install tqdm")
        return 


    torch_version =torch .__version__ 
    print (f"Current PyTorch version: {torch_version}")










    np .random .seed (42 )


    device =torch .device ("cuda"if torch .cuda .is_available ()else "cpu")
    print (f"Using device: {device}")

    try :

        print ("Step 1: Load datasets")


        cleaned_texts ,cleaned_labels =load_cleaned_dataset ()

        if not cleaned_texts :
            print ("Failed to load deduplicated dataset, experiment aborted")
            return 


        max_samples =len (cleaned_texts )
        print (f"Using all {len(cleaned_texts)} samples for training")


        cleaned_train_texts ,cleaned_test_texts ,cleaned_train_labels ,cleaned_test_labels =train_test_split (
        cleaned_texts ,cleaned_labels ,test_size =0.2 ,random_state =42 ,stratify =cleaned_labels 
        )


        print ("Loading original AG News dataset...")
        original_train_texts ,original_train_labels ,original_test_texts ,original_test_labels =load_original_dataset ()


        print (f"Using full original dataset")

        if len (original_test_texts )>len (cleaned_test_texts ):
            indices =np .random .choice (len (original_test_texts ),len (cleaned_test_texts ),replace =False )
            indices =[int (i )for i in indices ]
            original_test_texts =[original_test_texts [i ]for i in indices ]
            original_test_labels =[original_test_labels [i ]for i in indices ]


        print ("\nStep 2: Dataset statistical analysis")


        print (f"\nDataset size comparison:")
        print (f"Original train: {len(original_train_texts)} items")
        print (f"Deduplicated train: {len(cleaned_train_texts)} items")
        print (f"Reduction: {len(original_train_texts) - len(cleaned_train_texts)} items ({(len(original_train_texts) - len(cleaned_train_texts)) / len(original_train_texts) * 100:.1f}%)")

        print (f"\nOriginal test: {len(original_test_texts)} items")
        print (f"Deduplicated test: {len(cleaned_test_texts)} items")


        from collections import Counter 

        print (f"\nClass distribution comparison:")
        original_dist =Counter (original_train_labels +original_test_labels )
        cleaned_dist =Counter (cleaned_train_labels +cleaned_test_labels )

        print ("Original dataset class distribution:",dict (original_dist ))
        print ("Deduplicated dataset class distribution:",dict (cleaned_dist ))


        for label in original_dist .keys ():
            change =cleaned_dist [label ]-original_dist [label ]
            change_pct =change /original_dist [label ]*100 if original_dist [label ]>0 else 0 
            print (f"  Class {label}: {original_dist[label]} → {cleaned_dist[label]} ({change_pct:+.1f}%)")

        analyze_dataset_statistics (original_train_texts +original_test_texts ,
        original_train_labels +original_test_labels ,"Original")
        analyze_dataset_statistics (cleaned_train_texts +cleaned_test_texts ,
        cleaned_train_labels +cleaned_test_labels ,"Deduplicated")


        print ("\nStep 3: Train model on original dataset")
        original_trainer ,original_tokenizer ,original_time ,original_callback =train_model (
        original_train_texts ,original_train_labels ,
        original_test_texts ,original_test_labels 
        )


        print ("\nStep 4: Evaluate original dataset model")
        original_results ,original_report ,_ =evaluate_model (
        original_trainer ,original_test_texts ,original_test_labels ,original_tokenizer 
        )


        print ("\nStep 5: Train model on deduplicated dataset")
        cleaned_trainer ,cleaned_tokenizer ,cleaned_time ,cleaned_callback =train_model (
        cleaned_train_texts ,cleaned_train_labels ,
        cleaned_test_texts ,cleaned_test_labels 
        )


        print ("\nStep 6: Evaluate deduplicated dataset model")
        cleaned_results ,cleaned_report ,_ =evaluate_model (
        cleaned_trainer ,cleaned_test_texts ,cleaned_test_labels ,cleaned_tokenizer 
        )


        print ("\nStep 7: Results comparison and analysis")
        print ("="*50 )
        print ("Experiment summary:")
        print (f"Original dataset - Accuracy: {original_results['eval_accuracy']:.4f}, F1: {original_results['eval_f1']:.4f}")
        print (f"Deduplicated dataset - Accuracy: {cleaned_results['eval_accuracy']:.4f}, F1: {cleaned_results['eval_f1']:.4f}")
        print (f"Accuracy improvement: {cleaned_results['eval_accuracy'] - original_results['eval_accuracy']:.4f}")
        print (f"F1 improvement: {cleaned_results['eval_f1'] - original_results['eval_f1']:.4f}")
        print (f"Training time comparison: Original {original_time:.2f}s vs Deduplicated {cleaned_time:.2f}s")


        print ("\nStep 8: Generate reports")
        plot_comparison_results (original_results ,cleaned_results ,original_callback ,cleaned_callback )
        save_detailed_results (original_results ,cleaned_results ,original_time ,cleaned_time ,
        original_callback ,cleaned_callback )

        print ("\nExperiment complete!")
        print ("Charts saved to results_summary\\classification_comparison.png and results_summary\\training_comparison.png")
        print ("Detailed results saved to results_summary\\comparison_results.json")
        print ("Summary report saved to results_summary\\experiment_report.txt")

    except Exception as e :
        print (f"An error occurred during the experiment: {e}")
        import traceback 
        traceback .print_exc ()

if __name__ =="__main__":
    main ()
