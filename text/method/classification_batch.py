from text .method .dataset .classification_comparison import load_cleaned_dataset ,train_test_split ,train_model ,evaluate_model ,save_threshold_results 
from datasets import load_dataset 
import re 
import torch 
import numpy as np 
import random 
import os 


def main ():
    np .random .seed (42 )
    device =torch .device ('cuda'if torch .cuda .is_available ()else 'cpu')
    print (f"Using device: {device}")

    thresholds =[0.1 ,0.2 ,0.3 ,0.4 ,0.5 ,0.6 ,0.7 ,0.8 ,0.9 ]
    all_results =[]


    os .makedirs ('threshold_results',exist_ok =True )

    for threshold in thresholds :
        try :
            print (f"\n{'='*60}")
            print (f"Processing threshold: {threshold}")
            print (f"{'='*60}")

            print ("Loading deduplicated dataset...")
            cleaned_texts ,cleaned_labels =load_cleaned_dataset (threshold )
            if not cleaned_texts :
                print (f"Dataset for threshold {threshold} failed to load or is empty, skipping")
                continue 

            dataset_size =len (cleaned_texts )
            print (f"Dataset size: {dataset_size} items")


            cleaned_train_texts ,cleaned_test_texts ,cleaned_train_labels ,cleaned_test_labels =train_test_split (
            cleaned_texts ,cleaned_labels ,test_size =0.2 ,random_state =42 ,stratify =cleaned_labels 
            )

            print (f"Training set: {len(cleaned_train_texts)} items")
            print (f"Test set: {len(cleaned_test_texts)} items")

            print ("\nStarting model training...")
            cleaned_trainer ,cleaned_tokenizer ,cleaned_time ,cleaned_callback =train_model (
            cleaned_train_texts ,cleaned_train_labels ,
            cleaned_test_texts ,cleaned_test_labels 
            )

            print ("\nEvaluating model performance...")
            cleaned_results ,cleaned_report ,_ =evaluate_model (
            cleaned_trainer ,cleaned_test_texts ,cleaned_test_labels ,cleaned_tokenizer 
            )

            print (f"\nThreshold {threshold} training completed:")
            print (f"Accuracy: {cleaned_results['eval_accuracy']:.4f}")
            print (f"F1 score: {cleaned_results['eval_f1']:.4f}")
            print (f"Training time: {cleaned_time:.2f} seconds")


            threshold_data =save_threshold_results (
            threshold ,cleaned_results ,cleaned_time ,cleaned_callback ,dataset_size 
            )
            all_results .append (threshold_data )

        except Exception as e :
            print (f"Error processing threshold {threshold}: {str(e)}")
            import traceback 
            traceback .print_exc ()
            continue 


    if all_results :
        summary_results ={
        'experiment_summary':{
        'total_thresholds_processed':len (all_results ),
        'thresholds':[r ['threshold']for r in all_results ],
        'best_threshold':max (all_results ,key =lambda x :x ['model_performance']['accuracy'])['threshold'],
        'best_accuracy':max (r ['model_performance']['accuracy']for r in all_results ),
        'best_f1':max (r ['model_performance']['f1_score']for r in all_results )
        },
        'detailed_results':all_results 
        }

        import json 
        with open ('threshold_results\\all_thresholds_summary.json','w',encoding ='utf-8')as f :
            json .dump (summary_results ,f ,indent =2 ,ensure_ascii =False )


        with open ('threshold_results\\summary_report.txt','w',encoding ='utf-8')as f :
            f .write ("All thresholds experiment summary report\n")
            f .write ("="*50 +"\n\n")
            f .write (f"Total thresholds processed: {len(all_results)}\n")
            f .write (f"Best threshold: {summary_results['experiment_summary']['best_threshold']}\n")
            f .write (f"Best accuracy: {summary_results['experiment_summary']['best_accuracy']:.4f}\n")
            f .write (f"Best F1 score: {summary_results['experiment_summary']['best_f1']:.4f}\n\n")

            f .write ("Detailed results for each threshold:\n")
            f .write ("-"*80 +"\n")
            for result in all_results :
                f .write (f"Threshold {result['threshold']:<4} | ")
                f .write (f"Accuracy: {result['model_performance']['accuracy']:.4f} | ")
                f .write (f"F1: {result['model_performance']['f1_score']:.4f} | ")
                f .write (f"Dataset size: {result['dataset_info']['size']:>6} | ")
                f .write (f"Training time: {result['training_info']['training_time_seconds']:>6.1f}s\n")

        print (f"\nAll thresholds processed!")
        print (f"Processed {len(all_results)} thresholds")
        print (f"Best threshold: {summary_results['experiment_summary']['best_threshold']}")
        print (f"Summary saved to: threshold_results\\all_thresholds_summary.json")
        print (f"Summary report saved to: threshold_results\\summary_report.txt")
    else :
        print ("No thresholds were successfully processed")


if __name__ =="__main__":
    main ()
