from datasets import load_dataset 
from text .method .dataset .clean_the_dataset import DatasetCleaner 
from text .method .dataset .jaccard_deduplication import clear_global_memory ,quick_jaccard_deduplicate 
import time 
import os 
from tqdm import tqdm 

def chunked_deduplication (dataset ,text_field ,threshold ,ngram_size ,chunk_size =5000 ,sample_size =1000 ):
    print (f"Starting chunked deduplication: total={len(dataset)}, chunk_size={chunk_size}, sample_size={sample_size}")

    all_kept_indices =[]
    processed_count =0 


    for chunk_start in tqdm (range (0 ,len (dataset ),chunk_size ),desc ="chunk processing"):
        chunk_end =min (chunk_start +chunk_size ,len (dataset ))
        chunk =dataset .select (range (chunk_start ,chunk_end ))

        print (f"Processing chunk {chunk_start}-{chunk_end} ({len(chunk)} items)")


        deduplicated_chunk =quick_jaccard_deduplicate (
        chunk ,text_field ,threshold ,ngram_size ,sample_size 
        )


        if len (deduplicated_chunk )>0 :

            chunk_kept_indices =list (range (len (deduplicated_chunk )))
            global_indices =[chunk_start +i for i in chunk_kept_indices ]
            all_kept_indices .extend (global_indices )

        processed_count +=len (chunk )
        print (f"Chunk complete: kept {len(deduplicated_chunk)}/{len(chunk)} items")


    if all_kept_indices :
        final_dataset =dataset .select (all_kept_indices )
        print (f"Chunked dedup complete: {len(dataset)} -> {len(final_dataset)} items")
        return final_dataset 
    else :
        print ("Warning: all data were deduplicated")
        return dataset .select ([])


dataset =load_dataset ("ag_news")


thresholds =[0.1 ,0.2 ,0.3 ,0.4 ,0.5 ,0.6 ,0.7 ,0.8 ,0.9 ]


os .makedirs ("batch_cleaned_datasets",exist_ok =True )

    for threshold in thresholds :
    print (f"\n{'='*60}")
    print (f"Processing threshold: {threshold}")
    print (f"{'='*60}")


    clear_global_memory ()

    start_time =time .time ()


    print ("Processing training set...")
    cleaned_train =chunked_deduplication (
    dataset ['train'],
    'text',
    threshold ,
    ngram_size =3 ,
    chunk_size =5000 ,
    sample_size =1000 
    )

    print ("\nProcessing test set...")
    cleaned_test =chunked_deduplication (
    dataset ['test'],
    'text',
    threshold ,
    ngram_size =3 ,
    chunk_size =2000 ,
    sample_size =500 
    )

    processing_time =time .time ()-start_time 


    train_path =f"batch_cleaned_datasets/ag_news_train_threshold_{threshold}.json"
    test_path =f"batch_cleaned_datasets/ag_news_test_threshold_{threshold}.json"

    cleaned_train .to_json (train_path )
    cleaned_test .to_json (test_path )


    train_reduction =(len (dataset ['train'])-len (cleaned_train ))/len (dataset ['train'])*100 
    test_reduction =(len (dataset ['test'])-len (cleaned_test ))/len (dataset ['test'])*100 
    total_processed =len (dataset ['train'])+len (dataset ['test'])
    speed =total_processed /processing_time if processing_time >0 else 0 

    print (f"\nThreshold {threshold} processing complete!")
    print (f"Train: {len(dataset['train'])} -> {len(cleaned_train)} items (reduction {train_reduction:.1f}%)")
    print (f"Test: {len(dataset['test'])} -> {len(cleaned_test)} items (reduction {test_reduction:.1f}%)")
    print (f"Processing time: {processing_time:.2f} seconds")
    print (f"Processing speed: {speed:.0f} items/sec")


    clear_global_memory ()

print ("\n🎉 All threshold processing complete!")
