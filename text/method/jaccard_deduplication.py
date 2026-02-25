
import re 
from datasets import load_dataset 
from collections import defaultdict 
from tqdm import tqdm 
import gc 


global_kept_ngrams =[]

def normalize_text (text ):
    if not isinstance (text ,str ):
        text =str (text )
    text =text .lower ()
    text =re .sub (r'\s+',' ',text )
    text =re .sub (r'[^\w\s\u4e00-\u9fff]','',text )
    return text .strip ()

def clear_global_memory ():
    global global_kept_ngrams 
    global_kept_ngrams .clear ()
    global_kept_ngrams =[]
    gc .collect ()
    print ("Global memory cleared.")

def clear_all_memory ():
    global global_kept_ngrams 


    global_kept_ngrams .clear ()
    global_kept_ngrams =[]


    gc .collect ()

    print ("All memory cleared.")

def get_ngrams (text ,n =3 ):
    normalized =normalize_text (text )
    words =normalized .split ()


    char_ngrams =set ()
    for i in range (len (normalized )-n +1 ):
        char_ngrams .add (normalized [i :i +n ])


    word_ngrams =set ()
    for i in range (len (words )-n +1 ):
        word_ngrams .add (' '.join (words [i :i +n ]))


    return char_ngrams |word_ngrams 

def jaccard_similarity (text1 ,text2 ,n =3 ):
    ngrams1 =get_ngrams (text1 ,n )
    ngrams2 =get_ngrams (text2 ,n )

    if len (ngrams1 )==0 and len (ngrams2 )==0 :
        return 1.0 

    intersection =len (ngrams1 &ngrams2 )
    union =len (ngrams1 |ngrams2 )

    return intersection /union if union >0 else 0.0 

def deduplicate_dataset_jaccard (dataset ,text_field ='text',threshold =0.8 ,ngram_size =3 ):
    print (f"Starting Jaccard deduplication, original items: {len(dataset)}")
    print (f"Similarity threshold: {threshold:.2f}, n-gram size: {ngram_size}")

    keep_indices =[]
    kept_texts =[]

    for i ,item in tqdm (enumerate (dataset ),total =len (dataset ),desc ="Jaccard dedup progress"):
        text =item [text_field ]
        is_duplicate =False 


        for kept_text in kept_texts :
            similarity =jaccard_similarity (text ,kept_text ,ngram_size )
            if similarity >=threshold :
                is_duplicate =True 
                break 

        if not is_duplicate :
            keep_indices .append (i )
            kept_texts .append (text )

    undeduplicated =dataset .select (keep_indices )
    removed =len (dataset )-len (undeduplicated )

    print (f"Deduplication complete, kept: {len(undeduplicated)}")
    print (f"Removed duplicates: {removed} ({removed/len(dataset)*100:.1f}%)")

    return undeduplicated 
def deduplicate_cross_splits_jaccard (dataset_dict ,text_field ='text',threshold =0.8 ,ngram_size =3 ):
    global global_kept_ngrams 

    print ("Starting cross-split Jaccard dedup...")
    print (f"Similarity threshold: {threshold:.2f}, n-gram size: {ngram_size}")
    print (f"Current global_kept_ngrams length: {len(global_kept_ngrams)}")

    result ={}


    splits =['train','test']+[k for k in dataset_dict .keys ()if k not in ['train','test']]

    for split_name in splits :
        if split_name not in dataset_dict :
            continue 

        dataset =dataset_dict [split_name ]
        print (f"Processing split {split_name} (original: {len(dataset)})")

        keep_indices =[]


        from tqdm import tqdm 

        for i ,item in tqdm (enumerate (dataset ),total =len (dataset ),desc =f"processing {split_name} split"):
            text =item [text_field ]
            current_ngrams =get_ngrams (normalize_text (text ),ngram_size )
            is_duplicate =False 


            for kept_ngrams in global_kept_ngrams :
                similarity =len (current_ngrams &kept_ngrams )/len (current_ngrams |kept_ngrams )if (current_ngrams |kept_ngrams )else 0 
                if similarity >=threshold :
                    is_duplicate =True 
                    break 

            if not is_duplicate :
                keep_indices .append (i )
                global_kept_ngrams .append (current_ngrams )


                if len (global_kept_ngrams )>50000 :

                    global_kept_ngrams =global_kept_ngrams [-25000 :]
                    print (f"Memory optimization: keeping last 25000 n-gram sets")

        result [split_name ]=dataset .select (keep_indices )
        print (f"{split_name} deduplicated: {len(result[split_name])} items")


        gc .collect ()

    return result 

def quick_jaccard_deduplicate (dataset ,text_field ='text',threshold =0.8 ,ngram_size =3 ,sample_size =100 ):
    print (f"Starting quick Jaccard dedup, original items: {len(dataset)}")
    print (f"Similarity threshold: {threshold:.2f}, sample size: {sample_size}")

    keep_indices =[]
    kept_ngrams =[]

    for i ,item in tqdm (enumerate (dataset ),total =len (dataset ),desc ="Quick Jaccard dedup"):
        text =item [text_field ]
        current_ngrams =get_ngrams (text ,ngram_size )
        is_duplicate =False 



        start_idx =max (0 ,len (kept_ngrams )-sample_size )
        for j in range (start_idx ,len (kept_ngrams )):
            kept_ngrams_set =kept_ngrams [j ]


            intersection =len (current_ngrams &kept_ngrams_set )
            union =len (current_ngrams |kept_ngrams_set )
            similarity =intersection /union if union >0 else 0.0 

            if similarity >=threshold :
                is_duplicate =True 
                break 

        if not is_duplicate :
            keep_indices .append (i )
            kept_ngrams .append (current_ngrams )

    deduplicated =dataset .select (keep_indices )
    removed =len (dataset )-len (deduplicated )

    print (f"Deduplication complete, kept: {len(deduplicated)}")
    print (f"Removed duplicates: {removed} ({removed/len(dataset)*100:.1f}%)")

    return deduplicated 

def main ():
    print ("Dataset deduplication tool based on Jaccard similarity")
    print ("="*50 )

    try :

        print ("Loading IMDB dataset...")
        ds =load_dataset ("stanfordnlp/imdb")

        print (f"Original dataset info:")
        print (f"  Train: {len(ds['train'])} items")
        print (f"  Test: {len(ds['test'])} items")


        print (f"\nParameter settings:")
        threshold =float (input ("Enter similarity threshold (0.0-1.0, default 0.8): ")or "0.8")
        ngram_size =int (input ("Enter n-gram size (default 3): ")or "3")

        print (f"\nChoose dedup method:")
        print ("1. Jaccard dedup on train set")
        print ("2. Jaccard dedup on test set")
        print ("3. Cross-split Jaccard dedup (recommended)")
        print ("4. Quick Jaccard dedup (sampling optimization)")
        print ("5. Small sample test (first 100 items)")

        choice = input("Enter choice (1/2/3/4/5): ").strip()

        if choice =='1':
            print (f"\nPerforming Jaccard dedup on train set...")
            train_dedup =deduplicate_dataset_jaccard (ds ['train'],'text',threshold ,ngram_size )

        elif choice =='2':
            print (f"\nPerforming Jaccard dedup on test set...")
            test_dedup =deduplicate_dataset_jaccard (ds ['test'],'text',threshold ,ngram_size )

        elif choice =='3':
            print (f"\nPerforming cross-split Jaccard dedup...")
            dedup_ds =deduplicate_cross_splits_jaccard (ds ,'text',threshold ,ngram_size )

            print (f"\nCross-split dedup results:")
            for split ,data in dedup_ds .items ():
                original_size =len (ds [split ])
                new_size =len (data )
                reduction = (original_size - new_size) / original_size * 100
                print (f"  {split}: {original_size} -> {new_size} items (reduction {reduction:.1f}%)")

        elif choice =='4':
            print (f"\nPerforming quick Jaccard dedup...")
            sample_size = int(input("Enter sample size (default 100): ") or "100")
            train_dedup =quick_jaccard_deduplicate (ds ['train'],'text',threshold ,ngram_size ,sample_size )

        elif choice =='5':
            print (f"\nSmall sample test...")

            small_dataset =ds ['train'].select (range (100 ))
            dedup_result =deduplicate_dataset_jaccard (small_dataset ,'text',threshold ,ngram_size )

            print (f"\nTest results:")
            print (f"  Original: 100 items")
            print (f"  After dedup: {len(dedup_result)} items")
            print (f"  Duplicate rate: {(100 - len(dedup_result))}%")

        else :
            print ("Invalid choice...")

    except Exception as e :
        print (f"An error occurred during processing: {e}")
        print ("Please ensure network connection is available and the 'datasets' library is installed")

if __name__ =="__main__":
    main ()
