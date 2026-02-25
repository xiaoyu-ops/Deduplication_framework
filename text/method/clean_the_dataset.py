

import os 
import time 
import multiprocessing as mp 
from concurrent .futures import ProcessPoolExecutor ,ThreadPoolExecutor 
from datasets import load_dataset ,Dataset ,DatasetDict ,concatenate_datasets 
from tqdm import tqdm 
from jaccard_deduplication import (
deduplicate_dataset_jaccard ,
deduplicate_cross_splits_jaccard ,
quick_jaccard_deduplicate ,
jaccard_similarity ,
get_ngrams ,
normalize_text 
)
import sys 
current_dir =os .path .dirname (os .path .abspath (__file__ ))
parent_dir =os .path .dirname (current_dir )
root_dir =os .path .dirname (parent_dir )


sys .path .insert (0 ,root_dir )
print (f"Added repository root to path: {root_dir}")


from env_manager .manager import EnvManager 


def process_chunk_worker (chunk_data ):
    try :
        from jaccard_deduplication import jaccard_similarity ,normalize_text ,get_ngrams 
        from datasets import Dataset 

        text_field ,threshold ,ngram_size =chunk_data ['params']
        chunk_items =chunk_data ['chunk_items']
        chunk_id =chunk_data .get ('chunk_id',0 )
        original_range =chunk_data .get ('original_range','unknown')

        print (f"Chunk {chunk_id} ({original_range}): starting processing...")

        if not chunk_items :
            print (f"Chunk {chunk_id}: no data")
            return None 

        print (f"Chunk {chunk_id}: received {len(chunk_items)} items")


        valid_items =[]
        for i ,item in enumerate (chunk_items ):
            if isinstance (item ,dict )and text_field in item :
                text =item [text_field ]
                if text and isinstance (text ,str )and len (text .strip ())>0 :
                    valid_items .append (item )
                else :
                    print (f"Chunk {chunk_id}: item {i} invalid text")
            else :
                print (f"Chunk {chunk_id}: item {i} invalid format")

        if not valid_items :
            print (f"Chunk {chunk_id}: no valid data")
            return None 

        print (f"Chunk {chunk_id}: valid items {len(valid_items)}")


        try :
            chunk =Dataset .from_list (valid_items )
            print (f"Chunk {chunk_id}: Dataset rebuild succeeded, contains {len(chunk)} items")
        except Exception as e :
            print (f"Chunk {chunk_id}: Dataset rebuild failed: {e}")
            return None 


        kept_indices =[]
        seen_ngram_sets =[]
        seen_text_hashes =set ()
        processed_count =0 
        start_time =time .time ()


        check_interval =min (500 ,len (chunk )//10 )
        early_stop_threshold =0.95 


        pbar =tqdm (
        total =len (chunk ),
        desc =f"Chunk {chunk_id}",
        position =chunk_id ,
        leave =False ,
        bar_format ='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] kept:{postfix}'
        )

        for i ,item in enumerate (chunk ):
            processed_count +=1 

            if not isinstance (item ,dict )or text_field not in item :
                pbar .update (1 )
                continue 

            text =item [text_field ]
            if not text or not isinstance (text ,str ):
                pbar .update (1 )
                continue 


            text_hash =hash (text .strip ())
            if text_hash in seen_text_hashes :
                pbar .update (1 )
                continue 


            current_ngrams =set (get_ngrams (normalize_text (text ),ngram_size ))


            text_len =len (text )
            ngram_count =len (current_ngrams )

            is_duplicate =False 
            similarity_checks =0 


            for j ,seen_ngrams in enumerate (seen_ngram_sets ):

                if ngram_count ==0 or len (seen_ngrams )==0 :
                    continue 


                size_ratio =ngram_count /len (seen_ngrams )
                if size_ratio >2.0 or size_ratio <0.5 :
                    continue 


                intersection_size =len (current_ngrams &seen_ngrams )
                min_size =min (ngram_count ,len (seen_ngrams ))

                if min_size >0 :
                    quick_similarity =intersection_size /min_size 
                    if quick_similarity <0.3 :
                        continue 


                try :
                    similarity_checks +=1 
                    union_size =len (current_ngrams |seen_ngrams )
                    if union_size >0 :
                        similarity =intersection_size /union_size 
                        if similarity >=threshold :
                            is_duplicate =True 
                            break 
                except Exception as e :
                    continue 

            if not is_duplicate :
                kept_indices .append (i )
                seen_ngram_sets .append (current_ngrams )
                seen_text_hashes .add (text_hash )


            if i %100 ==0 and i >0 :
                elapsed =time .time ()-start_time 
                speed =(i +1 )/elapsed if elapsed >0 else 0 
                retention_rate =len (kept_indices )/(i +1 )*100 if i >0 else 100 
                pbar .set_postfix_str (f"kept:{len(kept_indices)} ({retention_rate:.1f}%) speed:{speed:.1f}/s")


                if i >=check_interval and retention_rate >early_stop_threshold :
                    print (f"\nChunk {chunk_id}: limited dedup effect detected (retention rate {retention_rate:.1f}%), consider checking data quality")
            else :
                pbar .set_postfix_str (f"{len(kept_indices)}")
            pbar .update (1 )

        pbar .close ()

        processing_time =time .time ()-start_time 
        final_speed =len (chunk )/processing_time if processing_time >0 else 0 

        print (f"Chunk {chunk_id}: processing complete, kept {len(kept_indices)}/{len(chunk)} items [elapsed: {processing_time:.1f}s, avg speed: {final_speed:.1f} items/s]")

        if kept_indices :
            result =chunk .select (kept_indices )
            print (f"Chunk {chunk_id}: result Dataset created successfully, contains {len(result)} items")
            return result 
        else :
            print (f"Chunk {chunk_id}: all data deduplicated, returning empty result")
            return None 

    except Exception as e :
        chunk_id =chunk_data .get ('chunk_id','unknown')
        print (f"Chunk {chunk_id} processing failed: {str(e)}")
        import traceback 
        print (f"Chunk {chunk_id} error details: {traceback.format_exc()}")
        return None 

def load_local_dataset (file_path ,text_field ='text'):
    from datasets import Dataset 
    import json 
    import pandas as pd 
    import os 

    if not os .path .exists (file_path ):
        print (f"Error: file not found - {file_path}")
        return None 

    try :
        file_ext =os .path .splitext (file_path )[1 ].lower ()

        if file_ext =='.json':
            print (f"Loading JSON file: {file_path}")


            try :

                dataset =Dataset .from_json (file_path )
                print (f"Successfully loaded line-delimited JSON")
                return dataset 
            except Exception as e :
                print (f"Line-delimited JSON load failed, trying standard JSON format: {e}")

                try :

                    with open (file_path ,'r',encoding ='utf-8')as f :
                        data =json .load (f )


                        if isinstance (data ,list ):
                            dataset =Dataset .from_list (data )
                            print (f"Successfully loaded JSON array with {len(dataset)} records")
                    elif isinstance (data ,dict ):

                        if 'data'in data and isinstance (data ['data'],list ):
                            dataset =Dataset .from_list (data ['data'])
                            print (f"Successfully loaded JSON with 'data' field, containing {len(dataset)} records")
                        else :

                            dataset =Dataset .from_list ([data ])
                            print ("Successfully loaded single JSON object")
                    else :
                        print (f"Unsupported JSON format")
                        return None 

                    return dataset 
                except Exception as e2 :
                    print (f"JSON load failed: {e2}")
                    return None 

        elif file_ext =='.csv':
            print (f"Loading CSV file: {file_path}")
            df =pd .read_csv (file_path )
            dataset =Dataset .from_pandas (df )
            print (f"Successfully loaded CSV with {len(dataset)} records")
            return dataset 

        elif file_ext =='.parquet':
            print (f"Loading Parquet file: {file_path}")
            dataset =Dataset .from_parquet (file_path )
            print (f"Successfully loaded Parquet with {len(dataset)} records")
            return dataset 

        else :
            print (f"Unsupported file format: {file_ext}")
            return None 

    except Exception as e :
        print (f"Failed to load local dataset: {e}")
        import traceback 
        print (f"Error details: {traceback.format_exc()}")
        return None 

class DatasetCleaner :

    def __init__ (self ,threshold =0.8 ,ngram_size =3 ,enable_parallel =False ,num_workers =None ,enable_prefilter =True ):
        self .threshold =threshold 
        self .ngram_size =ngram_size 
        self .enable_parallel =enable_parallel 
        self .num_workers =num_workers or max (1 ,mp .cpu_count ()-1 )
        self .enable_prefilter =enable_prefilter 
        self .stats ={}

        print (f"Cleaner initialized:")
        print (f"  Parallel processing: {'Enabled' if enable_parallel else 'Disabled'}")
        if enable_parallel :
            print (f"  Worker processes: {self.num_workers}")
        print (f"  Prefilter optimization: {'Enabled' if enable_prefilter else 'Disabled'}")
        print (f"  Similarity threshold: {threshold}")
        print (f"  N-gram size: {ngram_size}")
        print ()

    def load_dataset_safe (self ,dataset_name ,config =None ,split =None ):
        try :
            if config :
                print (f"Loading dataset: {dataset_name} (config: {config})")
                if split :
                    dataset =load_dataset (dataset_name ,config ,split =split )
                else :
                    dataset =load_dataset (dataset_name ,config )
            else :
                print (f"Loading dataset: {dataset_name}")
                if split :
                    dataset =load_dataset (dataset_name ,split =split )
                else :
                    dataset =load_dataset (dataset_name )
            print (f"Dataset loaded successfully")
            return dataset 
        except Exception as e :
            print (f"Dataset load failed: {e}")
            return None 

    def analyze_dataset (self ,dataset ,text_field ='text'):
        print ("\nDataset analysis")
        print ("="*50 )

        if isinstance (dataset ,DatasetDict ):

            total_samples =0 
            for split_name ,split_data in dataset .items ():
                samples =len (split_data )
                total_samples +=samples 
                print (f"  {split_name}: {samples:,} items")


                if samples >0 :
                    sample_texts =[split_data [i ][text_field ]for i in range (min (100 ,samples ))]
                    avg_length =sum (len (text )for text in sample_texts )/len (sample_texts )
                    print (f"    Average text length: {avg_length:.0f} characters")


                    self ._check_data_format (split_data ,text_field ,5 )

            print (f"  Total: {total_samples:,} items")

        else :

            samples =len (dataset )
            print (f"  Sample count: {samples:,} items")


                if samples >0 :
                sample_texts =[dataset [i ][text_field ]for i in range (min (100 ,samples ))]
                avg_length =sum (len (text )for text in sample_texts )/len (sample_texts )
                    print (f"  Average text length: {avg_length:.0f} characters")


                self ._check_data_format (dataset ,text_field ,5 )

        print ()
        return dataset 

    def _check_data_format (self ,dataset ,text_field ,num_samples =5 ):
        print (f"  Data format check (first {num_samples} samples):")

        for i in range (min (num_samples ,len (dataset ))):
            try :
                item =dataset [i ]
                print (f"    [{i}] Type: {type(item)}")

                if isinstance (item ,dict ):
                    print (f"        Fields: {list(item.keys())}")
                    if text_field in item :
                        text =item [text_field ]
                        print (f"        Text type: {type(text)}")
                        if isinstance (text ,str ):
                            print (f"        Text length: {len(text)}")
                            print (f"        Text preview: {repr(text[:50])}...")
                        else :
                            print (f"        Text content: {text}")
                    else :
                        print (f"        Missing field '{text_field}'")
                else :
                    print (f"        Data: {item}")

            except Exception as e :
                print (f"    [{i}] access failed: {e}")

    def quick_filter (self ,text1 ,text2 ):
        if not self .enable_prefilter :
            return True 


        len_ratio =len (text1 )/len (text2 )if len (text2 )>0 else float ('inf')
        if len_ratio >2.0 or len_ratio <0.5 :
            return False 


        ngrams1 =set (get_ngrams (normalize_text (text1 ),self .ngram_size ))
        ngrams2 =set (get_ngrams (normalize_text (text2 ),self .ngram_size ))

        size_ratio =len (ngrams1 )/len (ngrams2 )if len (ngrams2 )>0 else float ('inf')
        if size_ratio >1.5 or size_ratio <0.67 :
            return False 


        intersection_size =len (ngrams1 &ngrams2 )
        min_size =min (len (ngrams1 ),len (ngrams2 ))
        if min_size >0 and intersection_size <min_size *0.3 :
            return False 

        return True 

    def enhanced_jaccard_similarity (self ,text1 ,text2 ):

        if not self .quick_filter (text1 ,text2 ):
            return 0.0 


        return jaccard_similarity (text1 ,text2 ,self .ngram_size )

    def parallel_deduplicate_chunks (self ,dataset ,text_field ):
        original_size =len (dataset )
        print (f"Starting parallel chunk dedup, original items: {original_size}")


        if original_size <1000 :
            chunk_size =max (100 ,original_size //max (2 ,self .num_workers ))
        elif original_size <10000 :
            chunk_size =max (200 ,original_size //self .num_workers )
        else :

            chunk_size =max (500 ,min (2000 ,original_size //(self .num_workers *2 )))

        print (f"Using chunk size: {chunk_size} items/chunk")

        chunks =[]

        for i in range (0 ,original_size ,chunk_size ):
            end_idx =min (i +chunk_size ,original_size )


            chunk_items =[]
            try :
                print (f"Preparing chunk {i}-{end_idx}...")


                for j in range (i ,end_idx ):
                    try :
                        item =dataset [j ]

                        if isinstance (item ,dict )and text_field in item :

                            text_content =item [text_field ]
                            if text_content and isinstance (text_content ,str )and len (text_content .strip ())>0 :
                                chunk_items .append (item )
                            else :
                                print (f"Warning: index {j} text empty or invalid")
                        else :
                            print (f"Warning: index {j} invalid format or missing field '{text_field}'")
                    except Exception as e :
                        print (f"Warning: index {j} data access failed: {e}")
                        continue 

                print (f"Chunk {i}-{end_idx}: valid data {len(chunk_items)} items")

                if chunk_items :
                    chunk_data ={
                    'chunk_items':chunk_items ,
                    'params':(text_field ,self .threshold ,self .ngram_size ),
                    'chunk_id':len (chunks ),
                    'original_range':f"{i}-{end_idx}"
                    }
                    chunks .append (chunk_data )
                else :
                    print (f"Warning: chunk {i}-{end_idx} has no valid data")

            except Exception as e :
                print (f"Chunk {i}-{end_idx} data preparation failed: {e}")

                try :
                    simple_items =[]
                    chunk_slice =dataset [i :end_idx ]
                    for idx ,item in enumerate (chunk_slice ):
                        if hasattr (item ,'get')and item .get (text_field ):
                            simple_items .append (dict (item ))
                        elif isinstance (item ,dict )and text_field in item :
                            simple_items .append (item )

                    if simple_items :
                        chunk_data ={
                        'chunk_items':simple_items ,
                        'params':(text_field ,self .threshold ,self .ngram_size ),
                        'chunk_id':len (chunks ),
                        'original_range':f"{i}-{end_idx}"
                        }
                        chunks .append (chunk_data )
                        print (f"Chunk {i}-{end_idx}: obtained {len(simple_items)} items using fallback method")
                except Exception as e2 :
                    print (f"Chunk {i}-{end_idx} fallback method also failed: {e2}")
                    continue 

        if not chunks :
            print ("No valid chunks found, falling back to single-thread processing")
            return deduplicate_dataset_jaccard (dataset ,text_field ,self .threshold ,self .ngram_size )

        print (f"Successfully created {len(chunks)} valid chunks")

        try :

            cleaned_chunks =[]

            if self .num_workers ==1 or len (chunks )==1 :

                for i ,chunk_data in enumerate (chunks ):
                    print (f"Processing chunk {i+1}/{len(chunks)} (range: {chunk_data['original_range']})")
                    result =process_chunk_worker (chunk_data )
                    if result and len (result )>0 :
                        cleaned_chunks .append (result )
                        print (f"Chunk {i+1} complete: kept {len(result)} items")
                    else :
                        print (f"Chunk {i+1} result empty")
            else :

                print (f"Starting {min(self.num_workers, len(chunks))} threads for parallel processing...")


                overall_pbar =tqdm (
                total =len (chunks ),
                desc ="Overall progress",
                position =len (chunks ),
                bar_format ='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] completed chunks'
                )

                with ThreadPoolExecutor (max_workers =min (self .num_workers ,len (chunks )))as executor :
                    '''
                    executor.submit() submits a function to the thread pool for asynchronous execution
                    Returns a Future object representing a task that is running or will run
                    process_chunk_worker is the worker function to execute
                    chunk_data is the argument passed to the worker function
                    '''
                    future_to_chunk ={executor .submit (process_chunk_worker ,chunk_data ):i 
                    for i ,chunk_data in enumerate (chunks )}

                    for future in future_to_chunk :
                        try :
                            result =future .result (timeout =600 )
                            chunk_idx =future_to_chunk [future ]
                            if result and len (result )>0 :
                                cleaned_chunks .append (result )
                                print (f"Completed chunk {chunk_idx+1}/{len(chunks)}: kept {len(result)} items")
                            else :
                                print (f"Chunk {chunk_idx+1} result empty")
                            overall_pbar .update (1 )
                        except Exception as e :
                            chunk_idx =future_to_chunk [future ]
                            print (f"Chunk {chunk_idx+1} processing exception: {e}")
                            overall_pbar .update (1 )

                overall_pbar .close ()


            if not cleaned_chunks :
                print ("All parallel chunks failed, falling back to single-thread processing")
                return deduplicate_dataset_jaccard (dataset ,text_field ,self .threshold ,self .ngram_size )

            print (f"Successfully processed {len(cleaned_chunks)}/{len(chunks)} chunks")


            merged_dataset =concatenate_datasets (cleaned_chunks )
            merged_size =len (merged_dataset )
            print (f"Merged data size: {merged_size} items")

            if merged_size ==0 :
                print ("Merged result empty, falling back to single-thread processing")
                return deduplicate_dataset_jaccard (dataset ,text_field ,self .threshold ,self .ngram_size )


            reduction_ratio =merged_size /original_size 
            print (f"Data reduction ratio: {(1-reduction_ratio)*100:.1f}%")

            if reduction_ratio <0.2 :
                print ("Performing cross-chunk dedup...")
                final_dataset =deduplicate_dataset_jaccard (
                merged_dataset ,text_field ,self .threshold ,self .ngram_size 
                )
                print (f"Final data: {len(final_dataset)} items")
                return final_dataset 
            else :
                print ("Skipping cross-chunk dedup (reduction not significant)")
                return merged_dataset 

        except Exception as e :
            print (f"Parallel processing overall failed: {e}")
            print ("Falling back to single-thread processing...")
            return deduplicate_dataset_jaccard (dataset ,text_field ,self .threshold ,self .ngram_size )

    def clean_single_split (self ,dataset ,text_field ='text',mode ='standard'):
        print (f"Starting dataset cleaning (mode: {mode})")
        print (f"Parameters: threshold={self.threshold}, n-gram={self.ngram_size}")
        if self .enable_parallel :
            print (f"Parallel processing: {self.num_workers} worker processes")

        start_time =time .time ()
        original_size =len (dataset )


        if original_size >200000 and mode =='standard':
            print (f"Dataset is large ({original_size} items), consider using fast mode")
            suggest_fast =input ("Switch to fast mode? (y/n, default y): ").strip ().lower ()!='n'
            if suggest_fast :
                mode ='fast'

        if mode =='standard':
            if self .enable_parallel and original_size >1000 :
                try :

                    cleaned_dataset =self .parallel_deduplicate_chunks (dataset ,text_field )
                except Exception as e :
                    print (f"Parallel processing failed: {e}")
                    print ("Falling back to single-thread standard processing...")
                    cleaned_dataset =deduplicate_dataset_jaccard (
                    dataset ,text_field ,self .threshold ,self .ngram_size 
                    )
            else :

                cleaned_dataset =deduplicate_dataset_jaccard (
                dataset ,text_field ,self .threshold ,self .ngram_size 
                )
        elif mode =='fast':

            sample_size =min (1000 ,original_size //10 )
            print (f"Fast mode: using {sample_size} samples for dedup")
            cleaned_dataset =quick_jaccard_deduplicate (
            dataset ,text_field ,self .threshold ,self .ngram_size ,sample_size =sample_size 
            )
        elif mode =='parallel':
            try :

                cleaned_dataset =self .parallel_deduplicate_chunks (dataset ,text_field )
            except Exception as e :
                print (f"Forced parallel mode failed: {e}")
                print ("Falling back to single-thread processing...")
                cleaned_dataset =deduplicate_dataset_jaccard (
                dataset ,text_field ,self .threshold ,self .ngram_size 
                )
        else :
            raise ValueError (f"Unknown cleaning mode: {mode}")

        end_time =time .time ()
        processing_time =end_time -start_time 


        cleaned_size =len (cleaned_dataset )
        removed_count =original_size -cleaned_size 
        removal_rate =(removed_count /original_size )*100 if original_size >0 else 0 


        speed =original_size /processing_time if processing_time >0 else 0 

        self .stats ={
        'original_size':original_size ,
        'cleaned_size':cleaned_size ,
        'removed_count':removed_count ,
        'removal_rate':removal_rate ,
        'processing_time':processing_time ,
        'processing_speed':speed 
        }

        print (f"\nCleaning complete")
        print (f"Processing time: {processing_time:.2f} seconds")
        print (f"Processing speed: {speed:.0f} items/sec")
        print (f"Original items: {original_size:,}")
        print (f"After cleaning: {cleaned_size:,}")
        print (f"Removed items: {removed_count:,} ({removal_rate:.1f}%)")

        return cleaned_dataset 

    def clean_multi_split (self ,dataset_dict ,text_field ='text'):
        print ("Starting cross-split cleaning")
        print (f"Parameters: threshold={self.threshold}, n-gram={self.ngram_size}")

        start_time =time .time ()


        cleaned_dataset =deduplicate_cross_splits_jaccard (
        dataset_dict ,text_field ,self .threshold ,self .ngram_size 
        )

        end_time =time .time ()
        processing_time =end_time -start_time 


        print (f"\nCross-split cleaning complete")
        print (f"Processing time: {processing_time:.2f} seconds")

        total_original =0 
        total_cleaned =0 

        for split_name in dataset_dict .keys ():
            if split_name in cleaned_dataset :
                original =len (dataset_dict [split_name ])
                cleaned =len (cleaned_dataset [split_name ])
                removed =original -cleaned 
                removal_rate =(removed /original )*100 if original >0 else 0 

                total_original +=original 
                total_cleaned +=cleaned 

                print (f"  {split_name}: {original:,} -> {cleaned:,} items (reduction {removal_rate:.1f}%)")

        overall_removal_rate =((total_original -total_cleaned )/total_original )*100 if total_original >0 else 0 
        print (f"  Total: {total_original:,} -> {total_cleaned:,} items (reduction {overall_removal_rate:.1f}%)")

        return cleaned_dataset 

    def save_cleaned_dataset (self ,dataset ,output_path ,format ='json'):
        try :
            print (f"Saving dataset to: {os.path.abspath(output_path)}")
            print (f"Dataset size: {len(dataset)} items")

            if format =='json':
                dataset .to_json (output_path )
            elif format =='csv':
                dataset .to_csv (output_path )
            elif format =='parquet':
                dataset .to_parquet (output_path )
            else :
                raise ValueError (f"Unsupported format: {format}")


            if os .path .exists (output_path ):
                file_size =os .path .getsize (output_path )
                print (f"Dataset saved successfully")
                print (f"   File size: {file_size:,} bytes")
                print (f"   File path: {os.path.abspath(output_path)}")
            else :
                print ("Warning: file may not have been created successfully")

        except Exception as e :
            print (f"Save failed: {e}")
            import traceback 
            print (f"Error details: {traceback.format_exc()}")

    def sample_comparison (self ,original_dataset ,cleaned_dataset ,text_field ='text',num_samples =5 ):
        print (f"\nSample comparison (showing first {num_samples} samples)")
        print ("="*80 )


        original_texts =[original_dataset [i ][text_field ]for i in range (len (original_dataset ))]
        cleaned_texts =[cleaned_dataset [i ][text_field ]for i in range (len (cleaned_dataset ))]

        print ("Retained samples:")
        for i ,text in enumerate (cleaned_texts [:num_samples ]):
            print (f"{i+1}. {text[:100]}...")


        removed_samples =[]
        for text in original_texts [:50 ]:
            if text not in cleaned_texts :
                removed_samples .append (text )
                if len (removed_samples )>=num_samples :
                    break 

        if removed_samples :
            print (f"\nRemoved samples:")
            for i ,text in enumerate (removed_samples ):
                print (f"{i+1}. {text[:100]}...")

    def get_cleaning_stats (self ):
        return self .stats 

def show_dataset_info ():
    print ("\nCommon dataset configuration info:")
    print ("-"*40 )
    print ("Recommended datasets for classification tasks (raw, unprocessed):")
    print ("="*40 )
    print ("AG News (ag_news):")
    print ("  - Use: news topic classification (4 classes)")
    print ("  - Characteristics: raw news text, ~120K training samples")
    print ("  - Fields: 'text' (content), 'label' (0-3)")
    print ("  - Labels: World, Sports, Business, Sci/Tech")
    print ("  - Notes: Unprocessed dataset, suitable for deduplication testing")
    print ()
    print ("Yelp Reviews Full (yelp_review_full):")
    print ("  - Use: sentiment analysis (5-star rating)")
    print ("  - Characteristics: user reviews, ~650K training samples")
    print ("  - Fields: 'text' (review), 'label' (0-4 mapping to 1-5 stars)")
    print ("  - Notes: includes colloquial language, useful for deduplication testing")
    print ()
    print ("Other datasets:")
    print ("="*40 )
    print ("Go Emotions (google-research-datasets/go_emotions):")
    print ("  - 'simplified': simplified version with basic emotion labels")
    print ("  - 'raw': raw version with detailed emotion categories")
    print ("\nAmazon Reviews (amazon_reviews_multi):")
    print ("  - 'en': English reviews")
    print ("  - 'fr': French reviews")
    print ("  - 'de': German reviews")
    print ("\nCommon Voice (mozilla-foundation/common_voice_11_0):")
    print ("  - 'en': English")
    print ("  - 'zh-CN': Chinese")
    print ("  - 'fr': French")
    print ("-"*40 )

def interactive_cleaning ():
    print ("Interactive dataset cleaning tool")
    print ("="*50 )


    show_info =input ("Show common dataset configuration info? (y/n, default n): ").strip ().lower ()=='y'
    if show_info :
        show_dataset_info ()


    print ("\nSelect dataset to clean:")
    print ("Recommended for classification tasks (raw data):")
    print ("1. AG News - news topic classification (4 classes, 120K samples)")
    print ("2. Yelp Reviews - sentiment analysis (5-star, ~650K samples)")
    print ()
    print ("Other common datasets:")
    print ("3. IMDB movie reviews dataset")
    print ("4. Go Emotions (simplified)")
    print ("5. Go Emotions (raw)")
    print ("6. Custom dataset (Hugging Face)")
    print ("7. Local dataset file (JSON/CSV/Parquet)")

    choice =input ("Please choose (1-7): ").strip ()

    if choice =='1':
        dataset_name ="ag_news"
        dataset_config =None 
        text_field ='text'
        print ("AG News - news topic classification dataset")
        print ("Contains 4 classes: World, Sports, Business, Sci/Tech")
        print ("120,000 training samples + 7,600 test samples")
    elif choice =='2':
        dataset_name ="yelp_review_full"
        dataset_config =None 
        text_field ='text'
        print ("Yelp Reviews - 5-star sentiment dataset")
        print ("Contains 1-5 star user reviews, suitable for sentiment intensity analysis")
        print ("650,000 training samples + 50,000 test samples")
    elif choice =='3':
        dataset_name ="stanfordnlp/imdb"
        dataset_config =None 
        text_field ='text'
    elif choice =='4':
        dataset_name ="google-research-datasets/go_emotions"
        dataset_config ="simplified"
        text_field ='text'
    elif choice =='5':
        dataset_name ="google-research-datasets/go_emotions"
        dataset_config ="raw"
        text_field ='text'
    elif choice =='6':
        dataset_name =input ("Enter dataset name: ").strip ()


        has_config =input ("Does the dataset have a config? (y/n, default n): ").strip ().lower ()=='y'
        if has_config :
            dataset_config =input ("Enter config name (e.g., 'simplified', 'original'): ").strip ()
            if not dataset_config :
                dataset_config =None 
        else :
            dataset_config =None 

        text_field =input ("Enter text field name (default 'text'): ").strip ()or 'text'


        load_mode ='huggingface'
    elif choice =='7':

        print ("\nLocal dataset loading")
        print ("-"*40 )


        default_dir =r"D:\Deduplication_framework\text\dataset"
        print (f"Default dataset directory: {default_dir}")


        use_default =input ("Use default directory? (y/n, default y): ").strip ().lower ()!='n'

        if use_default :

            import os 
            try :
                files =[f for f in os .listdir (default_dir )
                if os .path .isfile (os .path .join (default_dir ,f ))and 
                f .endswith (('.json','.csv','.parquet'))]

                if files :
                    print ("\nAvailable dataset files:")
                    for i ,file in enumerate (files ,1 ):
                        print (f"{i}. {file}")

                    file_choice =input ("Choose file number (or enter full filename): ").strip ()

                    try :
                        file_idx =int (file_choice )-1 
                        if 0 <=file_idx <len (files ):
                            file_path =os .path .join (default_dir ,files [file_idx ])
                        else :
                            print ("Invalid file number, please enter file path")
                            file_path =input ("Enter full file path: ").strip ()
                    except ValueError :

                        if file_choice in files :
                            file_path =os .path .join (default_dir ,file_choice )
                        else :
                            file_path =input ("File not found, please enter full path: ").strip ()
                else :
                    print (f"No JSON/CSV/Parquet files found in default directory")
                    file_path =input ("Enter full file path: ").strip ()
            except Exception as e :
                print (f"Failed to read default directory: {e}")
                file_path =input ("Enter full file path: ").strip ()
        else :
            file_path =input ("Enter full file path: ").strip ()


        text_field =input ("Enter text field name (default 'text'): ").strip ()or 'text'


        load_mode ='local'
        dataset_name =file_path 
        dataset_config =None 
    else :
        print ("Invalid choice")
        return 


    print (f"\nConfigure cleaning parameters:")
    threshold =float (input ("Similarity threshold (0.0-1.0, default 0.8): ")or "0.8")
    ngram_size =int (input ("n-gram size (default 3): ")or "3")


    print (f"\nOptimization options:")
    enable_parallel =input ("Enable parallel processing? (y/n, default y): ").strip ().lower ()!='n'
    enable_prefilter =input ("Enable prefilter optimization? (y/n, default y): ").strip ().lower ()!='n'


    print (f"\nSave file settings:")


    if choice =='7':
        import os 
        base_filename =os .path .splitext (os .path .basename (file_path ))[0 ]
        default_output =f"{base_filename}_clean_t{threshold}.json"
    else :
        default_output =f"{dataset_name.split('/')[-1]}_clean_t{threshold}.json"

    output_path =input (f"Output file path (default '{default_output}'): ").strip ()or default_output 


    output_dir =os .path .dirname (output_path )if os .path .dirname (output_path )else "."
    os .makedirs (output_dir ,exist_ok =True )
    print (f"Output directory: {os.path.abspath(output_dir)}")


    if not output_path .endswith ('.json'):
        output_path +='.json'
    print (f"Full output path: {os.path.abspath(output_path)}")

    if enable_parallel :
        max_workers =mp .cpu_count ()
        num_workers =int (input (f"Number of parallel worker processes (1-{max_workers}, default {max_workers-1}): ")or str (max_workers -1 ))
        num_workers =max (1 ,min (num_workers ,max_workers ))
    else :
        num_workers =1 


    print (f"\nChoose cleaning mode:")
    print ("1. Standard mode (auto select serial/parallel)")
    print ("2. Fast mode (sampling optimization)")
    print ("3. Cross-split mode (recommended)")
    print ("4. Forced parallel mode (for large datasets)")

    mode_choice =input ("Please choose (1/2/3/4): ").strip ()


    cleaner =DatasetCleaner (
    threshold =threshold ,
    ngram_size =ngram_size ,
    enable_parallel =enable_parallel ,
    num_workers =num_workers ,
    enable_prefilter =enable_prefilter 
    )


    if choice =='7'or load_mode =='local':
        print (f"\nLoading local dataset: {dataset_name}")
        dataset =load_local_dataset (dataset_name ,text_field )
    else :

        dataset =None 
        retry_count =0 
        max_retries =3 

        while dataset is None and retry_count <max_retries :
            dataset =cleaner .load_dataset_safe (dataset_name ,config =dataset_config )

                if dataset is None :
                retry_count +=1 
                if retry_count <max_retries :
                    print (f"\nLoad failed, retry {retry_count}/{max_retries}")
                    print ("Please check dataset name and config")


                    action =input ("1. Edit parameters  2. Try loading local file  3. Exit  Choose(1/2/3): ").strip ()

                    if action =='1':
                        new_dataset_name =input (f"Dataset name (current: {dataset_name}): ").strip ()
                        if new_dataset_name :
                            dataset_name =new_dataset_name 

                        if dataset_config :
                            new_config =input (f"Config name (current: {dataset_config}, empty for none): ").strip ()
                            dataset_config =new_config if new_config else None 
                        else :
                            has_config =input ("Add a config? (y/n): ").strip ().lower ()=='y'
                            if has_config :
                                dataset_config =input ("Enter config name: ").strip ()

                    elif action =='2':

                        print ("\nAttempting to load local file...")
                        file_path =input ("Enter local file path: ").strip ()
                        dataset =load_local_dataset (file_path ,text_field )
                        break 
                    else :
                        print ("Exiting program")
                        return 
                else :
                    print ("Reached max retries, attempting to load local file...")
                    file_path =input ("Enter local file path (press Enter to exit): ").strip ()
                    if file_path :
                        dataset =load_local_dataset (file_path ,text_field )
                    else :
                        return 

    if dataset is None :
        print ("Dataset load failed, exiting")
        return 


    cleaner .analyze_dataset (dataset ,text_field )


    if mode_choice =='1':

        if isinstance (dataset ,DatasetDict )and 'train'in dataset :
            cleaned_data =cleaner .clean_single_split (dataset ['train'],text_field ,'standard')
        else :
            cleaned_data =cleaner .clean_single_split (dataset ,text_field ,'standard')

    elif mode_choice =='2':

        if isinstance (dataset ,DatasetDict )and 'train'in dataset :
            cleaned_data =cleaner .clean_single_split (dataset ['train'],text_field ,'fast')
        else :
            cleaned_data =cleaner .clean_single_split (dataset ,text_field ,'fast')

    elif mode_choice =='3':

        if isinstance (dataset ,DatasetDict ):
            cleaned_data =cleaner .clean_multi_split (dataset ,text_field )
        else :
            print ("Single-split dataset, using standard mode")
            cleaned_data =cleaner .clean_single_split (dataset ,text_field ,'standard')

    elif mode_choice =='4':

        if isinstance (dataset ,DatasetDict )and 'train'in dataset :
            cleaned_data =cleaner .clean_single_split (dataset ['train'],text_field ,'parallel')
        else :
            cleaned_data =cleaner .clean_single_split (dataset ,text_field ,'parallel')
    else :
        print ("Invalid selection")
        return 


    if not isinstance (cleaned_data ,DatasetDict ):
        original_for_comparison =dataset ['train']if isinstance (dataset ,DatasetDict )and 'train'in dataset else dataset 
        cleaner .sample_comparison (original_for_comparison ,cleaned_data ,text_field )


    print (f"\nStarting to save cleaned results...")
    if isinstance (cleaned_data ,DatasetDict ):

        print ("Detected multi-split dataset, saving each split...")
        for split_name ,split_data in cleaned_data .items ():

            base_name =os .path .splitext (output_path )[0 ]
            extension =os .path .splitext (output_path )[1 ]or '.json'
            split_path =f"{base_name}_{split_name}{extension}"

            print (f"Saving {split_name} split to: {split_path}")
            cleaner .save_cleaned_dataset (split_data ,split_path )
    else :

        print (f"Saving dataset to: {output_path}")
        cleaner .save_cleaned_dataset (cleaned_data ,output_path )

    print (f"\nDataset cleaning complete!")

def batch_cleaning_example ():
    print ("Batch cleaning example - performance comparison")
    print ("="*50 )


    print ("Loading IMDB dataset...")
    dataset =load_dataset ("stanfordnlp/imdb")
    if dataset is None :
        return 

    print ("\nDataset info:")
    print (f"  Train: {len(dataset['train']):,} samples")
    print (f"  Test: {len(dataset['test']):,} samples")


    print ("\n=== Performance comparison tests ===")


    print ("\n1. Traditional method test (first 1000):")
    small_dataset =dataset ['train'].select (range (1000 ))

    cleaner_traditional =DatasetCleaner (
    threshold =0.8 ,ngram_size =3 ,
    enable_parallel =False ,enable_prefilter =False 
    )
    start_time =time .time ()
    cleaned_traditional =cleaner_traditional .clean_single_split (small_dataset ,'text','standard')
    traditional_time =time .time ()-start_time 


    print ("\n2. Optimized method test (first 1000):")
    cleaner_optimized =DatasetCleaner (
    threshold =0.8 ,ngram_size =3 ,
    enable_parallel =True ,enable_prefilter =True 
    )
    start_time =time .time ()
    cleaned_optimized =cleaner_optimized .clean_single_split (small_dataset ,'text','standard')
    optimized_time =time .time ()-start_time 


    print ("\n3. Full dataset test (optimized method only):")
    cleaner_full =DatasetCleaner (
    threshold =0.8 ,ngram_size =3 ,
    enable_parallel =True ,enable_prefilter =True 
    )


    cleaned_dataset =cleaner_full .clean_multi_split (dataset ,'text')


    print ("\n"+"="*60 )
    print ("Performance comparison summary:")
    print (f"Traditional (1K samples): {traditional_time:.2f}s")
    print (f"Optimized (1K samples): {optimized_time:.2f}s")
    if traditional_time >0 :
        speedup =traditional_time /optimized_time 
        print (f"Speedup: {speedup:.1f}x")

    print (f"\nFinal cleaning results:")
    print (f"Train: {len(dataset['train']):,} -> {len(cleaned_dataset['train']):,} samples")
    print (f"Test: {len(dataset['test']):,} -> {len(cleaned_dataset['test']):,} samples")

    total_original =len (dataset ['train'])+len (dataset ['test'])
    total_cleaned =len (cleaned_dataset ['train'])+len (cleaned_dataset ['test'])
    reduction_rate =((total_original -total_cleaned )/total_original )*100 
    print (f"Overall data reduction: {reduction_rate:.1f}%")

    return cleaned_dataset 

if __name__ =="__main__":

    switcher =EnvManager ()
    res =switcher .setup_text_env ()
    if res :
        mp .set_start_method ('spawn',force =True )

        print ("Dataset cleaning tool")
        print ("Smart deduplication using Jaccard similarity")
        print ("="*60 )

        print ("\nChoose run mode:")
        print ("1. Interactive cleaning")
        print ("2. Batch cleaning example")
        print ("3. Show dataset info")

        mode =input ("Please choose (1/2/3): ").strip ()

        if mode =='1':
            interactive_cleaning ()
        elif mode =='2':
            batch_cleaning_example ()
        elif mode =='3':
            show_dataset_info ()
            print ("\n")

            choice =input ("Continue to interactive cleaning? (y/n): ").strip ().lower ()
            if choice =='y':
                interactive_cleaning ()
        else :
            print ("Invalid choice, running interactive cleaning...")
            interactive_cleaning ()
    else :
        print ("Environment setup failed, exiting")