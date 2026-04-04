import os
import glob
import torch
import pandas as pd
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# --- Configuration ---
INPUT_DIR = "data/text/wikivoyager"
OUTPUT_DIR = "data/embeddings/wikivoyager"
MODEL_ID = "jinaai/jina-embeddings-v3"
EMBEDDING_DIM = 128   # Target Matryoshka dimension
BATCH_SIZE = 8        # Safe batch size for chunked processing
MAX_SEQ_LENGTH = 4096 # Hard limit per chunk to prevent VRAM spikes

os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(text):
    """
    Removes raw newlines and replaces old [SEP] tokens with spaces.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.replace("[SEP]", " ")
    cleaned = re.sub(r'\s+', ' ', text)
    return cleaned.strip()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_ID} on {device}...")
    
    # Initialize Jina v3
    model = SentenceTransformer(
        MODEL_ID, 
        trust_remote_code=True, 
        model_kwargs={"torch_dtype": torch.float16}
    )
    model.to(device)
    model.eval()
    
    # Enforce sequence length limit on the model
    model.max_seq_length = MAX_SEQ_LENGTH

    csv_files = glob.glob(os.path.join(INPUT_DIR, "*_h3_*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in '{INPUT_DIR}'. Please check your paths.")
        return

    print(f"Found {len(csv_files)} files to process.")

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        print(f"\nProcessing: {filename}")
        
        df = pd.read_csv(file_path)
        if df.empty or 'text_content' not in df.columns:
            print("  -> Empty or missing 'text_content' column. Skipping.")
            continue
            
        df['text_content'] = df['text_content'].apply(clean_text)
        
        h3_cols = [col for col in df.columns if col.startswith('h3res')]
        if not h3_cols:
            print("  -> No H3 columns found. Skipping.")
            continue
            
        finest_h3_col = sorted(h3_cols, key=lambda x: int(x.replace('h3res', '')))[-1]
        
        texts = df['text_content'].tolist()
        h3_indices = df[finest_h3_col].tolist()
        
        all_final_embeddings = []
        
        # Batch processing
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding Hexes"):
            batch_texts = texts[i:i+BATCH_SIZE]
            
            all_chunks = []
            chunk_to_text_idx = []
            
            # 1. Tokenize and Chunk
            for idx, text in enumerate(batch_texts):
                # Get tokens without special tokens to measure length
                tokens = model.tokenizer(text, add_special_tokens=False, truncation=False)['input_ids']
                
                if len(tokens) <= MAX_SEQ_LENGTH:
                    all_chunks.append(text)
                    chunk_to_text_idx.append(idx)
                else:
                    # Split into strictly sized chunks
                    for j in range(0, len(tokens), MAX_SEQ_LENGTH):
                        chunk_tokens = tokens[j:j+MAX_SEQ_LENGTH]
                        chunk_text = model.tokenizer.decode(chunk_tokens)
                        all_chunks.append(chunk_text)
                        chunk_to_text_idx.append(idx)
            
            # 2. Encode all chunks
            with torch.no_grad():
                chunk_embeddings = model.encode(
                    all_chunks, 
                    batch_size=BATCH_SIZE, # Prevents VRAM spike if a text spawned many chunks
                    task="retrieval.passage",
                    show_progress_bar=False,
                    convert_to_tensor=True,
                    normalize_embeddings=False
                )
            
            # 3. Average the chunks back into their original text representations
            aggregated_embeddings = []
            for idx in range(len(batch_texts)):
                # Find all chunk embeddings that belong to this specific text
                indices = [k for k, x in enumerate(chunk_to_text_idx) if x == idx]
                embs_for_text = chunk_embeddings[indices]
                
                # Mean pooling across chunks
                avg_emb = torch.mean(embs_for_text, dim=0)
                aggregated_embeddings.append(avg_emb)
                
            batch_embeddings = torch.stack(aggregated_embeddings)
            
            # 4. Matryoshka Slicing
            batch_embeddings_128 = batch_embeddings[:, :EMBEDDING_DIM]
            
            # 5. L2 Normalization
            normalized = torch.nn.functional.normalize(batch_embeddings_128, p=2, dim=1)
            
            # Move to CPU and store
            all_final_embeddings.extend(normalized.cpu().float().numpy().tolist())
            
            # 6. Aggressive Memory Management
            del chunk_embeddings
            del aggregated_embeddings
            del batch_embeddings
            del batch_embeddings_128
            del normalized
            torch.cuda.empty_cache() 
            
        out_df = pd.DataFrame({
            'h3_index': h3_indices,
            'embedding': all_final_embeddings
        })
        
        out_filename = filename.replace('.csv', '_embeddings.parquet')
        out_path = os.path.join(OUTPUT_DIR, out_filename)
        
        out_df.to_parquet(out_path, engine='pyarrow', index=False)
        print(f"  -> Saved {len(out_df)} Matryoshka (128-dim) embeddings to {out_path}")

    print("\nAll files embedded successfully!")

if __name__ == "__main__":
    main()