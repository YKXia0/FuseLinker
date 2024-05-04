## Run code

    python main.py --data suppkg --text_embedding_file pubmedbert_embeddings_768.npy --knowledge_embedding_file poincare_embeddings.npy --num_hidden_layers 2 --iterations 40000 --evaluate_every 1000 --neg_sample_size_eval 100 --w 0.25

The text embedding file `pubmedbert_embeddings_768.npy` and domain knowledge embedding file `poincare_embeddings.npy` can be obtained from https://drive.google.com/drive/folders/1aIsdgX7IUqMl4Wn3TFEU-1iAU0kVfBpu?usp=sharing. Download them and put them in the directory `suppkg` as `suppkg/pubmedbert_embeddings_768.npy` and `suppkg/poincare_embeddings.npy`.
