## Run code

Train model on suppkg:

    python main.py --data suppkg --text_embedding_file pubmedbert_pretrained_embeddings_768.npy --knowledge_embedding_file poincare_embeddings.npy --num_hidden_layers 2 --iterations 40000 --evaluate_every 1000 --validate_every 2000 --neg_sample_size_eval 100 --w 0.75 --model_state_file suppkg_model_state.pth


The text embedding file `pubmedbert_embeddings_768.npy` and domain knowledge embedding file `poincare_embeddings.npy` can be obtained from https://drive.google.com/drive/folders/1aIsdgX7IUqMl4Wn3TFEU-1iAU0kVfBpu?usp=sharing. Download them and put them in the directory `suppkg` before running the model.
