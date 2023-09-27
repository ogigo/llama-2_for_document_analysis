class CFG:
    # LLMs
    model_name = 'bloom' # wizardlm, bloom, falcon, llama2-7b, llama2-13b
    temperature = 0,
    top_p = 0.95,
    repetition_penalty = 1.15

    # splitting
    split_chunk_size = 800
    split_overlap = 0

    # embeddings
    embeddings_model_repo = 'sentence-transformers/all-MiniLM-L6-v2'

    # similar passages
    k = 3

    # paths
    PDFs_path = '/HP books/'
    Embeddings_path =  ''
    Persist_directory = './harry-potter-vectordb'