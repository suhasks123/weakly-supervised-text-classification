import numpy as np
np.random.seed(1234)
from time import time
import os
import json

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
from model import WSTC, f1, accuracy
from keras.optimizers import SGD
from gen import augment, pseudodocs
from load_data import load_dataset
from gensim.models import word2vec

# Importing the PyTorch Bert Model and related utilities
import torch
from transformers import BertTokenizer
from transformers import BertModel
import pickle

'''
This function is used to set the optional arguments for the
command-line arguments parser. These arguments are used to configure
the functioning of the application. The parsing is handled by `argparse`
'''
def arguments_parser(parser):
    set_parser_arguments_basic(parser)
    set_parser_arguments_training(parser)
    set_parser_arguments_hyperparameters(parser)
    set_parser_arguments_misc(parser)

    # Parse the arguments
    args = parser.parse_args()
    print(args)

    return args

# Basic configuration settings
def set_parser_arguments_basic(parser):
    parser.add_argument('--language_model', '-l', default='w2v', choices=['w2v', 'bert']) # Selecting the language model
    parser.add_argument('--model', '-m', default='cnn', choices=['cnn']) # Selecting proper neural network model to use
    parser.add_argument('--sup_source', '-s', default='labels', choices=['labels', 'keywords', 'docs']) # Selecting the weak supervision source
    parser.add_argument('--dataset', '-d', default='agnews', choices=['agnews', 'yelp', 'imdb']) # Selecting appropriate dataset for training
    parser.add_argument('--with_evaluation', '-w', default='True', choices=['True', 'False']) # Selecting whether ground truth labels are available for evaluation

# Settings related to the training
def set_parser_arguments_training(parser):
    parser.add_argument('--batch_size', '-z', default=256, type=int) # Selecting the mini-batch size for both pre-training and self-training
    parser.add_argument('--pretrain_epochs', '-e', default=None, type=int) # Number of epochs in pre-training
    parser.add_argument('--update_interval', '-u', default=None, type=int) # Selecting self-training update interval
    parser.add_argument('--max_iter', '-i', default=5e3, type=int) # Upper limit of self-training iterations

# Settings related to hyperparameters
def set_parser_arguments_hyperparameters(parser):
    # Delta - The stopping criterion for self-training
    default_delta = 0.1
    parser.add_argument('--delta', '-c', default=default_delta, type=float) # Selecting the stopping criterion for self-training
    # Alpha - The background word distribution weight
    default_alpha = 0.2
    parser.add_argument('--alpha', '-a', default=default_alpha, type=float) # Selecting background word distribution weight
    # Beta - The number of pseudo documents generated for every class
    default_beta = 500
    parser.add_argument('--beta', '-b', default=default_beta, type=int) # Selecting the number of pseudo documents generated for every class
    # Gamma - The size of the keyword vocabulary
    default_gamma = 50
    parser.add_argument('--gamma', '-g', default=default_gamma, type=int) # Selecting the size of the keyword vocabulary

# Miscellaneous arguments
def set_parser_arguments_misc(parser):
    parser.add_argument('--trained_weights', '-t', default=None) # Directory containing the trained model


'''
This function is used for printing the predictions
for the self training data into a text file
'''
def write_output_to_file(write_path, y_pred, perm):
    # Get length of perm
    perm_length = len(perm)

    # Desired datatype
    datatype = 'int32'
    invperm = np.zeros(perm_length, dtype=datatype)

    for i,v in enumerate(perm):
        invperm[v] = i

    y_pred = y_pred[invperm]

    with open(os.path.join(write_path, 'out.txt'), 'w') as fptr:
        # For each value in the prediction, print to the file
        for value in y_pred:
            # Write to file
            fptr.write(str(value) + '\n')
    print("Classification results are written to {}".format(os.path.join(write_path, 'out.txt')))
    return

'''
This function writes the word embeddings obtained from Bert into a file
'''
def write_bert_embeddings_to_file(dataset_name, bert_embedding_weights):
    # Dataset directory
    bert_model_directory = './' + dataset_name

    # Write to file
    with open(os.path.join(bert_model_directory, 'BERTembedding.txt'), 'wb+') as fptr:
        pickle.dump(bert_embedding_weights, fptr)
        fptr.close()
        print("Embeddings written to file!\n")

    return

'''
This function reads the word embeddings obtained from Bert from a file
'''
def read_bert_embeddings_from_file(dataset_name):
    # The dataset name
    bert_model_directory = './' + dataset_name

    # Load the embeddings into a dict
    with open(os.path.join(bert_model_directory, 'BERTembedding.txt'), 'rb+') as fptr:
        embeddings = pickle.load(fptr)

    # Close file
    fptr.close()

    return embeddings


def train_bert_model(sentence_matrix, vocabulary_inv, dataset_name):
    ## Load pretrained model/tokenizer

    bert_model_directory = './' + dataset_name
    bert_model_name = os.path.join(bert_model_directory, "BERTembedding.txt")

    if not os.path.exists(bert_model_name):
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        bert_model.eval()

        # Obtain the sentences
        sentences = [[vocabulary_inv[w] for w in s] for s in sentence_matrix]

        embedding_model = {}
        c = 0
        # Add the special tokens.
        for s in sentences:
        
            print(c)
            s[0] = "[CLS] " + s[0] + " [SEP]"
            # Split the sentence into tokens.
            tokenized_s = bert_tokenizer.tokenize(s[0])
            # print(tokenized_s)
            # Map the token strings to their vocabulary indeces.
            indexed_s = bert_tokenizer.convert_tokens_to_ids(tokenized_s)

            # Convert inputs to PyTorch tensors
            s_tensor = torch.tensor([indexed_s])
        
            #Run the text through BERT, get the output and collect all of the hidden states produced from all 12 layers.
            with torch.no_grad():
                outputs = bert_model(s_tensor)
            # can use last hidden state as word embeddings
            last_hidden_state = outputs[0]
            word_embed_1 = last_hidden_state
            hidden_states = outputs[2]
            # initial embeddings can be taken from 0th layer of hidden states
            word_embed_2 = hidden_states[0]
            # sum of all hidden states
            word_embed_3 = torch.stack(hidden_states).sum(0)
            # sum of second to last layer
            word_embed_4 = torch.stack(hidden_states[2:]).sum(0)
            # sum of last four layer
            word_embed_5 = torch.stack(hidden_states[-4:]).sum(0)
            # concatenate last four layers
            final_word_embed = torch.cat([hidden_states[i] for i in [-1,-2,-3,-4]], dim=-1)

            # print(final_word_embed.shape)
            # (1,x,3072)
            token_vecs = final_word_embed[0]
            # print(token_vecs.shape)
            # # Calculate the average of all 22 token vectors.
            final_word_embed = torch.mean(token_vecs, dim=0)

            final_embed_np = final_word_embed.numpy()
            # # print(final_embed_np.shape)
            embedding_model[s[0]] = final_embed_np
            c += 1

        bert_embedding_weights = {}
        for key, word in vocabulary_inv.items():
            if word in embedding_model.keys():
                bert_embedding_weights[key] = embedding_model[word]
            else:
                bert_embedding_weights[key] = np.random.uniform(-0.25, 0.25, 3072)

        write_bert_embeddings_to_file(dataset_name, bert_embedding_weights)

    else:
        print("Preparing to load the existing bert model...")
        bert_embedding_weights = read_bert_embeddings_from_file(dataset_name)

    return bert_embedding_weights


'''
This function is used for obtaining the embedding weights from the word2vec model
that is obtained either by importing a pretraining model or training a new model in the
current runtime if the pretrained model doesn't exist.
'''
def train_word2vec_model(sentence_matrix, vocabulary_inv, dataset_name, mode='skipgram',
                   num_of_features=100, min_word_count=5, context=5):

    # Defining the path for the word2vec model
    w2v_model_directory = './' + dataset_name
    w2v_model_name = os.path.join(w2v_model_directory, "embedding")

    # If the pretrained model already exists, import it
    if os.path.exists(w2v_model_name):
        w2v_embedding_model = word2vec.Word2Vec.load(w2v_model_name)
        print("Preparing to load the existing Word2Vec model...")

    # If the pretrained model doesn't exist, train a new model
    else:
        print('Starting the training of the word2vec model...')

        # Obtain the sentences
        sentences = [[vocabulary_inv[w] for w in s] for s in sentence_matrix]

        # Change parameters based on whether the word2vec mode is skipgram or CBOW
        if mode == 'skipgram':
            sg = 1
            print('Model being used is skip-gram')
        elif mode == 'cbow':
            sg = 0
            print('Model being used is CBOW')

        # Training the word2vec model
        w2v_embedding_model = word2vec.Word2Vec(sentences, workers=15, sg=sg,
                                            size=num_of_features, min_count=min_word_count,
                                            window=context, sample=1e-3)

        w2v_embedding_model.init_sims(replace=True)

        # If the specified directory doesnt exist, make the directory
        if not os.path.exists(w2v_model_directory):
            os.makedirs(w2v_model_directory)

        # Saving the word2vec model
        w2v_embedding_model.save(w2v_model_name)
        print("Word2Vec model saved!")

    # Obtain the embedding weights from the word2vec model
    w2v_embedding_weights = {key: w2v_embedding_model[word] if word in w2v_embedding_model else
                        np.random.uniform(-0.25, 0.25, w2v_embedding_model.vector_size)
                         for key, word in vocabulary_inv.items()}
    # print(w2v_embedding_weights)

    # Return the embedding weights
    return w2v_embedding_weights

# The main function:
if __name__ == "__main__":

    import argparse

    # Defining an argument parser
    parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    args = arguments_parser(parser)

    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    delta = args.delta

    if args.language_model == "w2v":
        word_embedding_dim = 100
    elif args.language_model == "bert":
        word_embedding_dim = 3072
    
    # Change parameters depending on dataset for CNN
    if args.dataset == 'agnews':
        update_interval = 50
        pretrain_epochs = 20
        self_lr = 1e-3
        max_sequence_length = 100

    elif args.dataset == 'yelp':
        update_interval = 50
        pretrain_epochs = 30
        self_lr = 1e-4
        max_sequence_length = 500
    elif args.dataset == 'imdb':
        update_interval = 50
        pretrain_epochs = 30
        self_lr = 1e-4
        max_sequence_length = 500

    decay = 1e-6

    if args.update_interval is not None:
        update_interval = args.update_interval
    if args.pretrain_epochs is not None:
        pretrain_epochs = args.pretrain_epochs

    if args.with_evaluation == 'True':
        with_evaluation = True
    else:
        with_evaluation = False
    if args.sup_source == 'labels' or args.sup_source == 'keywords':
        x, y, word_counts, vocabulary, vocabulary_inv_list, len_avg, len_std, word_sup_list, perm = \
            load_dataset(args.dataset, model=args.model, sup_source=args.sup_source, with_evaluation=with_evaluation, truncate_len=max_sequence_length)
        sup_idx = None
    elif args.sup_source == 'docs':
        x, y, word_counts, vocabulary, vocabulary_inv_list, len_avg, len_std, word_sup_list, sup_idx, perm = \
            load_dataset(args.dataset, model=args.model, sup_source=args.sup_source, with_evaluation=with_evaluation, truncate_len=max_sequence_length)
    
    np.random.seed(1234)
    # print(vocabulary_inv_list)
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
    # print("  ... .. . ")
    # print(vocabulary_inv)
    vocab_sz = len(vocabulary_inv)
    n_classes = len(word_sup_list)

    # Set sequence length for CNN
    if x.shape[1] < max_sequence_length:
            max_sequence_length = x.shape[1]
    x = x[:, :max_sequence_length]
    sequence_length = max_sequence_length

    print("\n### Input preparation ###")
    if args.language_model == "w2v":
        embedding_weights = train_word2vec_model(x, vocabulary_inv, args.dataset)
    elif args.language_model == "bert":
        embedding_weights = train_bert_model(x, vocabulary_inv, args.dataset)
    embedding_mat = np.array([np.array(embedding_weights[word]) for word in vocabulary_inv])
    # print(embedding_mat)
    # print(embedding_mat.size)
    
    wstc = WSTC(input_shape=x.shape, n_classes=n_classes, y=y, model=args.model,
                vocab_sz=vocab_sz, embedding_matrix=embedding_mat, word_embedding_dim=word_embedding_dim)

    if args.trained_weights is None:
        print("\n### Phase 1: vMF distribution fitting & pseudo document generation ###")
        
        word_sup_array = np.array([np.array([vocabulary[word] for word in word_class_list]) for word_class_list in word_sup_list])
        
        total_counts = sum(word_counts[ele] for ele in word_counts)
        total_counts -= word_counts[vocabulary_inv_list[0]]
        background_array = np.zeros(vocab_sz)
        for i in range(1,vocab_sz):
            background_array[i] = word_counts[vocabulary_inv[i]]/total_counts
        seed_docs, seed_label = pseudodocs(word_sup_array, gamma, background_array,
                                           sequence_length, len_avg, len_std, beta, alpha, 
                                           vocabulary_inv, embedding_mat, args.model, 
                                           './results/{}/{}/phase1/'.format(args.dataset, args.model))
        
        if args.sup_source == 'docs':
            # Number of real documents for CNN
            num_real_doc = len(sup_idx.flatten()) * 10
            real_seed_docs, real_seed_label = augment(x, sup_idx, num_real_doc)
            seed_docs = np.concatenate((seed_docs, real_seed_docs), axis=0)
            seed_label = np.concatenate((seed_label, real_seed_label), axis=0)

        perm_seed = np.random.permutation(len(seed_label))
        seed_docs = seed_docs[perm_seed]
        seed_label = seed_label[perm_seed]

        print('\n### Phase 2: pre-training with pseudo documents ###')

        wstc.pretrain(x=seed_docs, pretrain_labels=seed_label,
                     sup_idx=sup_idx, optimizer=SGD(lr=0.1, momentum=0.9),
                     epochs=pretrain_epochs, batch_size=args.batch_size,
                     save_dir='./results/{}/{}/phase2'.format(args.dataset, args.model))

        y_pred = wstc.predict(x)
        if y is not None:
            f1_score = np.round(f1(y, y_pred), 5)
            print('F1 score after pre-training: {}'.format(f1_score))

            accuracy_value = np.round(accuracy(y, y_pred), 5)
            print('Accuracy score after pre-training: accuracy = {}'.format(accuracy_value))

        t0 = time()
        print("\n### Phase 3: self-training ###")
        selftrain_optimizer = SGD(lr=self_lr, momentum=0.9, decay=decay)
        wstc.compile(optimizer=selftrain_optimizer, loss='kld')
        y_pred = wstc.fit(x, y=y, tol=delta, maxiter=args.max_iter, batch_size=args.batch_size,
                         update_interval=update_interval, save_dir='./results/{}/{}/phase3'.format(args.dataset, args.model), 
                         save_suffix=args.dataset+'_'+str(args.sup_source))
        print('Self-training time: {:.2f}s'.format(time() - t0))

    else:
        print("\n### Directly loading trained weights ###")
        wstc.load_weights(args.trained_weights)
        y_pred = wstc.predict(x)
        if y is not None:
            f1_score = np.round(f1(y, y_pred), 5)
            print('F1 score after pre-training: {}'.format(f1_score))

            accuracy_value = np.round(accuracy(y, y_pred), 5)
            print('Final Accuracy score: accuracy = {}'.format(accuracy_value))

    print("\n### Generating outputs ###")
    write_output_to_file('./' + args.dataset, y_pred, perm)
