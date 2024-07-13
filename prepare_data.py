import json
from typing import List, Tuple, Dict
import pickle


data_p = 'ccpc_train_v1.0.json'
voc_size = 5000
SPECIAL_TOKENS = {
    '<PAD>': 0,
    '<SOS>': 1,
    '<EOS>': 2,
    '<UNK>': 3
}


def get_data(data_p: str) -> List[str]:
    data = []
    with open(data_p, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    text_lst = [_['content'].replace('\n', '').replace(' ', '') for _ in data]
    return text_lst


def make_dict(data: List[str], 
              voc_size: int) -> Tuple[Dict, Dict, Dict]:
    """
    input: list of documents
    return 3 dict: word2idx, idx2word, word2count
    """
    word2count = dict()
    for doc in data:
        for word in doc:
            if word not in word2count.keys():
                word2count[word] = 1
            else:
                word2count[word] += 1
    word2count = sorted(word2count.items(), key=lambda item: item[1], reverse=True)
    
    word2count_left = word2count[:voc_size]
    word2idx = {item[0]: len(SPECIAL_TOKENS) + i for i, item in enumerate(word2count_left)}
    word2idx.update(SPECIAL_TOKENS)
    idx2word = {v:k for k, v in word2idx.items()}
    return word2idx, idx2word, word2count


def pretoknize(data: List[str], word2idx: Dict[str, int]) -> List[int]:
    """
    tokenize List of string into List of int
    """
    out = []
    for doc in data:
        tokenized = [word2idx.get(char, SPECIAL_TOKENS['<UNK>']) for char in doc]
        out.append(tokenized)
    return out

def encode_text(text: str, word2idx: Dict):
    return [word2idx.get(c, SPECIAL_TOKENS['<UNK>']) for c in text]

def decode_text(idxes: List[int], idx2word: Dict):
    return ''.join([idx2word[idx] for idx in idxes])


if __name__ == '__main__':
    # use train data to build vocab and pretokenize train data
    text_lst = get_data(data_p)
    print(f'get {len(text_lst)} lines of data')
    word2idx, idx2word, word2count = make_dict(text_lst, voc_size)
    print(f'ori voc size: {len(word2count)}, target voc size: {voc_size}')
    print('final voc size with special tokens: ', len(word2idx))
    print(SPECIAL_TOKENS)
    pretoknized_data = pretoknize(text_lst, word2idx)
    with open('pretoknized_data.pkl', 'wb') as f:
        pickle.dump(pretoknized_data, f)


    with open('word2idx.pkl', 'wb') as f:
        pickle.dump(word2idx, f)

    with open('idx2word.pkl', 'wb') as f:
        pickle.dump(idx2word, f)


    # use  builded vocab and to pretokenize val data
    print()
    print('use builded vocab to pretokenize val data')
    val_data_p = 'ccpc_valid_v1.0.json'
    text_lst = get_data(val_data_p)
    print(f'get {len(text_lst)} lines of data')
    with open('word2idx.pkl', 'rb') as f:
        word2idx = pickle.load(f)

    with open('idx2word.pkl', 'rb') as f:
        idx2word = pickle.load(f)

    pretoknized_data = pretoknize(text_lst, word2idx)
    with open('pretoknized_val_data.pkl', 'wb') as f:
        pickle.dump(pretoknized_data, f)


    print()
    print('use builded vocab to pretokenize test data')
    val_data_p = 'ccpc_test_v1.0.json'
    text_lst = get_data(val_data_p)
    print(f'get {len(text_lst)} lines of data')
    with open('word2idx.pkl', 'rb') as f:
        word2idx = pickle.load(f)

    with open('idx2word.pkl', 'rb') as f:
        idx2word = pickle.load(f)

    pretoknized_data = pretoknize(text_lst, word2idx)
    with open('pretoknized_test_data.pkl', 'wb') as f:
        pickle.dump(pretoknized_data, f)