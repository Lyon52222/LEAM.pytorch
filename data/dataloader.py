import spacy
from torchtext import data
from torchtext.data import TabularDataset, Iterator
from torchtext.vocab import Vectors
from torch.nn import init
import os
class LEAMDataLoader():
    def __init__(self, opt):

        spacy_en = spacy.load('en')

        def tokenizer(text):
            return [tok.text for tok in spacy_en.tokenizer(text)]

        self.TEXT = data.Field(tokenize=tokenizer, batch_first=True)
        LABEL = data.Field(sequential=False, use_vocab=False)
        self.LABEL_TEXT = data.Field(tokenize=tokenizer, batch_first=True, fix_length=4)

        fields = [("label", LABEL), ("text", self.TEXT), ("label_text", self.LABEL_TEXT)]

        train_ds = TabularDataset(path=os.path.join(opt.data_path,opt.train_file), format='csv', fields=fields, skip_header=True)
        self.train_itr = Iterator(train_ds, batch_size=opt.batch_size, device=opt.device, sort_key=lambda x: len(x.text))

        val_ds = TabularDataset(path=os.path.join(opt.data_path,opt.val_file), format='csv', fields=fields, skip_header=True)
        self.val_itr = Iterator(val_ds, batch_size=opt.batch_size, device=opt.device, sort_key=lambda x: len(x.text))

        test_ds = TabularDataset(path=os.path.join(opt.data_path,opt.test_file), format='csv', fields=fields, skip_header=True)
        self.test_itr = Iterator(test_ds, batch_size=opt.batch_size, device=opt.device, sort_key=lambda x: len(x.text))

        vectors = Vectors(opt.word_vectors)
        self.TEXT.build_vocab(train_ds, vectors=vectors)
        self.TEXT.vocab.vectors.unk_init = init.xavier_uniform

        self.LABEL_TEXT.build_vocab(train_ds, vectors=vectors)
        self.LABEL_TEXT.vocab.vectors.unk_init = init.xavier_uniform

    def get_loader(self, type='all'):
        if type=='train':
            return self.train_itr
        if type=='val':
            return self.val_itr
        if type=='test':
            return self.test_itr
        if type=='all':
            return (self.train_itr, self.val_itr, self.test_itr)
    
    def get_text_vocab(self):
        return self.TEXT.vocab

    def get_label_text_vocab(self):
        return self.LABEL_TEXT.vocab