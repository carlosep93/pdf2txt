from bs4 import BeautifulSoup
import ctranslate2
import transformers
from texttokenizer import TextTokenizer
import pandas as pd
import unicodedata
import os
import argparse

def _normalize_input_string(result):
    result = unicodedata.normalize('NFC', result)
    return result

def _translate_batch(input_batch, spm, model, max_sentence_batch=10):

    batch_input_tokenized = []
    batch_input_markers = []

    #preserve_markup = PreserveMarkup()

    num_sentences = len(input_batch)
    for pos in range(0, num_sentences):
        tokenized = spm.convert_ids_to_tokens(spm.encode(input_batch[pos]))
        batch_input_tokenized.append(tokenized)

    batch_output = []
    for offset in range(0,len(batch_input_tokenized), max_sentence_batch):
      batch = batch_input_tokenized[offset:offset+max_sentence_batch]
      partial_result = model.translate_batch(batch, 
                                            return_scores=False, 
                                            replace_unknowns=True, 
                                            target_prefix=[target_prefix]*len(batch))
      for pos in range(0,len(partial_result)):
        tokenized = partial_result[pos][0]['tokens'][1:]
        translated = spm.decode(spm.convert_tokens_to_ids(tokenized))
        print(translated)
        batch_output.append(translated)

    return batch_output

def translate(text):
  sentences, translate = tokenizer.tokenize(text)
  num_sentences = len(sentences)
  sentences_batch = []
  indexes = []
  results = ["" for x in range(num_sentences)]
  for i in range(num_sentences):
      if translate[i] is False:
          continue

      sentences_batch.append(sentences[i])
      indexes.append(i)

  translated_batch = _translate_batch(sentences_batch, spm, translator, max_sentence_batch)
  for pos in range(0, len(translated_batch)):
      i = indexes[pos]
      results[i] = translated_batch[pos]

  #Rebuild split sentences
  translated = tokenizer.sentence_from_tokens(sentences, translate, results)
  return translated


def translate_xml_tree(xml_string, tags=['heading','p']):
  xml_string = xml_string.replace('\\n','\n')
  xml_tree = BeautifulSoup(xml_string, features="xml")
  for tag in tags:
    instances = xml_tree.findAll(tag)
    for  ins in instances:
        text = ins.getText()
        ins.string = translate(text)
  return str(xml_tree).replace('\n','\\n')

def get_out_file(data_file, out_dir):
    data_file = str(data_file).split('/')[-1]
    data_file = data_file.split('.')[0]
    return out_dir + '/' + data_file + '.tr'



parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", help="Directory where the model is stored.")
parser.add_argument("--tokenizer_dir", help="Directory where the tokenizer is stored.")
parser.add_argument("--data_dir", help="Directory where the data is stored.")
parser.add_argument("--output_dir", help="Directory where the output is stored.")
parser.add_argument("--parquet_file", help="Name of the .tsv file to translate.")
parser.add_argument("--src_lang", help="Source language in NLLB format.")
parser.add_argument("--tgt_lang", help="Target language in NLLB format.")
parser.add_argument("--tok_lang", help="Language used for pretokenization. Full Word in English.")
args = parser.parse_args()



#Init model to translate
model_dir=args.model_dir
tokenizer_dir=args.tokenizer_dir
language=args.tok_lang

spm=transformers.AutoTokenizer.from_pretrained(tokenizer_dir, src_lang=args.src_lang)

try:
    translator = ctranslate2.Translator(model_dir, device="cuda")
except:
    translator = ctranslate2.Translator(model_dir, device="cpu")

tokenizer=TextTokenizer(language)


#Load data from parquet file
data_dir=args.data_dir
out_dir=args.output_dir
target_prefix=[args.tgt_lang]
max_sentence_batch=40
input_field='text'
output_field='translation'


f = data_dir + '/' + args.parquet_file
print('Reading file:', f)
out_file = get_out_file(f,out_dir)

if os.path.isfile(out_file):
    print('File already processed:', f)
    exit(0)

df = pd.read_parquet(f)
df[output_field] = pd.Series(dtype='str')
for row_index,row in df.iterrows():
    print ('Translating sentece', row_index, "out of", len(df))
    text = _normalize_input_string(row[input_field].strip())
    df.loc[row_index,[output_field]] = translate_xml_tree(text)

print('Writing output file:', out_file)
df.to_parquet(out_file)


