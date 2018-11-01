import os
import codecs
from transliteration.IndictransTransliterator import *
from transliteration.EmbeddingsRewriter import *

raw_dir = 'data/raw'
embeddings_dir = os.path.join(raw_dir, 'indic-word2vec-embeddings')
clean_dir = 'data/clean'

if not os.path.exists(clean_dir):
    os.makedirs(clean_dir)

transliterator = IndictransTransliterator()

# transliterate Hindi training data and move to clean directory
input_file = os.path.join(raw_dir, "hi-ud-train.conllu")
output_file = os.path.join(clean_dir, "hi_roman-ud-train.conllu")

print "pre-transliterating Hindi training data..."
with codecs.open(input_file, mode='r', encoding='utf-8') as fin:
	with codecs.open(output_file, mode='w', encoding='utf-8') as fout:
		for line in fin.readlines():
			for index,token in enumerate(line.split()):
				# transliterate the original devanagari token
				if index == 1 and IndictransTransliterator._is_deva(token):
					token = transliterator.transliterate(token)
				fout.write(token + '\t')
			fout.write('\n')


# transliterate Hindi embeddings and move to clean directory
input_file = os.path.join(embeddings_dir, "Hindi_utf.vec")
output_file = os.path.join(clean_dir, "Hindi_roman.vec")

print "pre-transliterating Hindi word embeddings..."
rewriter = EmbeddingsRewriter(transliterator)
rewriter.rewrite(input_file, output_file)

print "done!"

