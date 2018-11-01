import codecs
import os
from IndictransTransliterator import *

class EmbeddingsRewriter(object):
	def __init__(self, transliterator):
		self.transliterator = transliterator

	# Given an embeddings file with standard (devanagari) Hindi as input, rewrites the embeddings with all possible transliterations
	# to a file with name output_file

	def rewrite(self, input_file, output_file):
		empty = 0
		with codecs.open(input_file, mode='r', encoding='utf-8') as fin:
			with codecs.open(output_file, mode='w', encoding='utf-8') as fout:
				for line in fin.readlines():
					tokens = line.split()
					original = tokens[0]
					embedding = tokens[1:]
					if IndictransTransliterator._is_deva(original):
							transliterations = self.transliterator.get_all(original)
							for target in transliterations:
								fout.write(target + ' ')
								fout.write(' '.join(embedding) + '\n')
					else:
						fout.write(line)

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--input-file', dest='input_file')
	parser.add_argument('--output-file', dest='output_file')
	options = parser.parse_args()
	rewriter = EmbeddingsRewriter(IndictransTransliterator())
	rewriter.rewrite(options.input_file, options.output_file)




