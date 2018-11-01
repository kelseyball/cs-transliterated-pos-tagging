from indictrans import Transliterator
import random

class IndictransTransliterator:
	def __init__(self):
		self.trn = Transliterator(source='hin', target='eng', decode='beamsearch', build_lookup=True)
		self.trans_dict = {}

	def transliterate(self, original):
		transliterations = self.get_all(original)
		return random.choice(transliterations)

	def get_all(self, original):
		if original in self.trans_dict:
			return self.trans_dict[original]
		else:			
			transliterations = self.trn.transform(original, k_best=5)
			self.trans_dict[original] = transliterations
			return transliterations

	@staticmethod
	def _is_deva(unicode_tok):
		"""Returns True if |unicode_tok| contains a Devanagari character"""
		for c in unicode_tok:
			if int('0900', 16) <= ord(c) <= int('097f', 16):
				return True
		return False
