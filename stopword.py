from posixpath import join
import re
import codecs
from nltk.util import pr
from underthesea import word_tokenize
# Đọc file chứa những từ không mang ý nghĩa trong tiếng việt
filename = "stopword.txt"
with codecs.open(filename, 'r', encoding='utf8') as f_obj:
    stopwords = f_obj.read()
    stopwords = stopwords.splitlines()
def stop_word(cau):
    tu = word_tokenize(cau)
    cau_hoan_chinh = []
    for tus in tu:
        if tus not in stopwords:
            cau_hoan_chinh.append(tus)
    return ' '.join(cau_hoan_chinh)
