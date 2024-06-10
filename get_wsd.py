import sys
if "dwsd-beta" not in sys.path:
  sys.path.append("dwsd-beta")

from dotted_wsd import DottedWsdTagger

tagger = DottedWsdTagger()

import torch
import ckip_transformers
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger
import re
import pandas as pd

device = 0 if torch.cuda.is_available() else -1
ws_driver = CkipWordSegmenter(device=device)
pos_driver = CkipPosTagger(device=device)

def get_dotted_wsd(text):

  id_pattern = '\[(.*?)\]'
  con_pattern = '\((.*?)\)'
  gloss_pattern = '(\[|\()\S+(\]|\))|\s'

  lemma, pos, senseID, confidence, gloss = [], [], [], [], []
  for t in text:
    tt = tagger.sense_tag_per_sentence(t)
    for i in tt:
      lemma.append(i[0])
      pos.append(i[1])
      if len(i[2]) > 0:
        id_search = re.search(id_pattern, i[2])
        con_search = re.search(con_pattern, i[2])
        senseID.append(id_search[0][1:-1])
        confidence.append(con_search[0][1:-1])
        gloss.append(re.sub(gloss_pattern, '', i[2]))
      else:
        gloss.append('')
        senseID.append('')
        confidence.append('')

  df = pd.DataFrame({
      'Lemma':lemma,
      'Part-of-Speech':pos,
      'Sense_id':senseID,
      'Gloss': gloss,
      'Confidence': confidence
  })
  return df



def get_wsd(data, save=False, output=None):
    
    content = [re.sub('\W+', ' ', c) for c in data]
    content = [s for c in content for s in c.split(' ') ]

    ws = ws_driver(content, show_progress=False)
    pos = pos_driver(ws, show_progress=False)
    tagged = [[(a[i], b[i]) for i in range(len(a))] for a,b in zip(ws, pos)]
    tagged = get_dotted_wsd(tagged)

    if save:
        tagged.to_csv(output, index=False)

    return tagged