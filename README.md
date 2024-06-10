[![en](https://img.shields.io/badge/lang-en-red.svg)](https://github.com/lopentu/CwnSenseTagger/blob/master/README.md)
[![zh](https://img.shields.io/badge/lang-zh-green.svg)](https://github.com/lopentu/CwnSenseTagger/blob/master/README.zh.md)

# Word Sense Disambiguaion by Chinese Word Net

Chinese word sense disambiguation has been known to a very difficult problem since Chinese is a complicated language. A word can have dozens or even hundreds of meanings on different occasions. Manually labels the senses of the words is labor-intensive and inefficient.

In this project, we aim to solve this problem by state-of-the-art Bert model.  It gives us huge performance gains and can score roughly 82% accuracy on Chinese word sense disambiguation problem.

## Prerequest
* Word Segmentation and POS Tagging is preferred but not required.
* Suppose we have m sentences and each sentence has $n_m$ words.
    * list_of_sentence[ [list_of_word[[target, pos] * $n_m$ ]] *m ]
    * The following is an example that has 2 sentences. Input data should be formed as following:

            tagged=[[["他","Nh"],["由","P"],["昏沈","VH"],["的","DE"],["睡夢","Na"],["中","Ng"],["醒來","VH"],["，","COMMACATEGORY"]],
             [["臉","Na"],["上","Ncd"],["濕涼","VH"],["的","DE"],["騷動","Nv"],["是","SHI"],["淚","Na"],["。","PERIODCATEGORY"]]]
             

## How to get sense
* In your Project root directory

        pip install -U gdown
        git clone https://github.com/lopentu/CwnSenseTagger
        pip install -q CwnGraph transformers
        import sys
        if "dwsd-beta" not in sys.path:
            sys.path.append("dwsd-beta")
        
        from dotted_wsd import DottedWsdTagger
        tagger = DottedWsdTagger()

* Basic Query

        tagger.wsd_tag("<打>電話")[0]

* Query with a result after Word Segmentation and POS tagging

        tagger.sense_tag_per_sentence(tagged[0])

## A Pipeline of [ckip-transformers](https://github.com/ckiplab/ckip-transformers) and the CWN sense tagger
* See [dwsd.ipynb](https://github.com/lopentu/CwnSenseTagger/blob/main/dwsd.ipynb)


## Acknowledgement
We thank Po-Wen Chen (b05902117@ntu.edu.tw) and Yu-Yu Wu (b06902104@ntu.edu.tw) for contributions in model development.
