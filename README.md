# Word Sense Disambiguaion by Chinese Word Net

Chinese word sense disambiguation has been known to a very difficult problem since Chinese is a complicated language. A word can have dozens or even hundreds of meanings on different occasions. Manually labels the senses of the words is labor-intensive and inefficient.

In this project, we aim to solve this problem by state-of-the-art Bert model.  It gives us huge performance gains and can score roughly 82% accuracy on Chinese word sense disambiguation problem.

## Prerequest
* Input should be tokenized first. POS Tagging is preferred but not required.
* Suppose we have m sentences and each sentence has $n_m$ words.
    * list_of_sentence[ [list_of_word[[target, pos, sense_id, sense] * $n_m$ ] *m ]
    * The following is an example that has 2 sentences, input data should be formed as following

            [[["他","Nh","",""],["由","P","",""],["昏沈","VH","",""],["的","DE","",""],["睡夢","Na","",""],["中","Ng","",""],["醒來","VH","",""],["，","COMMACATEGORY","",""]],
             [["臉","Na","",""],["上","Ncd","",""],["濕涼","VH","",""],["的","DE","",""],["騷動","Nv","",""],["是","SHI","",""],["淚","Na","",""],["。","PERIODCATEGORY","",""]]]
             

## How to get sense
* At Project root directory (same as setup.py)

        pip3 install .
        import CWN_WSD
        data = read_somewhere() #list of sentence, and sentence is composed as list of word
        sense = CWN_WSD.wsd(data)
* example can be found under example folder

## Acknowledgement
We thank Po-Wen Chen (b05902117@ntu.edu.tw) and Yu-Yu Wu (b06902104@ntu.edu.tw) for contributions in model development.
        
    
    


