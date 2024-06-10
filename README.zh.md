# 中文詞彙網路 (CWN) 詞義標記器

中文詞義消歧一直以來都是個難題，因為中文一詞多義的現象非常普遍。手動標記詞義既費時又低效。

本專案旨在運用 Bert 模型解決此問題，大幅提升了效能，在中文詞義消歧任務上達到約 82% 的準確度。

## 前置需求
* 建議先進行分詞與詞性標註，但非必要條件。
* 假設我們有 m 個句子，每個句子有 $n_m$ 個詞。
    * list_of_sentence[ [list_of_word[[target, pos] * $n_m$ ]] *m ]
    * 以下是一個包含 2 個句子的範例，輸入資料應該如下所示：

            tagged=[[["他","Nh"],["由","P"],["昏沈","VH"],["的","DE"],["睡夢","Na"],["中","Ng"],["醒來","VH"],["，","COMMACATEGORY"]],
             [["臉","Na"],["上","Ncd"],["濕涼","VH"],["的","DE"],["騷動","Nv"],["是","SHI"],["淚","Na"],["。","PERIODCATEGORY"]]]
             

## 如何取得詞義
* 在您的專案根目錄下執行：

        pip install -U gdown
        git clone https://github.com/lopentu/CwnSenseTagger
        pip install -q CwnGraph transformers
        import sys
        if "dwsd-beta" not in sys.path:
            sys.path.append("dwsd-beta")
        
        from dotted_wsd import DottedWsdTagger
        tagger = DottedWsdTagger()

* 基本查詢

        tagger.wsd_tag("<打>電話")[0]

* 查詢分詞與詞性標註後的結果

        tagger.sense_tag_per_sentence(tagged[0])

## 結合 [ckip-transformers](https://github.com/ckiplab/ckip-transformers) 與 CWN 詞義標記器的 Pipeline
* 請參考 [dwsd.ipynb](https://github.com/lopentu/CwnSenseTagger/blob/master/dwsd.ipynb)


## 致謝
感謝 Po-Wen Chen (b05902117@ntu.edu.tw) 和 Yu-Yu Wu (b06902104@ntu.edu.tw) 在模型開發上的貢獻。
