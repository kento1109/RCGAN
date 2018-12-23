# RCGAN

## 実行方法
```
python experiment.py
```

デフォルトでは、 `inputs/sin_wave.npz` のデータを入力とする。  
入力の `npz` の形式は以下のような保存形式を想定

```
ndarr_x = np.array([0.0, 0.1], [0.2, 0.3], [0.4, 0.5], [0.6, 0.7]])  # original data
ndarr_y = np.array([0, 1, 2, 3])  # label

np.savez('test.npz', x=ndarr_x, y=ndarr_y)
```

---

（参考論文）
- REAL-VALUED (MEDICAL) TIME SERIES GENERATION WITH RECURRENT CONDITIONAL GANS  
    https://arxiv.org/pdf/1706.02633.pdf
    
- Generative Adversarial Nets  
    https://arxiv.org/pdf/1406.2661.pdf
    
- Conditional Generative Adversarial Nets  
    https://arxiv.org/pdf/1411.1784.pdf
    
- GENERATIVE MODELS AND MODEL CRITICISM VIA OPTIMIZED MAXIMUM MEAN DISCREPANCY  
    https://arxiv.org/pdf/1611.04488.pdf

- Deep Learning with Differential Privacy  
    https://arxiv.org/pdf/1607.00133.pdf

- Improved Techniques for Training GANs  
    https://arxiv.org/pdf/1606.03498.pdf
    
---    

（参考リポジトリ）
- RGAN  
    https://github.com/ratschlab/RGAN

- SyntheticMedData  
    https://github.com/naegawa/SyntheticMedData
    
- opt-mmd  
    https://github.com/dougalsutherland/opt-mmd

- Deep Learning with Differential Privacy  
    https://github.com/tensorflow/models/tree/master/research/differential_privacy
    
- Keras-GAN  
    https://github.com/eriklindernoren/Keras-GAN

 ---
 
 （その他参考サイト）
 - 注目のプライバシー Differential Privacy  
     https://www.jstage.jst.go.jp/article/jssst/29/4/29_4_40/_pdf
