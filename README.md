# GrainNN

A transformer-embedded seq2seq LSTM for grain microstructure evolution

# Build

```
pip install -r requirements.txt
```

# Usage
training
```
cd GrainNN_2D
python3 grainNN.py train 
```

testing

```
cd GrainNN_2D
python3 grainNN.py test 
```



# Example 

Rectangular simulation
![readme2d](https://user-images.githubusercontent.com/62076142/172073935-421b9c17-d2ce-48be-b534-9e337deeb170.png)


https://user-images.githubusercontent.com/62076142/174496234-25d421b5-c53e-4a3f-9790-f5c93532155a.mp4

Metpool simulation


https://user-images.githubusercontent.com/62076142/195922359-2efcbc4a-ae53-4c0a-936d-fcc43f9d9138.mp4

# Reference
```
[1] Xingjian, S. et al. Convolutional lstm network: A machine learning approach for precipitation nowcasting. In Advances in neural information processing systems, 802–810 (2015).
[2] Vaswani, A. et al. Attention is all you need. In Advances in neural information processing systems, 5998–6008 (2017).
```
# Author
This software was primarily written by Yigong Qin who is advised by Prof. George Biros.
