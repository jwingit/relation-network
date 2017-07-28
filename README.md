# relation-network
keras implementation of  [A simple neural network module for relational reasoning](https://arxiv.org/pdf/1706.01427.pdf)

Relation network is a noval neural network introduced by deepmind in [A simple neural network module for relational reasoning](https://arxiv.org/pdf/1706.01427.pdf). It can achieve super-human performance in challenging visual question answering datasets such as CLEVR.

I implement Relation network using keras and train it on a challenging visual question answering dataset called [Cornell NLVR](https://github.com/cornell-lic/nlvr).

Jameson:
I modified train.py to have a what I think is a more intuitive construction of the relation network itself. I also added positional encoding for features, and altered the format of the images themselves (transposed).

Note you'll also need to obtain the glove.6B.50d.txt file which used to be at:
http://www-nlp.stanford.edu/data/glove.6B.50d.txt.gz
But this has been replaced by glove.6B.zip which contains among others the glove.6B.50d.txt file.


