# streaming_node2vec

This repository is a part of the course CS430 Algorithms for Data Science at IIT Gandhinagar. We have used three different sampling algorithms at ~10% and ~1% and generated [node2vec](https://github.com/aditya-grover/node2vec) feature vectors using that.  Then we have used Support Vector Machines for Link Predictions. The presentation contains entire training procedure. 

## Sampling Techniques

* [Reservoir Sampling](https://en.wikipedia.org/wiki/Reservoir_sampling)

* [Graph Priority Sampling](https://arxiv.org/pdf/1703.02625.pdf)

  * Using number of triangles completed by the edge as weight
  * Minimum degree of the nodes of the upcoming edge as weight

* [Graph Sample and Hold](https://www.cs.purdue.edu/homes/neville/papers/ahmed-et-al-kdd2014.pdf)

  ​

  ## Link Prediction

  [Supervised Link Prediction](http://www.siam.org/meetings/sdm06/workproceed/Link%20Analysis/12.pdf)

   