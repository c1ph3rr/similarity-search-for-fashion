# Similarity Search

<h3>Dataset</h3>
<p><a href="http://mvc-datasets.github.io/MVC/" target="_blank">MVC</a> dataset for cloth retrieval.</p>

* Xception network was fine tuned and trained on the MVC dataset.
* The last Dense layer was removed and the GlobalAveragePooling layer was used as feature extractor.
* L2 distance is calculated between the search image and all the images in the dataset and the top 4 indices are retrieved.

<h3>Result</h3>

![alt text](https://github.com/c1ph3rr/similarity-search-for-fashion/blob/master/results/q1.png)
![alt text](https://github.com/c1ph3rr/similarity-search-for-fashion/blob/master/results/s1.png)

![alt text](https://github.com/c1ph3rr/similarity-search-for-fashion/blob/master/results/q2.png)
![alt text](https://github.com/c1ph3rr/similarity-search-for-fashion/blob/master/results/s2.png)

![alt text](https://github.com/c1ph3rr/similarity-search-for-fashion/blob/master/results/q3.png)
![alt text](https://github.com/c1ph3rr/similarity-search-for-fashion/blob/master/results/s3.png)
