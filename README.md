# Similarity Search

<h3>Dataset</h3>
<p><a href="http://mvc-datasets.github.io/MVC/" target="_blank">MVC</a> dataset for cloth retrieval.</p>

* Xception network was fine tuned and trained on the MVC dataset.
* The last Dense layer was removed and the GlobalAveragePooling layer was used as feature extractor.
* L2 distance is calculated between the search image and all the images in the dataset and the top 4 indices are retrieved.
