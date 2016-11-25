# Automated Extraction of Product Attributes Documentation

Process
---

### Construct a Network of Words
Construct a list of dictionaries of all products excluding non-material items and items with model numbers as its only description. For each dictionary, the key is the product id and the value is the other important product values most notably description.

Build a list of lists where each list contains words from product description separated by space. Then removed any word with two characters or less and stop-words (list acquired from nltk.corpus module) to filter out any redundant word.

In each list, pair one word with every other word after it, count the distance between the two words, and repeat the procedure for every other word after the first one. 

Build a dictionary of links where the key is a tuple of two words and the value is the distance between them. To avoid duplicates of any two pair of words, the distance of a duplicate is added to the first pair value.

Use networkx package to build a non-directed weighted graph using the pair of words where the two words in each pair are nodes and the weight of the edge between the nodes is the distance. 

### Calculate Network Measures
Build a list of words using nodes from the graph of words.

Use networkx module functions to calculate assortativity (average neighbor degree), clustering coefficient, degree, eigenvector centrality, weight (use list of pairs of words), betweeness centrality, and closeness centrality for each word in the list of words.

Normalize specific network feature values (assortativity clustering coefficient, degree, eigenvector, weight) for each word to a range of 0 to 1 using formula from following site:
http://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range

### Remove Correlated Measures
Use corrcoef (correlation coefficient) module from numpy package to calculate correlation between all of the network features and remove the ones that are highly correlated (value greater than 0.80); degree, weight, and closeness centrality. See correlation table attached.

### Build a Vector Space
Build a dictionary of words with the word as the key and a list of normalized network features excluding the three redundant measures as the value.

### Perform Clustering
Use sklearn.cluster.KMeans with an optimized k value (k = 45), the number of iteration of the algorithm set to 50, and the number of time the algorithm runs with different centroid seeds set to 50 (and keeps the centroid seeds in the best run) in order to acquire the best result. There is no particular reason as to why 50 is chosen.

Use the list of centers and list of labels which are returned by the KMeans function to calculate the overlapping coefficient of each cluster. Repeat this procedure for different k value from 5 to 60 and locate the k value with the lowest coefficient that is greater than 0; k = 45.

### Locate Attribute Rich Clusters
Build a statistic table of cluster id, size of cluster, number of actual attribute found (verify if each word is an attribute by running all the regular expressions used to find attributes in product description), percentage of the cluster are attributes, and z-score. The attribute-rich clusters are ones with a z-score greater than 1.96.

To maximize the z-score of attribute-rich clusters, the number of network features used in the kmeans algorithm is decreased to only two: assortativity and clustering coefficient. This group of features results in the maximum z-score while other arrangement of features resulted in z-score less than 10. 

However, it is easier to visualize a graph of points in two dimension the number of network features is reduced to one; assortativity. 

In a graph where if a word is an attribute will have a y value of 1 (true) otherwise it will have a value of 0 (false). Using clustering coefficient values (not normalized, but already in range of 0 to 1) to plot, most attribute words have high coefficient values and very few with low value. On the other hand, the non-attribute words have coefficient spread across the range of 0 to 1. 

If these values are plotted using histogram function with “Probability” enabled in Mathematica, there is a distinct line between the attributes and non-attributes and can be concluded that any word with a cluster coefficient greater than 0.55 will most likely be an attribute.

Note
---
The data was intentionally left out. The purpose of this repository is to demonstrate the project I worked on as a research assistant.