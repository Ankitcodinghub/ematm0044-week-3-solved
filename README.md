# ematm0044-week-3-solved
**TO GET THIS SOLUTION VISIT:** [EMATM0044 Week 3 Solved](https://www.ankitcodinghub.com/product/ematm0044-week-2-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;100278&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;0&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;0&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;0\/5 - (0 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;EMATM0044 Week 3 Solved&quot;,&quot;width&quot;:&quot;0&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 0px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            <span class="kksr-muted">Rate this product</span>
    </div>
    </div>
<div class="page" title="Page 1">
<div class="layoutArea">
<div class="column">
This week we will look at some unsupervised clustering algorithms. In this worksheet, we will start off by implementing k-means from scratch. We go on to look at using elbow plots to select a good value of k.

We go on to complare the behaviour of k-means with hierarchical clustering and Gaussian mixture models.

NB: if you find implementing the k-means algorithm form scratch challenging, don’t worry! Have a go, but the rest of the worksheet is not dependent on doing this, so you can skip that question and come back to it when you finish the rest of the worksheet.

2 K-means clustering

In the lecture, we saw that k-means is an unsupervised clustering algorithm. Recall that the algo- rithm runs as follows:

</div>
</div>
<div class="layoutArea">
<div class="column">
Given a set of datapoints drawn from Ω = Rn:

1. Randomly partition the set of datapoints into k sets.

<ol start="3">
<li>For each datapoint evaluate the squared Euclidean distance from each of the mean vectors e.g. ||⃗x − xˆP||2. Reallocate the datapoint to the partition set the mean of which it is closest to.</li>
<li>If the partition sets remain unchanged then stop. Else go to 2.</li>
</ol>
2.1 Implementing k-means [ ]:

</div>
</div>
<div class="layoutArea">
<div class="column">
2. For each set P calculate its mean vector:

􏰄 ∑ ⃗x ∈ P x 1 ∑ ⃗x ∈ P x i

</div>
<div class="column">
∑ ⃗x ∈ P x n 􏰅 …, |P|

</div>
</div>
<div class="layoutArea">
<div class="column">
xˆP = |P| ,…, |P|

</div>
</div>
<div class="section">
<div class="layoutArea">
<div class="column">
<pre># The following code creates some artificial data and plots it
</pre>
<pre>from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
</pre>
<pre>X, y = make_blobs(centers=3,n_samples=100, cluster_std=2, random_state=100)
</pre>
</div>
</div>
</div>
<div class="layoutArea">
<div class="column">
1

</div>
</div>
</div>
<div class="page" title="Page 2">
<div class="layoutArea">
<div class="column">
<pre>fig, ax = plt.subplots()
ax.scatter(X[:,0], X[:,1])
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
Implement a function kmeans that takes a value k and the data X, clusters the data, and returns the centroids and the labels of the data

[ ]:

</div>
</div>
<div class="section">
<div class="layoutArea">
<div class="column">
def kmeans(k, X):

# randomly assign labels to the data ##TODO##

<pre>    # set up a while loop that will run until the data labels no longer change
</pre>
while True:

# Calculate the centroids of the data ##TODO##

<pre>        # For each datapoint:
</pre>
for i, x in enumerate(X):

#Calculate the squared Euclidean distance to each centroid ##TODO##

<pre>            # Assign new labels based on distance to the centroid
            ##TODO##
</pre>
<pre>        # If all the new labels are equal to the old labels,
        # break out of the while loop
        ##TODO##
</pre>
<pre>        # Assign the values of the new labels to the variable labels
        ##TODO##
</pre>
<pre>    # return the centres and the labels.
</pre>
<pre>    return centres, labels
</pre>
<pre># Plot the centroids on the data. Are they as you would expect?
</pre>
<pre>centres, labels = kmeans(3, X)
ax.scatter(centres[:,0],centres[:,1])
fig
</pre>
</div>
</div>
</div>
<div class="layoutArea">
<div class="column">
2.2 Using the k-means function from scikit-learn

Scikit-learn has k-means built in. We import it using the command from sklearn.cluster import KMeans. Look at the documentation for KMeans (https://scikit- learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#). The KMeans estimator has 4 attributes. What are they?

The attributes are: 1. 2. 3. 4.

</div>
</div>
<div class="layoutArea">
<div class="column">
2

</div>
</div>
</div>
<div class="page" title="Page 3">
<div class="layoutArea">
<div class="column">
[ ]:

</div>
</div>
<div class="layoutArea">
<div class="column">
Which attribute would you use if you wanted to look at the labels assigned to the datapoints? What if you wanted to look at the centroids? What would you use the attribute inertia_ for?

2.2.1 Generating elbow plots

We will run k means over our toy dataset for multiple values of k and generate an elbow plot. To do this we can use the attribute inertia_. This attribute measures the within-cluster sum of squares, or the variance of each cluster, and the k means algorithm works to minimize this quantity. The within-cluster sum of squares is defined as:

k

∑ ∑ ||x − μj||2 j=1 x∈Pj

To generate the elbow plot, we run k means for values of k from 1 to 10, and plot the inertia at each point. If there is a clear ‘elbow’ in the plot, then this gives us the optimal value of k. Do you see a clear ‘elbow’ in the plot? If so, what is the optimal value of k?

</div>
</div>
<div class="section">
<div class="layoutArea">
<div class="column">
<pre># Import KMeans from sklearn.cluster
##TODO##
</pre>
# Optional: write your own function to calculate the inertia # (otherwise you can just use the attribute inertia_)

def inertia(X, labels, centroids):

<pre>    ##TODO## (Optional)
# Set up a variable to store the inertias
</pre>
inertias = []

<pre># Loop over values of k from 1 to 10
</pre>
for k in range(1, K+1):

# Instantiate the KMeans class with k clusters ##TODO##

<pre>    # Fit the model to the data
    ##TODO##
</pre>
<pre>    # Store the value of the inertia for this value of k
    ##TODO##
</pre>
<pre># Plot the elbow
</pre>
<pre>plt.figure()
plt.plot(range(1, K+1), inertias, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('The elbow method showing the optimal k')
</pre>
</div>
</div>
</div>
<div class="layoutArea">
<div class="column">
3

</div>
</div>
</div>
<div class="page" title="Page 4">
<div class="layoutArea">
<div class="column">
3 Clustering the iris dataset

The Iris flower data set or Fisher’s Iris data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in 1936. The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). There are four features corre- sponding to the length and the width of the sepals and petals, in centimetres. Typically, the Iris data set is used as a classification problem, but by considering only the 4-D input feature space we can also apply clustering algorithms to it.

[ ]:

</div>
</div>
<div class="section">
<div class="layoutArea">
<div class="column">
<pre># Import the iris dataset, and save the data into a variable X
# (take a look at the documentation here:
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.
</pre>
􏰈→html) ##TODO##

</div>
</div>
</div>
<div class="layoutArea">
<div class="column">
Let’s begin by assuming that since there are 3 types of iris, then there may be 3 clusters. Instantiate a k-means classifier with 3 clusters, and fit it to the data. Print out the centroids. You can visualise the resulting clusters by generating scatter plots projected on 2 dimensions. Try generating scatter plots for various combinations of features.

</div>
</div>
<div class="layoutArea">
<div class="column">
Extra question Generate one large plot with subplots for each combination of features. [ ]:

[ ]:

[ ]:

</div>
</div>
<div class="section">
<div class="layoutArea">
<div class="column">
<pre># Fit the iris dataset
##TODO##
</pre>
<pre># Make a scatter plot of the data on the first two axes
# Experiment with looking at different axes
##TODO##
</pre>
</div>
</div>
</div>
<div class="section">
<div class="layoutArea">
<div class="column">
# Optional: Plot all combinations of data in one large plot with subplots for␣ 􏰈→each combination of features

##TODO##

</div>
</div>
</div>
<div class="layoutArea">
<div class="column">
Generate an elbow plot for this data set. To what extent does this elbow plot support the assump- tion that there are three clusters present in the data?

</div>
</div>
<div class="section">
<div class="layoutArea">
<div class="column">
<pre># Generate an elbow plot for this dataset
##TODO##
</pre>
<pre># Plot the elbow
##TODO##
</pre>
</div>
</div>
</div>
<div class="layoutArea">
<div class="column">
4 Hierarchical clustering

In this question we investigate the use of hierarchical clustering on the Iris data set. SciPy (pro- nounced ‘Sigh Pie’) is a Python-based ecosystem of open-source software for mathematics, science,

</div>
</div>
<div class="layoutArea">
<div class="column">
4

</div>
</div>
</div>
<div class="page" title="Page 5">
<div class="layoutArea">
<div class="column">
and engineering. We start by importing packages dendrogram and linkage. from scipy.cluster.hierarchy import dendrogram, linkage

The following will generate a dendogram for the iris data set:

[ ]:

</div>
</div>
<div class="layoutArea">
<div class="column">
[ ]:

</div>
</div>
<div class="section">
<div class="layoutArea">
<div class="column">
<pre>linked = linkage(X, 'single')
labelList = range(len(X))
plt.figure(figsize=(10, 7))
dendrogram(linked,labels=labelList)
plt.show()
</pre>
</div>
</div>
</div>
<div class="layoutArea">
<div class="column">
Recall from the lectures that there are a number of ways of measuring the distance between clus- ters. For example:

</div>
</div>
<div class="layoutArea">
<div class="column">
• Minimum distance: d(S, T) = min{d(x, y) : x ∈ S, y ∈ T} • Average distance: d(S, T) = 1 ∑(x,y) d(x, y)

</div>
</div>
<div class="layoutArea">
<div class="column">
|S||T|

• Maximum distance: d(S, T) = max{d(x, y) : x ∈ S, y ∈ T}

</div>
</div>
<div class="layoutArea">
<div class="column">
• Centroid distance: d(S, T) = d( ∑x∈S x ∑y∈T y ) |S| |T|

The parameter ‘single’ in linkage refers to minimum distance. This can be change to ‘average’ for average distance, ‘complete’ for maximum distance and ‘centroid’ for centroid distance. Generate the dendogram for each of these cases. Comment on which metrics are most consistent with the assumption of 3 clusters in the iris data set.

</div>
</div>
<div class="layoutArea">
<div class="column">
[ ]:

</div>
</div>
<div class="layoutArea">
<div class="column">
<pre># Generate dendrograms for each distance metric
##TODO##
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
The metrics most consistent with the assumption of 3 clusters are:

5 Gaussian Mixture models

In this question we investigate the use of Gaussian clustering on the Iris data set.

</div>
</div>
<div class="section">
<div class="layoutArea">
<div class="column">
<pre>from sklearn.mixture import GaussianMixture as GMM
gmm = GMM(n_components=3)
gmm.fit(X)
</pre>
</div>
</div>
</div>
<div class="layoutArea">
<div class="column">
[ ]:

[ ]:

</div>
<div class="column">
We can extract the parameters for the learnt Gaussian distributions as follows:

</div>
</div>
<div class="layoutArea">
<div class="column">
<pre>print(gmm.means_)
print(gmm.covariances_)
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
How do the means for the three distributions compare with the centroids from a 3-cluster k-means on this dataset?

</div>
</div>
<div class="layoutArea">
<div class="column">
5

</div>
</div>
</div>
<div class="page" title="Page 6">
<div class="section">
<div class="layoutArea">
<div class="column">
<pre># Compare the means from the GMM clusters with the means from
# the k-means clusters
##TODO##
</pre>
</div>
</div>
</div>
<div class="layoutArea">
<div class="column">
[ ]:

[ ]: ##TODO##

Generate scatter plots for different 2-D combinations of the features.

[ ]: ##TODO##

</div>
</div>
<div class="layoutArea">
<div class="column">
Use the command print(gmm.weights_) to look at the weights for each distribution. What do these weights tell us about the composition of the three clusters?

</div>
</div>
<div class="layoutArea">
<div class="column">
6

</div>
</div>
</div>
