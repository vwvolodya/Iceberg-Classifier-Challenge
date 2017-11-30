import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from tqdm import tqdm as progressbar
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from cnn.dataset import IcebergDataset


def get_best_clusters(x):
    range_n_clusters = [2, 3, 4, 5, 6]

    for n_clusters in range_n_clusters:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(x) + (n_clusters + 1) * 10])

        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(x)

        silhouette_avg = silhouette_score(x, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        sample_silhouette_values = silhouette_samples(x, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            # noinspection PyUnresolvedReferences
            color = cm.spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # noinspection PyUnresolvedReferences
        colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(x[:, 0], x[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        centers = clusterer.cluster_centers_
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

        plt.show()


def inspect_angle():
    data = IcebergDataset("../data/all.npy", return_angle=True, mu_sigma=None)
    result = []
    for el in data:
        image = el["inputs"]
        angle = el["angle"]
        label = el["targets"]
        mu1, sigma1, med1, maximum_1, minimum_1, percentile75_1 = IcebergDataset.get_image_stat(image[0, :, :])
        mu2, sigma2, med2, maximum_2, minimum_2, percentile75_2 = IcebergDataset.get_image_stat(image[1, :, :])
        result.append((mu1, sigma1, med1, maximum_1, minimum_1, percentile75_1, mu2, sigma2,
                       med2, maximum_2, minimum_2, percentile75_2, angle[0], label[0]))
    new_frame = pd.DataFrame(result, columns=["mu1", "sigma1", "med1", "max1", "min1", "per75_1",
                                              "mu2", "sigma2", "med2", "max2", "min2", "per75_2", "angle", "label"])
    new_frame.to_csv("../data/stats.csv", index=False)
    print()


if __name__ == "__main__":
    data = IcebergDataset("../data/orig/test.json", mu_sigma=None, inference_only=True,
                          colormap="inferno", im_dir="../data/vis/test/cluster_1")
    X = np.array([i["inputs"].ravel() for i in data])
    # get_best_clusters(X)
    clusterer = KMeans(n_clusters=2, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    positives = []
    for i in progressbar(range(len(data))):
        if cluster_labels[i] == 1:
            # sample = data[i]
            # positives.append(sample["targets"][0])
            data.vis(i, prefix="C1_")

    print("Len", len(positives))
    print("Positives", sum(positives))

    # inspect_angle()
    print("Finished!")
