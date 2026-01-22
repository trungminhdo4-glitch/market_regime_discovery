import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances_argmin_min

def validate_temporal_stability(labels, min_segments=2):
    """Flag clusters that don't appear in multiple time segments"""
    unique_clusters = np.unique(labels)
    stable_clusters = []
    
    segment_size = len(labels) // 3
    if segment_size < 100:  # Minimum segment size
        return unique_clusters.tolist()
        
    segments = [
        labels[:segment_size],
        labels[segment_size:2*segment_size],
        labels[2*segment_size:]
    ]
    
    for cluster in unique_clusters:
        appearances = sum(1 for seg in segments if cluster in seg)
        if appearances >= min_segments:
            stable_clusters.append(cluster)
    
    return stable_clusters

def compute_transition_metrics(labels):
    """Calculate transition entropy and abnormal frequencies"""
    transitions = np.diff(labels)
    unique, counts = np.unique(transitions, return_counts=True)
    probs = counts / len(transitions)
    entropy = -np.sum(probs * np.log(probs + 1e-8))
    
    # Flag rare transitions (<1% frequency)
    rare_transitions = unique[counts < 0.01 * len(transitions)]
    
    return {
        "transition_entropy": float(entropy),
        "rare_transitions": rare_transitions.tolist()
    }

def enforce_min_cluster_size(Z, labels, gmm_model, min_size=500):
    unique, counts = np.unique(labels, return_counts=True)
    small_clusters = unique[counts < min_size]
    large_clusters = unique[counts >= min_size]

    if len(small_clusters) == 0:
        return labels, gmm_model

    new_labels = labels.copy()
    cluster_centers = gmm_model.means_

    for small in small_clusters:
        small_center = cluster_centers[small].reshape(1, -1)
        large_centers = cluster_centers[large_clusters]
        nearest_idx = pairwise_distances_argmin_min(small_center, large_centers)[0][0]
        nearest_large = large_clusters[nearest_idx]
        new_labels[labels == small] = nearest_large

    return new_labels, gmm_model

def fit_clustering_model(Z: np.ndarray, method: str, config: dict, random_state: int = 42):
    Z = Z.astype(np.float64)
    
    if method == "kmeans":
        n_clusters = config.get("n_clusters", 6)
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = model.fit_predict(Z)
    elif method == "gmm_bic":
        n_min = config["n_clusters_min"]
        n_max = config["n_clusters_max"]
        bic_scores = []
        models = []
        reg_covar = 1e-4
        
        for k in range(n_min, n_max + 1):
            try:
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type="full",
                    random_state=random_state,
                    n_init=3,
                    reg_covar=reg_covar
                )
                gmm.fit(Z)
                bic_scores.append(gmm.bic(Z))
                models.append(gmm)
            except ValueError as e:
                bic_scores.append(np.inf)
                models.append(None)
        
        valid_models = [(i, m) for i, m in enumerate(models) if m is not None]
        if not valid_models:
            raise RuntimeError("Alle GMM-Modelle sind fehlgeschlagen.")
        
        best_idx = np.argmin([bic_scores[i] for i, _ in valid_models])
        best_k = valid_models[best_idx][0] + n_min
        model = valid_models[best_idx][1]
        labels = model.predict(Z)
        
        min_size = config.get("min_cluster_size", 100)
        if min_size > 0:
            labels, model = enforce_min_cluster_size(Z, labels, model, min_size=min_size)
            
        # Validate temporal stability
        stability_cfg = config.get("temporal_stability", {})
        min_segments = stability_cfg.get("min_segments", 2)
        stable_clusters = validate_temporal_stability(labels, min_segments)
        
        # Mark unstable clusters (implementation detail)
        # For now, just log them
        unstable = set(np.unique(labels)) - set(stable_clusters)
        if unstable:
            print(f"⚠️  Instabile Cluster identifiziert: {unstable}")
            
    else:
        raise ValueError(f"Unbekannte Methode: {method}")
    return model, labels

def save_cluster_model(model, path: str):
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_cluster_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)