import pandas as pd

from segmentation_analyzer import SegmentationAnalyzer


def test_rfm_recency_scores_recent_users_higher():
    data = pd.DataFrame({
        "user_id": [1, 2, 3, 4, 5],
        "recency": [1, 2, 3, 4, 5],
        "frequency": [1, 2, 3, 4, 5],
        "monetary": [10, 20, 30, 40, 50],
    })

    result = SegmentationAnalyzer(data).rfm_segmentation(n_bins=5)
    scores = result.set_index("user_id")["R_score"]

    assert scores.loc[1] == 4
    assert scores.loc[5] == 0
