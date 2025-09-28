from src.metrics.dataset_quality import metric
from src.utils.dataset_link_finder import find_dataset_url_from_hf

class FakeDatasetInfo:
    def __init__(self, cardData=None, downloads=0, likes=0):
        self.cardData = cardData
        self.downloads = downloads
        self.likes = likes

def test_dataset_quality_high_score(mocker):
    """Test a dataset with all quality indicators."""
    mocker.patch('src.metrics.dataset_quality.find_dataset_url_from_hf', return_value="https://huggingface.co/datasets/squad")
    mocker.patch('src.metrics.dataset_quality.dataset_info', return_value=FakeDatasetInfo(cardData={"dataset_card": True}, downloads=5000, likes=50))

    score, latency = metric({"name": "some/model"})
    assert score == 1.0 # 0.5 (card) + 0.3 (downloads) + 0.2 (likes)

def test_find_dataset_link_api_failure(mocker):
    """
    Tests that the link finder returns None if the hf_hub_download call fails.
    """
    # Mock the hf_hub_download function to raise an exception
    mocker.patch('src.utils.dataset_link_finder.hf_hub_download', side_effect=Exception("API Error"))

    # Call the function with a model ID
    found_url = find_dataset_url_from_hf("some/model")

    # Assert that the function returned None as expected
    assert found_url is None

def test_dataset_quality_no_link_found(mocker):
    """Test when no dataset link is found."""
    mocker.patch('src.metrics.dataset_quality.find_dataset_url_from_hf', return_value=None)

    score, latency = metric({"name": "some/model"})
    assert score == 0.0
    
