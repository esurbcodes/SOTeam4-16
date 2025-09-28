from src.utils.github_link_finder import find_github_url_from_hf

def test_find_github_link_success(mocker):
    """Test that the function correctly finds a GitHub link in a fake README."""
    # Create a fake README content with a GitHub link
    fake_readme_content = b'This is our model. Find our code here: <a href="https://github.com/some-org/some-repo">GitHub</a>'

    # Mock the hf_hub_download to return a path to a fake file
    mock_download = mocker.patch('src.utils.github_link_finder.hf_hub_download', return_value='fake_readme.md')

    # Mock the open() function to return our fake content
    mocker.patch('builtins.open', mocker.mock_open(read_data=fake_readme_content))

    # The repo_id for which we are searching
    repo_id = "some-hf-model/some-model"

    found_url = find_github_url_from_hf(repo_id)

    assert found_url == "https://github.com/some-org/some-repo"
    mock_download.assert_called_with(repo_id=repo_id, filename="README.md")

def test_find_github_link_api_failure(mocker):
    """
    Tests that the link finder returns None if the hf_hub_download call fails.
    """
    # Mock the hf_hub_download function to raise an exception
    mocker.patch('src.utils.github_link_finder.hf_hub_download', side_effect=Exception("API Error"))

    # Call the function with a model ID
    found_url = find_github_url_from_hf("some/model")

    # Assert that the function returned None as expected
    assert found_url is None

def test_no_github_link_found(mocker):
    """Test that the function returns None when no GitHub link is present."""
    fake_readme_content = b'This is our model. No links here.'
    mocker.patch('src.utils.github_link_finder.hf_hub_download', return_value='fake_readme.md')
    mocker.patch('builtins.open', mocker.mock_open(read_data=fake_readme_content))

    found_url = find_github_url_from_hf("some-hf-model/some-model")

    assert found_url is None