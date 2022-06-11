import os

from dotenv import load_dotenv

load_dotenv()

# Dummy test function
def test_placeholder():
    assert True


def test_environment_secrets_working():
    """Assert secret key is not null"""
    try:
        os.environ["KAGGLE_KEY"]
    except KeyError:
        assert False

    assert True
