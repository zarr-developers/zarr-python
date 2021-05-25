import pytest


@pytest.fixture(scope="session")
def azurite():
    import docker

    print("Starting azurite docker container")
    client = docker.from_env()
    azurite = client.containers.run(
        "mcr.microsoft.com/azure-storage/azurite",
        "azurite-blob --loose --blobHost 0.0.0.0",
        detach=True,
        ports={"10000": "10000"},
    )
    print("Successfully created azurite container...")
    yield azurite
    print("Teardown azurite docker container")
    azurite.stop()
