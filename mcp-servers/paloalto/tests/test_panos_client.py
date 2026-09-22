from panos_client import PanOSClient
 
 
def test_client_environment(monkeypatch):
 
    monkeypatch.setenv(
        "PANOS_HOST",
        "https://192.168.1.1",
    )
 
    monkeypatch.setenv(
        "PANOS_API_KEY",
        "dummy-key",
    )
 
    client = PanOSClient()
 
    assert client.host == "https://192.168.1.1"
    assert client.api_key == "dummy-key"