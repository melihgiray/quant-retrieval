from pathlib import Path


def test_container_healthcheck_uses_the_runtime_port():
    dockerfile = Path("Dockerfile").read_text()

    assert 'os.environ[\\"PORT\\"]' in dockerfile
    assert "127.0.0.1:7860/health" not in dockerfile
