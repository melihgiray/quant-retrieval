import pytest

from quant_retrieval.retrieval.index_artifacts import verify_export_files


@pytest.mark.parametrize("names", [(), [], ["answer_ids.npy", "answer_ids.npy"],
                                  "answer_ids.npy", ["../answer_ids.npy"], [None]])
def test_verification_cannot_claim_success_for_ambiguous_payload_requests(tmp_path, names):
    with pytest.raises(ValueError, match="unique supported payload names"):
        verify_export_files(tmp_path, {}, names)


@pytest.mark.parametrize("manifest", [None, [], "legacy"])
def test_verification_requires_an_object_manifest(tmp_path, manifest):
    with pytest.raises(ValueError, match="manifest must be an object"):
        verify_export_files(tmp_path, manifest)
