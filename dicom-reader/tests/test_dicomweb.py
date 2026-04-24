from dicom_reader.io.dicomweb import DicomWebClient, build_dicomweb_url


def test_build_url_escapes_segments_and_encodes_params():
    url = build_dicomweb_url(
        "https://example.com/dicom-web/",
        "studies",
        "1.2.3/4",
        PatientID="AB 42",
    )
    assert url == (
        "https://example.com/dicom-web/studies/1.2.3%2F4?PatientID=AB+42"
    )


def test_client_strips_trailing_slash():
    c = DicomWebClient("https://example.com/dicom-web/")
    assert c.base_url == "https://example.com/dicom-web"


def test_auth_header_prefers_token():
    c = DicomWebClient("https://example.com", token="abc")
    h = c._auth_headers("application/dicom+json")
    assert h["Authorization"] == "Bearer abc"
    assert h["Accept"] == "application/dicom+json"


def test_auth_tuple_used_for_basic():
    c = DicomWebClient("https://example.com", username="u", password="p")
    assert c._auth_tuple() == ("u", "p")
    assert c._auth_headers("x")["Accept"] == "x"
