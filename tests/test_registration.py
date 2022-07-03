from tiatoolbox.tools.registration.wsi_registration import RegisterationConfig


def test_registeration_config():
    reg_obj = RegisterationConfig()
    assert reg_obj.units in [
        "power",
        "baseline",
        "mpp",
    ]
